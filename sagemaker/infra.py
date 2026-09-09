"""Idempotent VPC for the SageMaker x AgentCore recipe.

Creates (or adopts, by Name tag) everything the two managed services need to see each other:

* VPC ``10.20.0.0/16`` with DNS hostnames, one public subnet holding a NAT gateway (the
  training job pulls models/datasets and the agent image is pulled through it), and one
  private subnet in **every** AZ -- AgentCore VPC mode is only offered in specific AZ *IDs*
  and the list is not queryable, so give it every option and let ``create-agent-runtime``
  tell us which ones it accepts.
* S3 gateway endpoint on both route tables (SageMaker channels and checkpoints).
* Two security groups: ``miles-train`` (self-referencing all traffic, for Ray/NCCL between
  hosts) and ``miles-agentcore`` (egress only). ``miles-train`` admits TCP
  ``SESSION_PORTS`` from ``miles-agentcore`` -- that single rule is the whole exposure.
* An inline policy on the AgentCore execution role so the runtime may create ENIs.

Usage:
    python infra.py            # create/adopt, print summary JSON, write .infra.json
    python infra.py --show     # print what exists, create nothing
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import boto3

REGION = os.environ.get("AWS_REGION", "us-west-2")
NAME = "miles-agentcore"
VPC_CIDR = "10.20.0.0/16"
PUBLIC_SUBNET = ("10.20.0.0/24", "us-west-2a")
PRIVATE_SUBNETS = {
    "us-west-2a": "10.20.1.0/24",
    "us-west-2b": "10.20.2.0/24",
    "us-west-2c": "10.20.3.0/24",
    "us-west-2d": "10.20.4.0/24",
}
# The recipe's session-server range: --session-server-port 30000, up to 32 workers.
SESSION_PORTS = (30000, 30031)
AGENTCORE_ROLE = os.environ.get("MILES_AGENTCORE_ROLE", "MilesAgentCoreExecRole")
OUT_FILE = Path(__file__).with_name(".infra.json")

ec2 = boto3.client("ec2", region_name=REGION)
iam = boto3.client("iam", region_name=REGION)


def _tags(name: str) -> list[dict]:
    return [{"Key": "Name", "Value": name}, {"Key": "project", "Value": NAME}]


def _find(describe, key: str, name: str, **filters) -> dict | None:
    flt = [{"Name": "tag:Name", "Values": [name]}] + [{"Name": k, "Values": [v]} for k, v in filters.items()]
    items = describe(Filters=flt)[key]
    items = [i for i in items if i.get("State", "available") not in ("deleted", "deleting", "failed")]
    return items[0] if items else None


def ensure_vpc() -> str:
    if vpc := _find(ec2.describe_vpcs, "Vpcs", NAME):
        return vpc["VpcId"]
    vpc_id = ec2.create_vpc(CidrBlock=VPC_CIDR, TagSpecifications=[{"ResourceType": "vpc", "Tags": _tags(NAME)}])[
        "Vpc"
    ]["VpcId"]
    ec2.get_waiter("vpc_available").wait(VpcIds=[vpc_id])
    ec2.modify_vpc_attribute(VpcId=vpc_id, EnableDnsSupport={"Value": True})
    ec2.modify_vpc_attribute(VpcId=vpc_id, EnableDnsHostnames={"Value": True})
    return vpc_id


def ensure_igw(vpc_id: str) -> str:
    name = f"{NAME}-igw"
    if igw := _find(ec2.describe_internet_gateways, "InternetGateways", name):
        return igw["InternetGatewayId"]
    igw_id = ec2.create_internet_gateway(TagSpecifications=[{"ResourceType": "internet-gateway", "Tags": _tags(name)}])[
        "InternetGateway"
    ]["InternetGatewayId"]
    ec2.attach_internet_gateway(InternetGatewayId=igw_id, VpcId=vpc_id)
    return igw_id


def ensure_subnet(vpc_id: str, name: str, cidr: str, az: str, public: bool) -> str:
    if subnet := _find(ec2.describe_subnets, "Subnets", name, **{"vpc-id": vpc_id}):
        return subnet["SubnetId"]
    subnet_id = ec2.create_subnet(
        VpcId=vpc_id,
        CidrBlock=cidr,
        AvailabilityZone=az,
        TagSpecifications=[{"ResourceType": "subnet", "Tags": _tags(name)}],
    )["Subnet"]["SubnetId"]
    if public:
        ec2.modify_subnet_attribute(SubnetId=subnet_id, MapPublicIpOnLaunch={"Value": True})
    return subnet_id


def ensure_nat(public_subnet_id: str) -> str:
    name = f"{NAME}-nat"
    if nat := _find(ec2.describe_nat_gateways, "NatGateways", name):
        nat_id = nat["NatGatewayId"]
    else:
        eip = ec2.allocate_address(Domain="vpc", TagSpecifications=[{"ResourceType": "elastic-ip", "Tags": _tags(name)}])
        nat_id = ec2.create_nat_gateway(
            SubnetId=public_subnet_id,
            AllocationId=eip["AllocationId"],
            TagSpecifications=[{"ResourceType": "natgateway", "Tags": _tags(name)}],
        )["NatGateway"]["NatGatewayId"]
    print(f"waiting for NAT {nat_id} ...", file=sys.stderr)
    ec2.get_waiter("nat_gateway_available").wait(NatGatewayIds=[nat_id])
    return nat_id


def ensure_route_table(vpc_id: str, name: str, default_route: dict, subnet_ids: list[str]) -> str:
    if rt := _find(ec2.describe_route_tables, "RouteTables", name, **{"vpc-id": vpc_id}):
        rt_id = rt["RouteTableId"]
    else:
        rt_id = ec2.create_route_table(VpcId=vpc_id, TagSpecifications=[{"ResourceType": "route-table", "Tags": _tags(name)}])[
            "RouteTable"
        ]["RouteTableId"]
    routes = ec2.describe_route_tables(RouteTableIds=[rt_id])["RouteTables"][0]["Routes"]
    if not any(r.get("DestinationCidrBlock") == "0.0.0.0/0" for r in routes):
        ec2.create_route(RouteTableId=rt_id, DestinationCidrBlock="0.0.0.0/0", **default_route)
    associated = {a["SubnetId"] for a in ec2.describe_route_tables(RouteTableIds=[rt_id])["RouteTables"][0]["Associations"] if "SubnetId" in a}
    for subnet_id in subnet_ids:
        if subnet_id not in associated:
            ec2.associate_route_table(RouteTableId=rt_id, SubnetId=subnet_id)
    return rt_id


def ensure_s3_endpoint(vpc_id: str, route_table_ids: list[str]) -> str:
    service = f"com.amazonaws.{REGION}.s3"
    existing = ec2.describe_vpc_endpoints(
        Filters=[{"Name": "vpc-id", "Values": [vpc_id]}, {"Name": "service-name", "Values": [service]}]
    )["VpcEndpoints"]
    existing = [e for e in existing if e["State"] not in ("deleted", "deleting")]
    if existing:
        return existing[0]["VpcEndpointId"]
    return ec2.create_vpc_endpoint(
        VpcId=vpc_id,
        ServiceName=service,
        VpcEndpointType="Gateway",
        RouteTableIds=route_table_ids,
        TagSpecifications=[{"ResourceType": "vpc-endpoint", "Tags": _tags(f"{NAME}-s3")}],
    )["VpcEndpoint"]["VpcEndpointId"]


def ensure_sg(vpc_id: str, name: str, description: str) -> str:
    if sg := _find(ec2.describe_security_groups, "SecurityGroups", name, **{"vpc-id": vpc_id}):
        return sg["GroupId"]
    return ec2.create_security_group(
        GroupName=name,
        Description=description,
        VpcId=vpc_id,
        TagSpecifications=[{"ResourceType": "security-group", "Tags": _tags(name)}],
    )["GroupId"]


def _authorize(sg_id: str, permission: dict) -> None:
    try:
        ec2.authorize_security_group_ingress(GroupId=sg_id, IpPermissions=[permission])
    except ec2.exceptions.ClientError as exc:
        if exc.response["Error"]["Code"] != "InvalidPermission.Duplicate":
            raise


def ensure_security_groups(vpc_id: str) -> tuple[str, str]:
    train = ensure_sg(vpc_id, f"{NAME}-train", "Miles SageMaker training hosts")
    agent = ensure_sg(vpc_id, f"{NAME}-agentcore", "Bedrock AgentCore runtime ENIs")
    # Ray, NCCL, torch rendezvous between the hosts of one job.
    _authorize(train, {"IpProtocol": "-1", "UserIdGroupPairs": [{"GroupId": train}]})
    # The single deliberate exposure: session-server ports on the head, from the agent only.
    _authorize(
        train,
        {
            "IpProtocol": "tcp",
            "FromPort": SESSION_PORTS[0],
            "ToPort": SESSION_PORTS[1],
            "UserIdGroupPairs": [{"GroupId": agent, "Description": "AgentCore to Miles session servers"}],
        },
    )
    return train, agent


def ensure_agentcore_role() -> str:
    """The runtime's execution role: trust bedrock-agentcore, pull from ECR, write logs."""
    try:
        return iam.get_role(RoleName=AGENTCORE_ROLE)["Role"]["Arn"]
    except iam.exceptions.NoSuchEntityException:
        pass
    trust = {
        "Version": "2012-10-17",
        "Statement": [
            {"Effect": "Allow", "Principal": {"Service": "bedrock-agentcore.amazonaws.com"}, "Action": "sts:AssumeRole"}
        ],
    }
    arn = iam.create_role(RoleName=AGENTCORE_ROLE, AssumeRolePolicyDocument=json.dumps(trust))["Role"]["Arn"]
    for policy_arn in (
        "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly",
        "arn:aws:iam::aws:policy/CloudWatchLogsFullAccess",
    ):
        iam.attach_role_policy(RoleName=AGENTCORE_ROLE, PolicyArn=policy_arn)
    iam.get_waiter("role_exists").wait(RoleName=AGENTCORE_ROLE)
    return arn


def ensure_agentcore_role_eni_policy() -> None:
    """VPC mode makes the runtime create ENIs in our subnets under its execution role."""
    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "ec2:CreateNetworkInterface",
                    "ec2:CreateNetworkInterfacePermission",
                    "ec2:DeleteNetworkInterface",
                    "ec2:DescribeNetworkInterfaces",
                    "ec2:DescribeSubnets",
                    "ec2:DescribeSecurityGroups",
                    "ec2:DescribeVpcs",
                    "ec2:DescribeDhcpOptions",
                    "ec2:DescribeRouteTables",
                ],
                "Resource": "*",
            }
        ],
    }
    iam.put_role_policy(RoleName=AGENTCORE_ROLE, PolicyName=f"{NAME}-vpc-eni", PolicyDocument=json.dumps(policy))


def show() -> dict | None:
    vpc = _find(ec2.describe_vpcs, "Vpcs", NAME)
    if not vpc:
        return None
    vpc_id = vpc["VpcId"]
    subnets = ec2.describe_subnets(Filters=[{"Name": "vpc-id", "Values": [vpc_id]}])["Subnets"]
    sgs = ec2.describe_security_groups(Filters=[{"Name": "vpc-id", "Values": [vpc_id]}])["SecurityGroups"]
    return {
        "vpc_id": vpc_id,
        "subnets": [
            {
                "id": s["SubnetId"],
                "name": next((t["Value"] for t in s.get("Tags", []) if t["Key"] == "Name"), ""),
                "az": s["AvailabilityZone"],
                "az_id": s["AvailabilityZoneId"],
                "cidr": s["CidrBlock"],
            }
            for s in sorted(subnets, key=lambda s: s["CidrBlock"])
        ],
        "security_groups": {g["GroupName"]: g["GroupId"] for g in sgs},
    }


def create() -> dict:
    vpc_id = ensure_vpc()
    igw_id = ensure_igw(vpc_id)
    public_id = ensure_subnet(vpc_id, f"{NAME}-public-{PUBLIC_SUBNET[1]}", *PUBLIC_SUBNET, public=True)
    private_ids = [
        ensure_subnet(vpc_id, f"{NAME}-private-{az}", cidr, az, public=False) for az, cidr in PRIVATE_SUBNETS.items()
    ]
    public_rt = ensure_route_table(vpc_id, f"{NAME}-public-rt", {"GatewayId": igw_id}, [public_id])
    nat_id = ensure_nat(public_id)
    private_rt = ensure_route_table(vpc_id, f"{NAME}-private-rt", {"NatGatewayId": nat_id}, private_ids)
    ensure_s3_endpoint(vpc_id, [public_rt, private_rt])
    ensure_security_groups(vpc_id)
    role_arn = ensure_agentcore_role()
    ensure_agentcore_role_eni_policy()
    summary = show()
    assert summary is not None
    summary["agentcore_role_arn"] = role_arn
    summary["private_subnet_ids"] = private_ids
    summary["public_subnet_id"] = public_id
    summary["nat_gateway_id"] = nat_id
    summary["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    OUT_FILE.write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--show", action="store_true", help="describe only; create nothing")
    args = parser.parse_args()
    print(json.dumps(show() if args.show else create(), indent=2))


if __name__ == "__main__":
    main()
