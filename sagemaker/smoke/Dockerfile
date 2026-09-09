# Network-smoke image for the SageMaker x AgentCore path. No GPU, no Miles: it only proves
# that a port opened by algo-1 inside a SageMaker training job is reachable from (a) the
# other hosts of the same job and (b) an AgentCore runtime attached to the same VPC -- either
# directly (fake session server on :30000) or through the RFT front door (:30100, with the
# head's private DNS name published in Route 53).
#
#   docker build -t <ecr>/miles-sagemaker-smoke:latest -f sagemaker/smoke/Dockerfile .   (context = agentcore/)
FROM python:3.12-slim

WORKDIR /app
RUN pip install --no-cache-dir fastapi uvicorn httpx boto3

COPY fake_session_server.py /app/fake_session_server.py
COPY rft_front_door.py /app/rft_front_door.py
COPY sagemaker/smoke/smoke_entrypoint.py /app/smoke_entrypoint.py

# SageMaker invokes the image as `docker run <image> train`; the entrypoint ignores argv.
ENTRYPOINT ["python", "-u", "/app/smoke_entrypoint.py"]
