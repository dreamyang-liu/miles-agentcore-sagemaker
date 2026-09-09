# AgentCore Runtime requires linux/arm64 for the default microVM compute type. The platform
# is supplied by the build command rather than pinned here, so this file stays portable and
# buildx does not warn about a constant --platform:
#   docker buildx build --platform linux/arm64 -t <ecr>/miles-agentcore-math:latest --push .
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY agent.py .

# Fixed by the Runtime HTTP contract: 0.0.0.0:8080 serving /invocations and /ping.
EXPOSE 8080
CMD ["python", "agent.py"]
