FROM python:3.11-slim AS builder
WORKDIR /app
COPY pyproject.toml README.md /app/
COPY src /app/src
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .

FROM python:3.11-slim
RUN useradd -m agentforge
WORKDIR /app
COPY --from=builder /usr/local /usr/local
USER agentforge
ENV AGENTFORGE_HOME=/data
ENTRYPOINT ["agentforge"]
