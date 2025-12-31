FROM python:3.11-slim

RUN pip install --no-cache-dir --upgrade pip

COPY entrypoint.py /action/entrypoint.py

WORKDIR /github/workspace

ENTRYPOINT ["python", "/action/entrypoint.py"]
