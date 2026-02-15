FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    espeak-ng \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install camel-tools

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV CAMELTOOLS_DATA=/opt/camel_tools
RUN mkdir -p /opt/camel_tools


COPY build/download_camel.py /app/build/download_camel.py

RUN python /app/build/download_camel.py

COPY . .

CMD ["python", "app.py"]
