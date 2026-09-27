FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir opencv-python-headless pillow numpy

WORKDIR /app
RUN mkdir -p /app/scratch
{video_step}
