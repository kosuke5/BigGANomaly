# GPU required
# https://hub.docker.com/r/pytorch/pytorch
FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

ENV DISPLAY host.docker.internal:0.0

COPY    requirements.txt .
RUN     pip install --upgrade pip
RUN     pip install -r requirements.txt