# Stage 1
FROM python:3.8-bullseye AS builder

COPY requirements.txt /requirements.txt
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir --user  \
    --no-warn-script-location -r /requirements.txt  \
    --extra-index-url https://download.pytorch.org/whl/cu116 && \
    rm -rf /requirements.txt

# Stage 2
FROM python:3.8-slim-bullseye

RUN apt-get update && apt-get install -y --no-install-recommends\
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /root/.local /root/.local
ENV PATH=$PATH:/root/.local

WORKDIR /workspace
CMD ["/bin/bash"]