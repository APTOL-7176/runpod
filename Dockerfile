FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    RP_VERBOSE=1 \
    BLENDER_VERSION=3.6.8 \
    BLENDER_DIR=/opt/blender \
    HF_HUB_ENABLE_HF_TRANSFER=0   # 기본은 끔(핸들러에서 자동 전환)

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip git wget ca-certificates git-lfs \
    libgl1 libglib2.0-0 libxrender1 libsm6 libxext6 \
    libgomp1 ffmpeg unzip && \
    rm -rf /var/lib/apt/lists/* && git lfs install

# Blender (headless)
RUN mkdir -p ${BLENDER_DIR} && \
    wget -q https://mirror.clarkson.edu/blender/release/Blender${BLENDER_VERSION%.*}/blender-${BLENDER_VERSION}-linux-x64.tar.xz -O /tmp/blender.txz && \
    tar -xJf /tmp/blender.txz -C ${BLENDER_DIR} --strip-components=1 && \
    ln -s ${BLENDER_DIR}/blender /usr/local/bin/blender

WORKDIR /app

# PyTorch CUDA 12.1
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 \
      torch torchvision torchaudio && \
    python3 -m pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu121 xformers==0.0.25.post1

# App deps
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Clone and set up 3D repos
# InstantMesh
RUN git clone --depth=1 https://github.com/TencentARC/InstantMesh /app/repos/InstantMesh && \
    python3 -m pip install --no-cache-dir -r /app/repos/InstantMesh/requirements.txt
# Wonder3D
RUN git clone --depth=1 https://github.com/3DTopia/wonder3d /app/repos/Wonder3D || true && \
    if [ -f /app/repos/Wonder3D/requirements.txt ]; then \
      python3 -m pip install --no-cache-dir -r /app/repos/Wonder3D/requirements.txt; \
    fi

# Download example checkpoints (placeholders: replace with your own if needed)
# You may need HF token for some weights; mount via env HF_TOKEN.
# ADD or RUN commands for ckpts can be added here if public.
RUN mkdir -p /weights/instantmesh /weights/wonder3d

# App code
COPY scripts /app/scripts
COPY handler.py /app/handler.py

# Ensure Blender addons (Rigify, glTF) can be enabled in headless
ENV BLENDER_USER_SCRIPTS=/root/.config/blender/${BLENDER_VERSION}/scripts
RUN mkdir -p ${BLENDER_USER_SCRIPTS}

CMD ["python3", "-u", "handler.py"]