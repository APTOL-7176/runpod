FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    RP_VERBOSE=1 \
    BLENDER_VERSION=3.6.8 \
    BLENDER_DIR=/opt/blender \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    CUDA_VISIBLE_DEVICES=0

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip git wget curl ca-certificates git-lfs \
    libgl1 libglib2.0-0 libxrender1 libsm6 libxext6 \
    libgomp1 ffmpeg unzip xz-utils \
    && rm -rf /var/lib/apt/lists/* && git lfs install

# Blender (headless) with retry + fallback mirror
RUN set -eux; \
    mkdir -p "${BLENDER_DIR}"; \
    BL_MAJOR_MINOR="$(echo ${BLENDER_VERSION} | awk -F. '{print $1"."$2}')"; \
    BL_BASENAME="blender-${BLENDER_VERSION}-linux-x64"; \
    URL_PRIMARY="https://download.blender.org/release/Blender${BL_MAJOR_MINOR}/${BL_BASENAME}.tar.xz"; \
    URL_MIRROR="https://mirror.clarkson.edu/blender/release/Blender${BL_MAJOR_MINOR}/${BL_BASENAME}.tar.xz"; \
    for U in "$URL_PRIMARY" "$URL_MIRROR"; do \
      echo "Downloading: $U"; \
      if curl -fL --retry 5 --retry-delay 3 --connect-timeout 20 -o /tmp/blender.txz "$U"; then \
        break; \
      fi; \
      echo "Download failed from $U, trying next..."; \
    done; \
    ls -lh /tmp/blender.txz; \
    # Extract (needs xz-utils)
    tar -xJf /tmp/blender.txz -C "${BLENDER_DIR}" --strip-components=1; \
    ln -sf "${BLENDER_DIR}/blender" /usr/local/bin/blender; \
    blender -v

WORKDIR /app

# PyTorch CUDA 12.1
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 \
      torch torchvision torchaudio && \
    python3 -m pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu121 xformers==0.0.25.post1

# App deps
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# 3D repos
RUN git clone --depth=1 https://github.com/TencentARC/InstantMesh /app/repos/InstantMesh && \
    python3 -m pip install --no-cache-dir -r /app/repos/InstantMesh/requirements.txt
RUN git clone --depth=1 https://github.com/3DTopia/wonder3d /app/repos/Wonder3D || true && \
    if [ -f /app/repos/Wonder3D/requirements.txt ]; then \
      python3 -m pip install --no-cache-dir -r /app/repos/Wonder3D/requirements.txt; \
    fi

# Weights mount points
RUN mkdir -p /weights/instantmesh /weights/wonder3d

COPY scripts /app/scripts
COPY handler.py /app/handler.py
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

ENV BLENDER_USER_SCRIPTS=/root/.config/blender/${BLENDER_VERSION}/scripts
RUN mkdir -p ${BLENDER_USER_SCRIPTS}

CMD ["/bin/bash", "/app/start.sh"]