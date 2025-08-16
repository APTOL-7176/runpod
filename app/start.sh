#!/bin/bash
set -euo pipefail

echo "[BOOT] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "[BOOT] Checking overlay at /vol/overlay ..."
if [ -d "/vol/overlay" ]; then
  shopt -s dotglob
  if compgen -G "/vol/overlay/*" > /dev/null; then
    echo "[BOOT] Applying overlay files to /app ..."
    cp -r /vol/overlay/* /app/
  else
    echo "[BOOT] /vol/overlay is empty."
  fi
else
  echo "[BOOT] /vol/overlay not found (skip)."
fi

echo "[BOOT] Launching handler ..."
exec python3 -u /app/handler.py