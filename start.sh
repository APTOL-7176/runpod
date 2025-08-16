#!/bin/bash
set -euo pipefail

echo "[BOOT] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "[BOOT] Applying overlay from /vol/overlay (if any) ..."

if [ -d "/vol/overlay" ]; then
  shopt -s dotglob nullglob
  FILES=(/vol/overlay/*)
  if [ ${#FILES[@]} -gt 0 ]; then
    cp -rf /vol/overlay/* /app/
    echo "[BOOT] Overlay applied: ${#FILES[@]} file(s) copied to /app"
  else
    echo "[BOOT] /vol/overlay is empty. Skipping."
  fi
else
  echo "[BOOT] /vol/overlay not found. Skipping."
fi

echo "[BOOT] Launching handler ..."
exec python3 -u /app/handler.py