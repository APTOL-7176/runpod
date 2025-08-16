#!/usr/bin/env bash
# 컨테이너/Pod 셸에서 바로 진단+실행 시도 스크립트
set -euo pipefail

IM_DIR="/app/repos/InstantMesh"
cd "$IM_DIR" || { echo "[ERR] InstantMesh 디렉터리 없음: $IM_DIR"; exit 1; }

IN="${1:-/tmp/in.png}"
OUT="${2:-/tmp/instantmesh_out}"
CFG="${CFG:-configs/instant-mesh-large.yaml}"

mkdir -p "$OUT"

echo "=== Python/CUDA 체크 ==="
python3 -V || true
nvidia-smi || true
python3 - <<'PY' || true
import torch
print("torch.cuda.is_available() =", torch.cuda.is_available())
print("torch.version.cuda       =", getattr(torch.version, "cuda", None))
PY

echo
echo "=== run.py -h 출력 ==="
python3 run.py -h || true

# config 경로 보정
if [ ! -f "$CFG" ]; then
  alt_cfg=$(ls configs/*instant*large*.yaml 2>/dev/null | head -n1 || true)
  if [ -n "${alt_cfg:-}" ]; then
    echo "[INFO] CFG 자동감지: $alt_cfg"
    CFG="$alt_cfg"
  else
    echo "[WARN] CFG를 찾지 못했습니다. 환경에 맞게 CFG를 지정하세요."
  fi
fi

# 입력 이미지 준비(없으면 샘플 다운로드)
if [ ! -f "$IN" ]; then
  echo "[INFO] 입력 이미지가 없어 샘플 다운로드합니다: $IN"
  curl -sSL -o "$IN" https://raw.githubusercontent.com/github/explore/main/topics/python/python.png
fi

echo
echo "=== 실행 시도 ==="
ARGS=(image input input_image image_path input_path)
OUTARGS=(output_dir output)

for arg in "${ARGS[@]}"; do
  for oarg in "${OUTARGS[@]}"; do
    echo
    echo "--- Try: --$arg + --$oarg ---"
    set +e
    python3 run.py --config "$CFG" --"$arg" "$IN" --"$oarg" "$OUT" --device cuda \
      >"$OUT/try_${arg}_${oarg}.out" 2>"$OUT/try_${arg}_${oarg}.err"
    rc=$?
    set -e
    echo "return code = $rc"
    if [ $rc -eq 0 ]; then
      echo "[OK] 성공: --$arg --$oarg"
      echo "[OUT] 결과 디렉터리: $OUT"
      exit 0
    else
      echo "[ERR] 실패: --$arg --$oarg (마지막 60줄 stderr)"
      tail -n 60 "$OUT/try_${arg}_${oarg}.err" || true
    fi
  done
done

echo
echo "[FAIL] 모든 플래그 조합 실패. 로그 확인:"
ls -l "$OUT"/try_* 2>/dev/null || true
echo "예) tail -n +1 $OUT/try_*.err | sed -n '1,200p'"
exit 2