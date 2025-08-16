# RunPod Serverless 기본 엔트리: handler(event) -> dict

import os
from typing import Any, Dict, List, Optional

from .instantmesh_runner import (
    INSTANTMESH_DIR,
    call_instantmesh,
    run_help,
)
from .utils_subproc import run_cmd

def _get(d: Dict, *keys, default=None):
    for k in keys:
        if isinstance(d, dict) and k in d and d[k] is not None:
            return d[k]
    return default

def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    # 입력/메타
    inp = event.get("input", {}) if isinstance(event, dict) else {}
    job_id = event.get("id") or event.get("job_id")
    diag = event.get("diag") or inp.get("diag") or {}

    # 0) 디버그: run.py -h 만 반환
    if _get(inp, "debug_help", default=False):
        help_res = run_help()
        return {
            "job_id": job_id,
            "diag": diag,
            "ok": help_res.get("ok", False),
            "stdout": help_res.get("stdout", ""),
            "stderr": help_res.get("stderr", ""),
            "returncode": help_res.get("returncode", 0),
        }

    # 0-1) 디버그: 임의 커맨드 실행(주의). 문자열 배열만 허용
    debug_cmd = _get(inp, "debug_cmd")
    if isinstance(debug_cmd, list) and all(isinstance(x, str) for x in debug_cmd):
        dbg = run_cmd(debug_cmd, cwd=INSTANTMESH_DIR, env=os.environ.copy())
        return {
            "job_id": job_id,
            "diag": diag,
            "ok": dbg.get("ok", False),
            "stdout": dbg.get("stdout", ""),
            "stderr": dbg.get("stderr", ""),
            "returncode": dbg.get("returncode", 0),
            "cmd": dbg.get("cmd"),
            "cwd": dbg.get("cwd"),
        }

    # 1) 입력 파싱(여러 키 지원)
    input_image = (
        _get(inp, "final_png")
        or _get(inp, "input_png")
        or _get(inp, "image")
        or _get(inp, "input")
        or _get(inp, "input_image")
        or _get(inp, "image_path")
        or _get(inp, "input_path")
    )

    # 필수 입력 체크
    if not input_image:
        return {
            "job_id": job_id,
            "diag": diag,
            "error_message": "Missing input image. Provide one of: final_png, input_png, image, input, input_image, image_path, input_path",
            "error_type": "BadRequest",
        }

    # 출력 디렉터리
    out_dir = (
        _get(inp, "out_dir")
        or _get(inp, "output_dir")
        or _get(inp, "output")
        or f"/tmp/jobs/{job_id or 'job'}/out"
    )
    os.makedirs(out_dir, exist_ok=True)

    # config / device
    cfg = _get(inp, "config")
    device = _get(inp, "device", default="cuda")

    # 환경 오버라이드 (선택)
    env_overrides = {}
    if "CUDA_VISIBLE_DEVICES" in inp:
        env_overrides["CUDA_VISIBLE_DEVICES"] = str(inp["CUDA_VISIBLE_DEVICES"])

    # 추가 인자 (필요 시)
    extra_args = _get(inp, "extra_args", default=None)
    if extra_args and not (isinstance(extra_args, list) and all(isinstance(x, str) for x in extra_args)):
        extra_args = None  # 잘못 들어오면 무시

    # 2) InstantMesh 실행(플래그/출력플래그 조합을 자동 시도)
    res = call_instantmesh(
        input_image=input_image,
        out_dir=out_dir,
        cfg_path=cfg,
        device=device,
        extra_args=extra_args,
        env_overrides=env_overrides if env_overrides else None,
    )

    # 3) 반환(성공/실패 모두 stdout/stderr 포함)
    result: Dict[str, Any] = {
        "job_id": job_id,
        "diag": diag,
        "config": res.get("config"),
        "config_tried": res.get("config_tried"),
        "tried": res.get("tried"),
        "out_dir": out_dir,
        "out_dir_listing": res.get("out_dir_listing"),
    }

    if res.get("ok"):
        result.update({
            "stdout": res.get("stdout", ""),
            "stderr": res.get("stderr", ""),
        })
        return result
    else:
        result.update({
            "error_message": f"InstantMesh failed (code={res.get('returncode')})",
            "error_type": "CalledProcessError",
            "stdout": res.get("stdout", ""),
            "stderr": res.get("stderr", ""),
        })
        return result