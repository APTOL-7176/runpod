import glob
import os
from typing import Dict, List, Optional, Tuple

from .utils_subproc import run_cmd

INSTANTMESH_DIR = "/app/repos/InstantMesh"

def _python_bin() -> str:
    # 우선 /usr/bin/python3 사용, 없으면 python3
    return "/usr/bin/python3" if os.path.exists("/usr/bin/python3") else "python3"

def run_help() -> Dict:
    cmd = [_python_bin(), "run.py", "-h"]
    return run_cmd(cmd, cwd=INSTANTMESH_DIR)

def find_config(preferred: Optional[str]) -> Tuple[Optional[str], List[str]]:
    tried: List[str] = []
    if preferred:
        p = preferred if os.path.isabs(preferred) else os.path.join(INSTANTMESH_DIR, preferred)
        tried.append(p)
        if os.path.isfile(p):
            return p, tried

    # 기본 후보
    defaults = [
        os.path.join(INSTANTMESH_DIR, "configs", "instant-mesh-large.yaml"),
        os.path.join(INSTANTMESH_DIR, "configs", "instant-mesh-small.yaml"),
    ]
    for p in defaults:
        tried.append(p)
        if os.path.isfile(p):
            return p, tried

    # 패턴 탐색
    for pattern in ("configs/*instant*large*.yaml", "configs/*.yaml"):
        for p in glob.glob(os.path.join(INSTANTMESH_DIR, pattern)):
            tried.append(p)
            if os.path.isfile(p):
                return p, tried

    return None, tried

def list_dir_snapshot(path: str, max_items: int = 200) -> List[str]:
    if not os.path.isdir(path):
        return []
    out: List[str] = []
    for root, dirs, files in os.walk(path):
        dirs.sort()
        files.sort()
        for d in dirs:
            out.append(os.path.relpath(os.path.join(root, d), path) + "/")
        for f in files:
            out.append(os.path.relpath(os.path.join(root, f), path))
        if len(out) > max_items:
            out.append(f"...(+{len(out)-max_items} more)")
            return out[: max_items + 1]
    return out

def call_instantmesh(
    input_image: str,
    out_dir: str,
    cfg_path: Optional[str] = None,
    device: Optional[str] = "cuda",
    extra_args: Optional[List[str]] = None,
    env_overrides: Optional[dict] = None,
) -> Dict:
    """
    - --image, --input, --input_image, --image_path, --input_path 순서로 시도
    - --output_dir, --output 순서로 시도
    - 성공하면 즉시 반환, 실패하면 모든 stderr를 모아 반환
    """
    os.makedirs(out_dir, exist_ok=True)

    cfg, cfg_tried = find_config(cfg_path)
    tried_commands: List[Dict] = []

    if not os.path.isfile(input_image):
        return {
            "ok": False,
            "stdout": "",
            "stderr": f"[call_instantmesh] input image not found: {input_image}",
            "returncode": 2,
            "cmd": [],
            "cwd": INSTANTMESH_DIR,
            "tried": tried_commands,
            "config": cfg,
            "config_tried": cfg_tried,
            "out_dir_listing": list_dir_snapshot(out_dir),
        }

    img_flags = ["image", "input", "input_image", "image_path", "input_path"]
    out_flags = ["output_dir", "output"]

    base = [_python_bin(), "run.py"]
    if cfg:
        base += ["--config", cfg]

    if extra_args:
        base += list(extra_args)

    # 환경
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if env_overrides:
        env.update(env_overrides)

    # 조합 시도
    all_errors: List[str] = []
    for img_flag in img_flags:
        for out_flag in out_flags:
            cmd = base + [f"--{img_flag}", input_image, f"--{out_flag}", out_dir]
            if device:
                cmd += ["--device", device]
            res = run_cmd(cmd, cwd=INSTANTMESH_DIR, env=env)
            tried_commands.append({"cmd": res.get("cmd"), "returncode": res.get("returncode")})
            if res["ok"]:
                res["tried"] = tried_commands
                res["config"] = cfg
                res["config_tried"] = cfg_tried
                res["out_dir_listing"] = list_dir_snapshot(out_dir)
                return res
            else:
                # 마지막 80줄만 축약 저장
                err = res.get("stderr", "")
                tail = "\n".join(err.splitlines()[-80:])
                all_errors.append(f"[{img_flag},{out_flag}] rc={res.get('returncode')}\n{tail}")

    # 전부 실패
    return {
        "ok": False,
        "stdout": "",
        "stderr": "\n\n".join(all_errors) if all_errors else "[call_instantmesh] unknown failure",
        "returncode": 1,
        "cmd": [],
        "cwd": INSTANTMESH_DIR,
        "tried": tried_commands,
        "config": cfg,
        "config_tried": cfg_tried,
        "out_dir_listing": list_dir_snapshot(out_dir),
    }