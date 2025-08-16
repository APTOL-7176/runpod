import subprocess
from typing import Dict, List, Optional, Union

def run_cmd(
    cmd: List[str],
    cwd: Optional[str] = None,
    env: Optional[dict] = None,
    timeout: Optional[int] = None,
) -> Dict[str, Union[bool, str, int, List[str], None]]:
    try:
        p = subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            timeout=timeout,
            capture_output=True,
            text=True,
            check=True,
        )
        return {
            "ok": True,
            "stdout": p.stdout,
            "stderr": p.stderr,
            "returncode": p.returncode,
            "cmd": cmd,
            "cwd": cwd,
        }
    except subprocess.CalledProcessError as e:
        return {
            "ok": False,
            "stdout": e.stdout or "",
            "stderr": e.stderr or "",
            "returncode": e.returncode,
            "cmd": cmd,
            "cwd": cwd,
        }
    except Exception as e:
        return {
            "ok": False,
            "stdout": "",
            "stderr": f"[run_cmd exception] {type(e).__name__}: {e}",
            "returncode": -1,
            "cmd": cmd,
            "cwd": cwd,
        }