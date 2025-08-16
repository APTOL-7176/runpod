import os, io, base64, json, shutil, subprocess, tempfile, uuid, pathlib, sys, time
from typing import Dict, Any, List
import requests
from PIL import Image
from rembg import remove as rembg_remove
import runpod

# Ensure a GPU is visible by default
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# Hugging Face fast transfer guard
try:
    if os.environ.get("HF_HUB_ENABLE_HF_TRANSFER") is None:
        import importlib.util
        has_fast = importlib.util.find_spec("hf_transfer") is not None
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1" if has_fast else "0"
except Exception:
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

INSTANTMESH_REPO = "/app/repos/InstantMesh"
WONDER3D_REPO = "/app/repos/Wonder3D"
BLENDER_BIN = shutil.which("blender") or "/usr/local/bin/blender"

# Optional: set your checkpoint paths via env or mount to /weights
INSTANTMESH_CKPT = os.environ.get("INSTANTMESH_CKPT", "/weights/instantmesh")  # repo may auto-download; keep as hint
WONDER3D_CKPT = os.environ.get("WONDER3D_CKPT", "/weights/wonder3d")
INSTANTMESH_CONFIG = os.environ.get(
    "INSTANTMESH_CONFIG",
    os.path.join(INSTANTMESH_REPO, "configs", "instant-mesh-large.yaml")
)

def run_cmd(cmd: List[str], env: Dict[str, str] | None = None, cwd: str | None = None):
    print("[RUN]", " ".join(cmd), flush=True)
    try:
        res = subprocess.run(
            cmd,
            check=True,
            env=env,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if res.stdout:
            print("[RUN][OK]\n" + res.stdout, flush=True)
        return res
    except subprocess.CalledProcessError as e:
        print("[RUN][ERR]\n" + (e.stdout or ""), flush=True)
        raise

def gpu_diag() -> Dict[str, Any]:
    info = {}
    try:
        out = subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT, timeout=5).decode("utf-8", "ignore")
        info["nvidia_smi"] = out.splitlines()[0:5]
    except Exception as e:
        info["nvidia_smi_err"] = str(e)
    try:
        import torch
        info["torch_cuda_is_available"] = torch.cuda.is_available()
        info["torch_cuda_version"] = getattr(torch.version, "cuda", None)
        info["torch_device_count"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if torch.cuda.is_available():
            info["torch_device_0"] = torch.cuda.get_device_name(0)
    except Exception as e:
        info["torch_err"] = str(e)
    try:
        import onnxruntime as ort
        info["ort_providers"] = ort.get_available_providers()
    except Exception as e:
        info["ort_err"] = str(e)
    info["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    return info

def download_image(image_url: str, out_path: str) -> str:
    r = requests.get(image_url, timeout=60)
    r.raise_for_status()
    with open(out_path, "wb") as f:
        f.write(r.content)
    return out_path

def decode_b64(data: str, out_path: str) -> str:
    with open(out_path, "wb") as f:
        f.write(base64.b64decode(data))
    return out_path

def bg_remove(in_path: str, out_path: str) -> str:
    with Image.open(in_path) as im:
        im = im.convert("RGBA")
        out = rembg_remove(im)  # onnxruntime-gpu가 없으면 CPU 폴백
        out.save(out_path)
    return out_path

def _ensure_instantmesh_config() -> str:
    # If the default config path doesn't exist, pick a plausible one from configs/
    if os.path.isfile(INSTANTMESH_CONFIG):
        return INSTANTMESH_CONFIG
    cfg_dir = os.path.join(INSTANTMESH_REPO, "configs")
    candidates = []
    if os.path.isdir(cfg_dir):
        for fn in os.listdir(cfg_dir):
            if fn.endswith(".yaml") and "instant" in fn and "mesh" in fn:
                candidates.append(os.path.join(cfg_dir, fn))
    if not candidates:
        raise RuntimeError("InstantMesh config file not found. Set INSTANTMESH_CONFIG env to a valid YAML.")
    # choose the first candidate
    print(f"[InstantMesh] Using fallback config: {candidates[0]}", flush=True)
    return candidates[0]

def _list_meshes(root_dir: str) -> List[str]:
    out = []
    for dp, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith((".glb", ".gltf", ".obj")):
                out.append(os.path.join(dp, f))
    return out

def run_instantmesh(image_path: str, workdir: str) -> str:
    """
    Runs InstantMesh via run.py and copies the produced .glb/.obj to workdir/instantmesh.glb (or .obj)
    """
    cfg = _ensure_instantmesh_config()
    outputs_dir = os.path.join(INSTANTMESH_REPO, "outputs")
    before = set(_list_meshes(outputs_dir)) if os.path.isdir(outputs_dir) else set()

    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    # Some repos respect TORCH_DEVICE env; most rely on CUDA_VISIBLE_DEVICES
    cmd = [sys.executable, os.path.join(INSTANTMESH_REPO, "run.py"), cfg, image_path]
    run_cmd(cmd, env=env, cwd=INSTANTMESH_REPO)

    # find new meshes
    time.sleep(1.0)
    after = set(_list_meshes(outputs_dir))
    new_files = sorted(list(after - before), key=lambda p: os.path.getmtime(p)) if after else []
    if not new_files:
        # As a fallback also scan entire repo outputs for latest mesh
        all_meshes = _list_meshes(outputs_dir)
        if not all_meshes:
            raise RuntimeError("InstantMesh finished but no mesh was found under outputs/. Check logs above.")
        new_files = sorted(all_meshes, key=lambda p: os.path.getmtime(p))
    src_mesh = new_files[-1]
    ext = os.path.splitext(src_mesh)[1].lower()
    dst = os.path.join(workdir, f"instantmesh{ext}")
    shutil.copy2(src_mesh, dst)
    print(f"[InstantMesh] Picked output: {src_mesh} -> {dst}", flush=True)
    return dst

def run_wonder3d_refine(mesh_path: str, image_path: str, workdir: str) -> str:
    """
    Placeholder for Wonder3D refinement. Adjust CLI to the repo you pinned.
    We keep a conservative call and then search outputs or the given output path.
    """
    refined_mesh = os.path.join(workdir, "wonder3d_refined.glb")
    # Try a generic CLI; if your Wonder3D fork uses a different entry, update here.
    cmd = [
        sys.executable, os.path.join(WONDER3D_REPO, "infer.py"),
        "--input_mesh", mesh_path,
        "--ref_image", image_path,
        "--output", refined_mesh,
        "--ckpt", WONDER3D_CKPT,
        "--device", "cuda:0"
    ]
    try:
        run_cmd(cmd, env=os.environ.copy(), cwd=WONDER3D_REPO)
        if os.path.isfile(refined_mesh):
            return refined_mesh
    except Exception:
        # If the above CLI isn't supported, raise a clearer error
        raise RuntimeError("Wonder3D inference failed. Verify the correct CLI/entry script and options for your commit.")

    return refined_mesh

def run_blender_autorig(in_mesh: str, out_fbx: str):
    script = "/app/scripts/rigify_autorig.py"
    cmd = [BLENDER_BIN, "-b", "-P", script, "--", "--in", in_mesh, "--out", out_fbx]
    run_cmd(cmd)
    return out_fbx

def package_zip(files: Dict[str, str], out_zip: str) -> str:
    with tempfile.TemporaryDirectory() as td:
        for rel, src in files.items():
            dst = os.path.join(td, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
        shutil.make_archive(out_zip.replace(".zip",""), "zip", td)
    return out_zip

def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Input:
      {
        "image_url": "...",  // or "image_b64": "..."
        "mode": "quick" | "hq",
        "background_removal": true/false,
        "rigging": true/false,
        "output": "glb" | "fbx"
      }
    """
    data = event.get("input", {}) if isinstance(event, dict) else {}
    img_url = data.get("image_url")
    img_b64 = data.get("image_b64")
    mode = data.get("mode", "hq")
    do_bg = bool(data.get("background_removal", True))
    do_rig = bool(data.get("rigging", True))
    out_fmt = data.get("output", "fbx")

    # GPU diag
    diag = gpu_diag()
    print("[GPU DIAG]", json.dumps(diag, ensure_ascii=False), flush=True)

    os.makedirs("/tmp/jobs", exist_ok=True)
    jid = str(uuid.uuid4())[:8]
    job_dir = f"/tmp/jobs/{jid}"
    os.makedirs(job_dir, exist_ok=True)

    try:
        src_img = os.path.join(job_dir, "input.png")
        if img_url:
            download_image(img_url, src_img)
        elif img_b64:
            decode_b64(img_b64, src_img)
        else:
            return {"error": "Provide image_url or image_b64."}

        if do_bg:
            src_img = bg_remove(src_img, os.path.join(job_dir, "input_nobg.png"))

        # 1) InstantMesh
        base_mesh = run_instantmesh(src_img, job_dir)

        # 2) Wonder3D (HQ only)
        final_mesh = base_mesh
        if mode == "hq":
            final_mesh = run_wonder3d_refine(base_mesh, src_img, job_dir)

        # 3) Rigging (Blender headless)
        rigged_path = final_mesh
        if do_rig:
            out_fbx = os.path.join(job_dir, "rigged.fbx")
            rigged_path = run_blender_autorig(final_mesh, out_fbx)

        files = {"meshes/base" + pathlib.Path(base_mesh).suffix: base_mesh}
        if final_mesh != base_mesh:
            files["meshes/hq" + pathlib.Path(final_mesh).suffix] = final_mesh
        if do_rig and rigged_path.endswith(".fbx"):
            files["meshes/rigged.fbx"] = rigged_path

        out_zip = os.path.join(job_dir, "asset_bundle.zip")
        package_zip(files, out_zip)

        return {
            "job_id": jid,
            "mode": mode,
            "files": files,
            "zip": out_zip,
            "diag": diag
        }
    except Exception as e:
        return {
            "job_id": jid,
            "error_type": str(type(e)),
            "error_message": str(e),
            "diag": diag
        }

runpod.serverless.start({"handler": handler})