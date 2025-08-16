import os, io, base64, json, shutil, subprocess, tempfile, uuid, pathlib, sys
from typing import Dict, Any
import requests
from PIL import Image
from rembg import remove as rembg_remove
import runpod

INSTANTMESH_REPO = "/app/repos/InstantMesh"
WONDER3D_REPO = "/app/repos/Wonder3D"
BLENDER_BIN = shutil.which("blender") or "/usr/local/bin/blender"

# Optional: set your checkpoint paths via env or mount to /weights
INSTANTMESH_CKPT = os.environ.get("INSTANTMESH_CKPT", "/weights/instantmesh")  # folder/ckpt as repo expects
WONDER3D_CKPT = os.environ.get("WONDER3D_CKPT", "/weights/wonder3d")

def download_image(image_url: str, out_path: str) -> str:
    r = requests.get(image_url, timeout=30)
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
        out = rembg_remove(im)
        out.save(out_path)
    return out_path

def run_instantmesh(image_path: str, workdir: str) -> str:
    # NOTE: Adjust CLI according to the repo version you use.
    # This wrapper assumes an inference script that outputs a GLB/OBJ mesh.
    out_mesh = os.path.join(workdir, "instantmesh.glb")
    cmd = [
        sys.executable, os.path.join(INSTANTMESH_REPO, "demo.py"),
        "--input", image_path,
        "--output", out_mesh,
        "--device", "cuda",
        "--ckpt", INSTANTMESH_CKPT
    ]
    print("[InstantMesh] Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    return out_mesh

def run_wonder3d_refine(mesh_path: str, image_path: str, workdir: str) -> str:
    # NOTE: Wonder3D pipelines vary; pick the appropriate entry (multiview + texopt).
    # This command is a placeholder; replace with the repo's current CLI.
    refined_mesh = os.path.join(workdir, "wonder3d_refined.glb")
    cmd = [
        sys.executable, os.path.join(WONDER3D_REPO, "infer.py"),
        "--input_mesh", mesh_path,
        "--ref_image", image_path,
        "--output", refined_mesh,
        "--ckpt", WONDER3D_CKPT,
        "--device", "cuda"
    ]
    print("[Wonder3D] Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    return refined_mesh

def run_blender_autorig(in_mesh: str, out_fbx: str):
    script = "/app/scripts/rigify_autorig.py"
    cmd = [BLENDER_BIN, "-b", "-P", script, "--", "--in", in_mesh, "--out", out_fbx]
    print("[Blender] Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
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

    os.makedirs("/tmp/jobs", exist_ok=True)
    jid = str(uuid.uuid4())[:8]
    job_dir = f"/tmp/jobs/{jid}"
    os.makedirs(job_dir, exist_ok=True)

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

    # 4) Package
    # Always include the unrigged mesh as well.
    files = {
        "meshes/base.glb": base_mesh,
    }
    if final_mesh != base_mesh:
        files["meshes/hq.glb"] = final_mesh
    if do_rig and rigged_path.endswith(".fbx"):
        files["meshes/rigged.fbx"] = rigged_path

    out_zip = os.path.join(job_dir, "asset_bundle.zip")
    package_zip(files, out_zip)

    return {
        "job_id": jid,
        "mode": mode,
        "files": files,
        "zip": out_zip
    }

runpod.serverless.start({"handler": handler})