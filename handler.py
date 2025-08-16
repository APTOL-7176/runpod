import io
import os
import base64
import json
import math
import requests
import numpy as np
from typing import List, Tuple, Dict, Any

import torch
from PIL import Image, ImageOps
import cv2

import runpod

from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
    CLIPSegProcessor,
    CLIPSegForImageSegmentation,
)

from diffusers import (
    StableDiffusionControlNetInpaintPipeline,
    ControlNetModel,
)
from rembg import remove as rembg_remove
from controlnet_aux import LineartDetector


# -----------------------------------
# Config
# -----------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# Models
OWL_VIT_ID = os.getenv("OWL_VIT_ID", "google/owlvit-base-patch32")
CLIPSEG_ID = os.getenv("CLIPSEG_ID", "CIDAS/clipseg-rd64-refined")
SD15_INPAINT_ID = os.getenv("SD15_INPAINT_ID", "runwayml/stable-diffusion-inpainting")
CN_OPENPOSE_ID = os.getenv("CN_OPENPOSE_ID", "lllyasviel/control_v11p_sd15_openpose")
CN_LINEART_ID = os.getenv("CN_LINEART_ID", "lllyasviel/control_v11p_sd15_lineart")

# Prompts (Genshin-like)
DEFAULT_PROMPT = os.getenv(
    "DEFAULT_PROMPT",
    "Genshin Impact style, anime cel shading, smooth soft gradients, clean thin lineart, "
    "high quality, detailed face, no weapons, natural relaxed hands, strict T-pose, "
    "character centered, soft vibrant colors, white studio lighting"
)
DEFAULT_NEGATIVE = os.getenv(
    "DEFAULT_NEGATIVE",
    "weapon, gun, sword, knife, rifle, spear, bow, axe, staff, grenade, bomb, "
    "pixelated, 8-bit, mosaic, dithering, voxel, lowres, jpeg artifacts, oversharp, "
    "deformed hands, extra fingers, missing fingers, text, watermark, harsh shadows, photorealistic"
)

# Weapon labels for OWL-ViT
WEAPON_LABELS = [
    "weapon", "gun", "pistol", "rifle", "machine gun", "knife", "dagger", "sword",
    "katana", "axe", "bow", "crossbow", "spear", "staff", "baton", "grenade", "bomb"
]

# -----------------------------------
# Lazy loads
# -----------------------------------
_owl_processor = None
_owl_detector = None
_clipseg_proc = None
_clipseg_model = None
_sd_pipe = None
_lineart_aux = None

def lazy_load_owl():
    global _owl_processor, _owl_detector
    if _owl_processor is None or _owl_detector is None:
        _owl_processor = AutoProcessor.from_pretrained(OWL_VIT_ID)
        _owl_detector = AutoModelForZeroShotObjectDetection.from_pretrained(OWL_VIT_ID).to(DEVICE)
    return _owl_processor, _owl_detector

def lazy_load_clipseg():
    global _clipseg_proc, _clipseg_model
    if _clipseg_proc is None or _clipseg_model is None:
        _clipseg_proc = CLIPSegProcessor.from_pretrained(CLIPSEG_ID)
        _clipseg_model = CLIPSegForImageSegmentation.from_pretrained(CLIPSEG_ID).to(DEVICE)
    return _clipseg_proc, _clipseg_model

def lazy_load_sd_pipe():
    global _sd_pipe
    if _sd_pipe is None:
        cn_openpose = ControlNetModel.from_pretrained(CN_OPENPOSE_ID, torch_dtype=DTYPE)
        cn_lineart = ControlNetModel.from_pretrained(CN_LINEART_ID, torch_dtype=DTYPE)
        _sd_pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            SD15_INPAINT_ID,
            controlnet=[cn_openpose, cn_lineart],
            torch_dtype=DTYPE
        ).to(DEVICE)
        _sd_pipe.enable_attention_slicing()
        if hasattr(_sd_pipe, "enable_vae_tiling"):
            _sd_pipe.enable_vae_tiling()
    return _sd_pipe

def lazy_load_lineart_aux():
    global _lineart_aux
    if _lineart_aux is None:
        # Coarse=False for clean, thin lines (Genshin-like)
        _lineart_aux = LineartDetector.from_pretrained("lllyasviel/Annotators")
    return _lineart_aux


# -----------------------------------
# Utils
# -----------------------------------
def load_image_from_url_or_b64(input_data: Dict[str, Any]) -> Image.Image:
    if "image_url" in input_data and input_data["image_url"]:
        resp = requests.get(input_data["image_url"], timeout=30)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGBA")
        return img
    if "image_b64" in input_data and input_data["image_b64"]:
        raw = base64.b64decode(input_data["image_b64"])
        img = Image.open(io.BytesIO(raw)).convert("RGBA")
        return img
    raise ValueError("Provide image_url or image_b64")

def ensure_rgba(img: Image.Image) -> Image.Image:
    return img if img.mode == "RGBA" else img.convert("RGBA")

def image_to_b64_png(pil_img: Image.Image) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def alpha_bbox(rgba: Image.Image, min_alpha: int = 5) -> Tuple[int, int, int, int]:
    a = np.array(rgba.getchannel("A"))
    ys, xs = np.where(a > min_alpha)
    if len(xs) == 0 or len(ys) == 0:
        return (0, 0, rgba.width, rgba.height)
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    return (int(x1), int(y1), int(x2), int(y2))

def upscale_bicubic(img: Image.Image, long_side: int = 1024) -> Image.Image:
    w, h = img.size
    scale = long_side / max(w, h)
    if scale <= 1.0:
        return img.copy()
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return img.resize((new_w, new_h), resample=Image.BICUBIC)

def downscale(img: Image.Image, target_wh: Tuple[int, int]) -> Image.Image:
    # LANCZOS for smooth downscale (non-pixel look)
    return img.resize(target_wh, resample=Image.LANCZOS)

def pil_to_cv(img: Image.Image) -> np.ndarray:
    arr = np.array(img.convert("RGB"))
    return arr[:, :, ::-1]  # RGB->BGR

def cv_to_pil(arr_bgr: np.ndarray) -> Image.Image:
    arr_rgb = arr_bgr[:, :, ::-1]
    return Image.fromarray(arr_rgb)

def normalize_to_uint8(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(np.float32)
    m -= m.min()
    mx = m.max()
    if mx > 0:
        m /= mx
    m = (m * 255.0).clip(0, 255).astype(np.uint8)
    return m

def anti_pixel_smooth(pil_img: Image.Image, d: int = 5, sigmaColor: int = 30, sigmaSpace: int = 30) -> Image.Image:
    """Bilateral smoothing to remove pixel stepping without blurring edges too much."""
    bgr = pil_to_cv(pil_img)
    sm = cv2.bilateralFilter(bgr, d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)
    return cv_to_pil(sm)

# -----------------------------------
# Detection / Masking
# -----------------------------------
def detect_weapons_owl(pil_img: Image.Image, score_thresh: float = 0.20) -> List[Dict[str, Any]]:
    processor, detector = lazy_load_owl()
    rgb = pil_img.convert("RGB")
    inputs = processor(text=WEAPON_LABELS, images=rgb, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = detector(**inputs)
    target_sizes = torch.tensor([rgb.size[::-1]]).to(DEVICE)  # (h, w)
    results = processor.post_process_object_detection(
        outputs=outputs, target_sizes=target_sizes, threshold=score_thresh
    )[0]
    boxes = []
    for score, label_id, box in zip(results["scores"], results["labels"], results["boxes"]):
        boxes.append({
            "label": WEAPON_LABELS[int(label_id)],
            "score": float(score),
            "box": [float(x) for x in box.tolist()]  # [x1, y1, x2, y2]
        })
    return boxes

def clipseg_mask(pil_img: Image.Image, prompts: List[str]) -> Image.Image:
    proc, model = lazy_load_clipseg()
    rgb = pil_img.convert("RGB")
    inputs = proc(text=prompts, images=[rgb] * len(prompts), padding="max_length", return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        logits = model(**inputs).logits  # (N, 1, H, W)
    logits = logits.squeeze(1)  # (N, H, W)
    agg = torch.max(logits, dim=0).values
    mask = torch.sigmoid(agg).detach().cpu().numpy()
    mask_u8 = normalize_to_uint8(mask)
    return Image.fromarray(mask_u8, mode="L")

def boxes_to_mask(boxes: List[Dict[str, Any]], size_hw: Tuple[int, int], dilate_px: int = 12) -> Image.Image:
    h, w = size_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    for b in boxes:
        x1, y1, x2, y2 = b["box"]
        x1 = max(0, int(math.floor(x1 - dilate_px)))
        y1 = max(0, int(math.floor(y1 - dilate_px)))
        x2 = min(w, int(math.ceil(x2 + dilate_px)))
        y2 = min(h, int(math.ceil(y2 + dilate_px)))
        mask[y1:y2, x1:x2] = 255
    return Image.fromarray(mask, mode="L")

def dilate_mask(mask_l: Image.Image, k: int = 7, iters: int = 1) -> Image.Image:
    m = np.array(mask_l)
    kernel = np.ones((k, k), np.uint8)
    d = cv2.dilate(m, kernel, iterations=iters)
    return Image.fromarray(d, mode="L")

# -----------------------------------
# T-pose maps & regions
# -----------------------------------
POSE_PAIRS = [
    (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13)
]
# 0-nose, 1-neck, 2-Rshoulder, 3-Relbow, 4-Rwrist, 5-Lshoulder, 6-Lelbow, 7-Lwrist,
# 8-Rhip, 9-Rknee, 10-Rankle, 11-Lhip, 12-Lknee, 13-Lankle
def draw_openpose_map_tpose(size_wh: Tuple[int, int], bbox: Tuple[int, int, int, int]) -> Image.Image:
    w, h = size_wh
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) // 2
    top = y1
    bottom = y2
    torso_h = max(1, (bottom - top))
    shoulder_y = y1 + int(0.30 * torso_h)
    hip_y = y1 + int(0.60 * torso_h)

    shoulder_span = int(0.8 * (x2 - x1))
    half_span = shoulder_span // 2
    kpts = {}
    kpts[1]  = (cx, shoulder_y)
    kpts[2]  = (cx + half_span//2, shoulder_y)
    kpts[5]  = (cx - half_span//2, shoulder_y)
    arm_len = int(0.55 * shoulder_span)
    kpts[3]  = (kpts[2][0] + arm_len//2, shoulder_y)
    kpts[4]  = (kpts[3][0] + arm_len//2, shoulder_y)
    kpts[6]  = (kpts[5][0] - arm_len//2, shoulder_y)
    kpts[7]  = (kpts[6][0] - arm_len//2, shoulder_y)
    kpts[8]  = (cx + int(0.12 * shoulder_span), hip_y)
    kpts[11] = (cx - int(0.12 * shoulder_span), hip_y)
    knee_y = y1 + int(0.85 * torso_h)
    ankle_y = y1 + int(1.05 * torso_h)
    kpts[9]  = (kpts[8][0], knee_y)
    kpts[10] = (kpts[8][0], ankle_y)
    kpts[12] = (kpts[11][0], knee_y)
    kpts[13] = (kpts[11][0], ankle_y)

    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    col_skel = (0, 255, 255)
    for a, b in POSE_PAIRS:
        if a in kpts and b in kpts:
            cv2.line(canvas, kpts[a], kpts[b], col_skel, thickness=6, lineType=cv2.LINE_AA)
    for _, pt in kpts.items():
        cv2.circle(canvas, pt, 6, (0, 0, 255), -1, lineType=cv2.LINE_AA)
    return Image.fromarray(canvas)

def build_upper_body_mask(size_wh: Tuple[int, int], bbox: Tuple[int, int, int, int], expand_ratio: float = 0.15) -> Image.Image:
    w, h = size_wh
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    y_mid = y1 + int(0.55 * height)
    ex = int(expand_ratio * width)
    mx1 = max(0, x1 - ex)
    mx2 = min(w, x2 + ex)
    my1 = max(0, y1 - ex)
    my2 = min(h, y_mid + ex)
    m = np.zeros((h, w), dtype=np.uint8)
    m[my1:my2, mx1:mx2] = 255
    return Image.fromarray(m, mode="L")

# -----------------------------------
# Control images
# -----------------------------------
def lineart_map_clean(pil_img: Image.Image, res_hint: int) -> Image.Image:
    det = lazy_load_lineart_aux()
    # Returns white background with black clean lines
    out = det(pil_img.convert("RGB"), coarse=False, resolution=res_hint)
    return out

# -----------------------------------
# Core pipeline
# -----------------------------------
def build_weapon_mask_ensemble(rgba_img: Image.Image, score_th: float, dilate_px: int) -> Tuple[Image.Image, List[Dict[str, Any]]]:
    boxes = detect_weapons_owl(rgba_img, score_thresh=score_th)
    seg_mask = clipseg_mask(rgba_img, prompts=["weapon", "gun", "sword", "knife", "rifle", "bow", "spear", "axe", "staff", "grenade"])
    box_mask = boxes_to_mask(boxes, size_hw=(rgba_img.height, rgba_img.width), dilate_px=dilate_px)
    seg = np.array(seg_mask)
    bx = np.array(box_mask)
    union = np.maximum(seg, bx)
    union = normalize_to_uint8(union)
    union = cv2.dilate(union, np.ones((5, 5), np.uint8), iterations=1)
    return Image.fromarray(union, mode="L"), boxes

def run_inpaint_tpose_anime(
    rgba_img: Image.Image,
    weapon_mask_l: Image.Image,
    prompt: str,
    negative_prompt: str,
    guidance_scale: float = 7.5,
    steps: int = 34,
    tpose_scope: str = "upper_body",
    controlnet_scales: Tuple[float, float] = (1.35, 0.5),  # stronger openpose, softer lineart
    out_long_side: int = 1024,
    anti_pixel: bool = True
) -> Image.Image:
    orig_w, orig_h = rgba_img.size

    # Upscale for stability (bicubic) and smooth pixel stepping
    up_rgba = upscale_bicubic(rgba_img, long_side=out_long_side)
    if anti_pixel:
        up_rgba = anti_pixel_smooth(up_rgba)

    up_mask = upscale_bicubic(weapon_mask_l, long_side=out_long_side).convert("L")

    # Build T-pose openpose map and optional upper-body mask
    bbox = alpha_bbox(up_rgba)
    openpose_map = draw_openpose_map_tpose(size_wh=up_rgba.size, bbox=bbox)
    lineart_map = lineart_map_clean(up_rgba, res_hint=out_long_side)

    if tpose_scope == "upper_body":
        ub_mask = build_upper_body_mask(size_wh=up_rgba.size, bbox=bbox, expand_ratio=0.18)
        up_mask_np = np.maximum(np.array(up_mask), np.array(ub_mask))
        up_mask = Image.fromarray(up_mask_np.astype(np.uint8), mode="L")

    up_mask = dilate_mask(up_mask, k=5, iters=1)

    pipe = lazy_load_sd_pipe()
    image_rgb = up_rgba.convert("RGB")
    mask_l = up_mask.convert("L")
    cn_scales = list(controlnet_scales)  # [openpose, lineart]

    with torch.autocast(device_type="cuda", dtype=DTYPE) if DEVICE == "cuda" else torch.no_grad():
        out = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image_rgb,
            mask_image=mask_l,
            controlnet_conditioning_scale=cn_scales,
            control_image=[openpose_map, lineart_map],
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
        ).images[0]

    # Re-matte to ensure transparent BG
    out_rgba = out.convert("RGB")
    out_rgba = Image.open(io.BytesIO(rembg_remove(out_rgba))).convert("RGBA")

    # Downscale to original size (smooth)
    out_rgba = downscale(out_rgba, (orig_w, orig_h))
    return out_rgba

# -----------------------------------
# Handler
# -----------------------------------
def handler(event):
    """
    Input JSON (event.input):
    {
      "image_url": "https://.../in.png"  // or "image_b64": "<base64>",
      "score_threshold": 0.20,
      "mask_dilate": 12,
      "prompt": "...", "negative_prompt": "...",
      "tpose_scope": "upper_body",       // "upper_body" | "full"
      "guidance_scale": 7.5,
      "steps": 34,
      "controlnet_scales": [1.35, 0.5],  // [openpose, lineart]
      "out_long_side": 1024,
      "pixel_preserve": false            // leave false for non-pixel look
    }
    """
    try:
        inp = event.get("input", {})
        img = load_image_from_url_or_b64(inp)
        img = ensure_rgba(img)

        # Ensure input size <= 400 (safety)
        if max(img.size) > 400:
            img = img.resize(
                (min(img.width, 400), min(img.height, 400)),
                resample=Image.BICUBIC
            )

        score_th = float(inp.get("score_threshold", 0.20))
        dilate_px = int(inp.get("mask_dilate", 12))
        prompt = inp.get("prompt", DEFAULT_PROMPT)
        negative = inp.get("negative_prompt", DEFAULT_NEGATIVE)
        tpose_scope = inp.get("tpose_scope", "upper_body")
        guidance_scale = float(inp.get("guidance_scale", 7.5))
        steps = int(inp.get("steps", 34))
        out_long_side = int(inp.get("out_long_side", 1024))
        cn_scales = inp.get("controlnet_scales", [1.35, 0.5])
        pixel_preserve = bool(inp.get("pixel_preserve", False))

        # 1) Weapon mask ensemble
        weapon_mask, boxes = build_weapon_mask_ensemble(img, score_th, dilate_px)

        # 2) Inpaint + T-pose + Genshin-like style
        out_rgba = run_inpaint_tpose_anime(
            rgba_img=img,
            weapon_mask_l=weapon_mask,
            prompt=prompt,
            negative_prompt=negative,
            guidance_scale=guidance_scale,
            steps=steps,
            tpose_scope=tpose_scope,
            controlnet_scales=(float(cn_scales[0]), float(cn_scales[1])),
            out_long_side=out_long_side,
            anti_pixel=(not pixel_preserve)
        )

        return {
            "found_weapons": len(boxes),
            "boxes": boxes,
            "image_b64": image_to_b64_png(out_rgba)
        }

    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})