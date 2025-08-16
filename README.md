# Runpod: Weapon Removal + T-Pose + Genshin-like Anime (RGBA)

입력 스프라이트(≤400×400, RGBA)에서 무기를 확실히 제거하고, 캐릭터를 T-포즈로 보정한 뒤, **원신 느낌의 셀쉐이딩(픽셀풍 없음)** 으로 자연스럽게 인페인팅합니다. 결과는 투명 배경 RGBA로 반환합니다.

## 파이프라인 개요
- 무기 검출(앙상블): OWL-ViT 박스 + CLIPSeg 텍스트 세그 → 유니온 마스크 + 확장
- T-포즈: OpenPose ControlNet에 목표 T-포즈 스켈레톤을 직접 생성하여 강제
- 라인 보정: controlnet-aux LineartDetector로 **매끈한 라인아트 맵** 생성 (Canny보다 깨끗)
- 인페인팅: SD1.5 Inpainting + ControlNet(OpenPose/Lineart) 병행
- 반픽셀화(anti-pixel): Bicubic 업스케일 + Bilateral smoothing으로 픽셀 계단 제거
- 투명 배경: rembg로 재-매팅하여 RGBA PNG 출력

## 호출 입력 포맷
```json
{
  "image_url": "https://example.com/sprite.png",
  "score_threshold": 0.20,
  "mask_dilate": 12,
  "tpose_scope": "upper_body",
  "guidance_scale": 7.5,
  "steps": 34,
  "controlnet_scales": [1.35, 0.5],
  "out_long_side": 1024,
  "pixel_preserve": false,
  "prompt": "Genshin Impact style, anime cel shading, smooth soft gradients, clean thin lineart, high quality, detailed face, no weapons, natural relaxed hands, strict T-pose, character centered, soft vibrant colors, white studio lighting",
  "negative_prompt": "weapon, gun, sword, knife, rifle, spear, bow, axe, staff, grenade, bomb, pixelated, 8-bit, mosaic, dithering, voxel, lowres, jpeg artifacts, oversharp, deformed hands, extra fingers, missing fingers, text, watermark, harsh shadows, photorealistic"
}
```

## 팁
- 픽셀풍을 확실히 없애려면 `pixel_preserve`는 반드시 false로 두세요(기본 false).
- 더 강한 T-포즈 고정을 원하면 `controlnet_scales[0]`(openpose)을 1.5~1.8로 올려보세요.
- 라인이 너무 강하면 `controlnet_scales[1]`(lineart)을 0.3~0.5로 낮추세요.
- 무기 잔흔이 남으면 `mask_dilate`를 16~24로 올리거나 `steps`를 40 내외로 증가.