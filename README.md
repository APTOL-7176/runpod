# Runpod: Weapon Removal + T-Pose + Genshin-like Anime (RGBA)

입력 스프라이트(≤400×400, RGBA)에서 무기를 확실히 제거하고, 캐릭터를 T-포즈로 보정한 뒤, **원신 느낌의 셀쉐이딩(픽셀풍 없음)** 으로 자연스럽게 인페인팅합니다. 결과는 투명 배경 RGBA로 반환합니다.

## 파이프라인 개요
- 무기 검출(앙상블): OWL-ViT 박스 + CLIPSeg 텍스트 세그 → 유니온 마스크 + 확장
- T-포즈: OpenPose ControlNet에 목표 T-포즈 스켈레톤을 직접 생성하여 강제
- 라인 보정: controlnet-aux LineartDetector로 **매끈한 라인아트 맵** 생성 (Canny보다 깨끗)
- 인페인팅: SD1.5 Inpainting + ControlNet(OpenPose/Lineart) 병행
- 반픽셀화(anti-pixel): Bicubic 업스케일 + Bilateral smoothing으로 픽셀 계단 제거
- 투명 배경: rembg로 재-매팅하여 RGBA PNG 출력

## 로컬 빌드
```bash
docker build -t ghcr.io/<owner>/runpod-weapon-tpose-genshin:latest .
```

## Runpod 배포
1) Runpod → Serverless → New Endpoint
- Container Image: `ghcr.io/<owner>/runpod-weapon-tpose-genshin:latest`
- GPU: A10G 24GB 권장
- Env(선택): `OWL_VIT_ID, CLIPSEG_ID, SD15_INPAINT_ID, CN_OPENPOSE_ID, CN_LINEART_ID, DEFAULT_PROMPT, DEFAULT_NEGATIVE`

2) 테스트 호출
```bash
curl -X POST https://api.runpod.ai/v2/<endpoint_id>/runsync \
  -H "Authorization: Bearer <runpod_key>" \
  -H "Content-Type: application/json" \
  -d '{"input": {"image_url": "https://example.com/sprite.png"}}'
```

## 팁
- 픽셀풍을 확실히 없애려면 `pixel_preserve=false` 유지(기본값)
- T-포즈 고정 강화: `controlnet_scales[0]`을 1.5~1.8로 상향
- 라인이 거치면: `controlnet_scales[1]`을 0.3~0.5로 조정
- 무기 잔흔: `mask_dilate` 16~24 또는 `steps` 40 내외

## 주의
- 일부 모델은 최초 실행 시 다운로드가 필요합니다(콜드스타트 지연). HuggingFace 토큰이 필요한 경우 Runpod 환경변수 `HUGGINGFACE_HUB_TOKEN`을 설정하세요.