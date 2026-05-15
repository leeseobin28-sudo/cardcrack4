# cardcrack.py
# 콘크리트 균열 자동 진단 V9.3 (안정판 - WebRTC 제거)

import streamlit as st
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="균열 자동 진단 V9.3", layout="wide")
st.title("🔍 콘크리트 균열 자동 진단 V9.3")
st.caption("📏 거리 선택 → 가이드 미리보기로 거리 고정 → 카드 제거 후 촬영")

# ════════════════════════════════════════════════════════════════
# 상수
# ════════════════════════════════════════════════════════════════
CARD_W_MM = 85.60
CARD_H_MM = 53.98
CARD_ASPECT = CARD_W_MM / CARD_H_MM

DISTANCE_RATIO_MAP = {
    0.2: 0.80,
    0.4: 0.40,
    0.6: 0.266,
    0.8: 0.20,
    1.0: 0.16,
    1.2: 0.133,
}

@st.cache_resource
def load_yolo():
    return YOLO("bestcrack.pt")

# ════════════════════════════════════════════════════════════════
# 가이드박스 그리기
# ════════════════════════════════════════════════════════════════
def draw_guide_box(frame, guide_ratio):
    H, W = frame.shape[:2]
    cx, cy = W // 2, H // 2
    box_w = int(W * guide_ratio)
    box_h = int(box_w / CARD_ASPECT)
    x1, y1 = cx - box_w // 2, cy - box_h // 2
    x2, y2 = cx + box_w // 2, cy + box_h // 2

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (W, H), (0, 0, 0), -1)
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)

    color = (102, 255, 102)
    dash_len = max(5, int(box_w * 0.05))
    gap = max(5, int(box_w * 0.03))
    for i in range(x1, x2, dash_len + gap):
        cv2.line(frame, (i, y1), (min(i + dash_len, x2), y1), color, 3)
        cv2.line(frame, (i, y2), (min(i + dash_len, x2), y2), color, 3)
    for i in range(y1, y2, dash_len + gap):
        cv2.line(frame, (x1, i), (x1, min(i + dash_len, y2)), color, 3)
        cv2.line(frame, (x2, i), (x2, min(i + dash_len, y2)), color, 3)

    corner_len, corner_thick = 25, 6
    cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, corner_thick)
    cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, corner_thick)
    cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, corner_thick)
    cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, corner_thick)
    cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, corner_thick)
    cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, corner_thick)
    cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, corner_thick)
    cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, corner_thick)

    cv2.line(frame, (cx - 15, cy), (cx + 15, cy), (255, 200, 0), 3)
    cv2.line(frame, (cx, cy - 15), (cx, cy + 15), (255, 200, 0), 3)

    text = "Fit card here, then REMOVE card"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, 0.6, 2)
    text_y = max(y1 - 15, th + 10)
    cv2.rectangle(frame, (cx - tw // 2 - 8, text_y - th - 8),
                  (cx + tw // 2 + 8, text_y + 8), color, -1)
    cv2.putText(frame, text, (cx - tw // 2, text_y), font, 0.6, (0, 0, 0), 2)
    return frame

# ════════════════════════════════════════════════════════════════
# 사이드바
# ════════════════════════════════════════════════════════════════
st.sidebar.header("⚙️ 옵션")
selected_distance = st.sidebar.selectbox(
    "📏 촬영 거리 (m)",
    options=[0.2, 0.4, 0.6, 0.8, 1.0, 1.2],
    index=1
)
current_guide_ratio = DISTANCE_RATIO_MAP[selected_distance]
conf_thres = st.sidebar.slider("YOLO 신뢰도", 0.05, 0.9, 0.25, 0.05)

st.sidebar.markdown("---")
st.sidebar.markdown(f"""
**📋 사용법**
1. 거리 선택 (현재: **{selected_distance}m**)
2. 아래 미리보기처럼 **카드가 화면에서 같은 비율**이 되도록 거리 맞추기
3. 폰을 그 자리에 **고정**
4. **카드 치우기**
5. 촬영 → 자동 분석
""")

# 세션 상태
if "captured_img" not in st.session_state:
    st.session_state.captured_img = None

# ════════════════════════════════════════════════════════════════
# 가이드 미리보기 (정적)
# ════════════════════════════════════════════════════════════════
st.markdown("### 🎯 1단계: 거리 가이드 미리보기")
preview_w, preview_h = 800, 600
preview = np.full((preview_h, preview_w, 3), 180, dtype=np.uint8)
preview = draw_guide_box(preview, current_guide_ratio)
cv2.putText(preview, f"Distance: {selected_distance}m | Card box ratio: {current_guide_ratio*100:.1f}%",
            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 2)
st.image(preview, caption=f"🔍 {selected_distance}m 기준 - 카드가 이 박스 크기로 보이게 거리 조절",
         use_container_width=True)

st.warning("""
💡 **거리 맞추는 법**
1. 카드를 균열 옆에 댑니다
2. 휴대폰 화면 속 카드가 위 미리보기 박스와 **같은 비율**이 될 때까지 앞뒤로 이동
3. **그 자리에서 휴대폰 고정**
4. 카드를 치우고 아래 카메라로 촬영
""")

# ════════════════════════════════════════════════════════════════
# 입력 (카메라 OR 파일)
# ════════════════════════════════════════════════════════════════
st.markdown("### 📷 2단계: 촬영 또는 업로드")
tab1, tab2 = st.tabs(["📸 카메라 촬영", "📁 파일 업로드"])

with tab1:
    cam_img = st.camera_input("카드 치운 후 촬영하세요")
    if cam_img is not None:
        pil_img = Image.open(cam_img).convert("RGB")
        st.session_state.captured_img = np.array(pil_img)

with tab2:
    upload = st.file_uploader("균열 사진 (카드 없이)", type=["jpg", "jpeg", "png"])
    if upload is not None:
        pil_img = Image.open(upload).convert("RGB")
        st.session_state.captured_img = np.array(pil_img)

# ════════════════════════════════════════════════════════════════
# 분석
# ════════════════════════════════════════════════════════════════
if st.session_state.captured_img is None:
    st.info("👆 거리를 맞춘 후 카메라 촬영 또는 파일 업로드를 진행하세요")
    st.stop()

img_np = st.session_state.captured_img
H, W = img_np.shape[:2]

st.markdown("---")
st.markdown("### 🎯 3단계: 분석 결과")

if st.button("🔁 다시 촬영"):
    st.session_state.captured_img = None
    st.rerun()

box_w_px = W * current_guide_ratio
scale = CARD_W_MM / box_w_px

col1, col2 = st.columns(2)
with col1:
    st.markdown("**📷 입력 이미지**")
    st.image(img_np, use_container_width=True)
with col2:
    st.markdown("**📐 측정 기준**")
    st.write(f"📏 촬영 거리: **{selected_distance} m**")
    st.write(f"🔬 1 픽셀 = **{scale:.4f} mm**")
    st.write(f"🖼️ 이미지: **{W} × {H} px**")
    st.write(f"📦 가이드 비율: **{current_guide_ratio*100:.1f}%**")

with st.spinner("🔍 균열 탐지 중..."):
    yolo = load_yolo()
    results = yolo.predict(img_np, conf=conf_thres, verbose=False)

if not results or results[0].masks is None:
    st.error("❌ 균열을 찾지 못했습니다. 신뢰도를 낮춰보세요.")
    st.stop()

masks = results[0].masks.data.cpu().numpy()
full_mask = np.zeros((H, W), dtype=np.uint8)
for m in masks:
    mr = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
    full_mask = np.maximum(full_mask, (mr > 0.5).astype(np.uint8))

if full_mask.sum() == 0:
    st.warning("⚠️ 균열 마스크가 비어있습니다.")
    st.stop()

pixel_cnt = int(full_mask.sum())
area_cm2 = (pixel_cnt * scale * scale) / 100.0
dt = cv2.distanceTransform(full_mask, cv2.DIST_L2, 5)
max_width_mm = 2 * float(dt.max()) * scale

c1, c2, c3 = st.columns(3)
c1.metric("📏 mm/pixel", f"{scale:.4f}")
c2.metric("📐 균열 면적", f"{area_cm2:.2f} cm²")
c3.metric("📏 최대 균열 폭", f"{max_width_mm:.2f} mm")

# KCS 등급
if max_width_mm < 0.2:
    grade, emoji = "A (양호)", "✅"
elif max_width_mm < 0.3:
    grade, emoji = "B (관찰 필요)", "🟡"
elif max_width_mm < 1.0:
    grade, emoji = "C (보수 필요)", "🟠"
else:
    grade, emoji = "D (긴급 보수)", "🔴"
st.markdown(f"### {emoji} KCS 안전등급: **{grade}**")

overlay = img_np.copy()
overlay[full_mask > 0] = [255, 50, 50]
blended = cv2.addWeighted(img_np, 0.6, overlay, 0.4, 0)
st.image(blended, caption="🎯 검출 결과 (빨강: 균열)", use_container_width=True)

st.success(
    f"✅ 측정 완료 — 거리 {selected_distance}m 기반\n"
    f"📊 균열 픽셀: {pixel_cnt:,}개 | 1px = {scale:.4f}mm"
)