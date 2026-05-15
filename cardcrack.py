# cardcrack.py
# 콘크리트 균열 자동 진단 V9.2 - WebRTC + Camera Input 이중 모드

import threading
import streamlit as st
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO

# WebRTC는 선택적 import (실패해도 앱은 작동)
WEBRTC_AVAILABLE = False
try:
    import av
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
    WEBRTC_AVAILABLE = True
except ImportError as e:
    st.sidebar.warning(f"⚠️ WebRTC 사용 불가: {e}")

st.set_page_config(page_title="균열 자동 진단 V9.2", layout="wide")
st.title("🔍 콘크리트 균열 자동 진단 V9.2")
st.caption("📏 거리 선택 → 가이드박스로 거리 고정 → 카드 제거 후 촬영")

# ════════════════════════════════════════════════════════════════
# 상수 및 거리-비율 매핑
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
# 가이드박스 그리기 (정적/동적 공용)
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
    for (px, py, dx, dy) in [
        (x1, y1, 1, 1), (x2, y1, -1, 1),
        (x1, y2, 1, -1), (x2, y2, -1, -1),
    ]:
        cv2.line(frame, (px, py), (px + dx*corner_len, py), color, corner_thick)
        cv2.line(frame, (px, py), (px, py + dy*corner_len), color, corner_thick)

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

input_mode = st.sidebar.radio(
    "📥 입력 방식",
    ["📸 단일 촬영 (안정)", "📹 실시간 가이드 (WebRTC)", "📁 파일 업로드"],
    index=0,
    help="WebRTC가 안 되면 단일 촬영 모드를 사용하세요"
)

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
2. 균열 위에 카드를 대고, **녹색 박스에 카드 맞추기**
3. 폰을 그 자리에 **고정**
4. **카드 치우기**
5. 촬영 → 자동 분석
""")

# 세션 상태
if "captured_img" not in st.session_state:
    st.session_state.captured_img = None
if "captured_ratio" not in st.session_state:
    st.session_state.captured_ratio = current_guide_ratio
if "captured_distance" not in st.session_state:
    st.session_state.captured_distance = selected_distance

# ════════════════════════════════════════════════════════════════
# 모드별 입력 처리
# ════════════════════════════════════════════════════════════════

# ━━━━━━━━━━ 모드 1: 단일 촬영 (가장 안정) ━━━━━━━━━━
if input_mode.startswith("📸"):
    st.markdown("### 📸 단일 촬영 모드 (권장)")
    
    # 가이드박스 미리보기 (정적)
    preview_w, preview_h = 800, 600
    preview = np.full((preview_h, preview_w, 3), 200, dtype=np.uint8)
    preview = draw_guide_box(preview, current_guide_ratio)
    cv2.putText(preview, f"Distance: {selected_distance}m guide preview",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    st.image(preview, caption=f"🔍 {selected_distance}m 기준 가이드박스 미리보기",
             use_container_width=True)
    
    st.warning(f"""
    💡 **촬영 순서**
    1. 균열 옆에 카드를 댑니다
    2. 위 미리보기처럼 **카드가 화면에서 같은 비율**이 되도록 거리 조정
    3. **휴대폰 위치 유지** 한 채 카드 치움
    4. 아래 카메라로 촬영
    """)
    
    cam_img = st.camera_input("📷 카드 치운 후 촬영")
    if cam_img is not None:
        pil_img = Image.open(cam_img).convert("RGB")
        st.session_state.captured_img = np.array(pil_img)
        st.session_state.captured_ratio = current_guide_ratio
        st.session_state.captured_distance = selected_distance

# ━━━━━━━━━━ 모드 2: WebRTC 실시간 가이드 ━━━━━━━━━━
elif input_mode.startswith("📹"):
    if not WEBRTC_AVAILABLE:
        st.error("❌ WebRTC를 사용할 수 없습니다. '단일 촬영' 모드를 사용하세요.")
        st.stop()
    
    st.markdown("### 📹 실시간 가이드 모드 (WebRTC)")
    st.warning("⚠️ 촬영 직전 **카드를 화면에서 치운 후** 버튼을 누르세요!")

    class VideoProcessor:
        def __init__(self):
            self.lock = threading.Lock()
            self.latest_frame = None
            self.guide_ratio = current_guide_ratio

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            with self.lock:
                self.latest_frame = img.copy()
                cur = self.guide_ratio
            out = draw_guide_box(img, cur)
            return av.VideoFrame.from_ndarray(out, format="bgr24")

    RTC_CONFIG = RTCConfiguration({
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
        ]
    })
    
    ctx = webrtc_streamer(
        key="cardcrack_v92",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIG,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={
            "video": {"facingMode": {"ideal": "environment"}},
            "audio": False
        },
        async_processing=True,
    )
    
    if ctx.video_processor:
        ctx.video_processor.guide_ratio = current_guide_ratio

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        capture = st.button(
            "📸 촬영 및 분석",
            use_container_width=True,
            type="primary",
            disabled=(ctx.video_processor is None)
        )
    
    if capture and ctx.video_processor is not None:
        with ctx.video_processor.lock:
            if ctx.video_processor.latest_frame is not None:
                frame_rgb = cv2.cvtColor(
                    ctx.video_processor.latest_frame, cv2.COLOR_BGR2RGB
                )
                st.session_state.captured_img = frame_rgb
                st.session_state.captured_ratio = current_guide_ratio
                st.session_state.captured_distance = selected_distance

# ━━━━━━━━━━ 모드 3: 파일 업로드 ━━━━━━━━━━
else:
    st.markdown("### 📁 파일 업로드 모드")
    st.info(f"⚠️ {selected_distance}m 거리에서 촬영된 사진을 업로드하세요")
    
    upload = st.file_uploader("균열 사진", type=["jpg", "jpeg", "png"])
    if upload is not None:
        pil_img = Image.open(upload).convert("RGB")
        st.session_state.captured_img = np.array(pil_img)
        st.session_state.captured_ratio = current_guide_ratio
        st.session_state.captured_distance = selected_distance

# ════════════════════════════════════════════════════════════════
# 분석
# ════════════════════════════════════════════════════════════════
if st.session_state.captured_img is None:
    st.info("👆 위에서 입력 방법을 선택하고 사진을 준비하세요")
    st.stop()

img_np = st.session_state.captured_img
ratio_used = st.session_state.captured_ratio
distance_used = st.session_state.captured_distance
H, W = img_np.shape[:2]

st.markdown("---")
st.markdown("### 🎯 분석 결과")

if st.button("🔁 다시 촬영"):
    st.session_state.captured_img = None
    st.rerun()

box_w_px = W * ratio_used
scale = CARD_W_MM / box_w_px

col1, col2 = st.columns(2)
with col1:
    st.markdown("**📷 입력 이미지**")
    st.image(img_np, use_container_width=True)
with col2:
    st.markdown("**📐 측정 기준**")
    st.write(f"📏 촬영 거리: **{distance_used} m**")
    st.write(f"🔬 1 픽셀 = **{scale:.4f} mm**")
    st.write(f"🖼️ 이미지: **{W} × {H} px**")
    st.write(f"📦 가이드박스 너비: **{box_w_px:.0f} px** ({ratio_used*100:.1f}%)")

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

# 등급
if max_width_mm < 0.2:
    grade, color = "✅ 미세균열 (A)", "green"
elif max_width_mm < 0.3:
    grade, color = "🟡 경미균열 (B)", "orange"
elif max_width_mm < 1.0:
    grade, color = "🟠 중간균열 (C)", "red"
else:
    grade, color = "🔴 심각균열 (D)", "red"
st.markdown(f"### 안전 등급: :{color}[{grade}]")

overlay = img_np.copy()
overlay[full_mask > 0] = [255, 50, 50]
blended = cv2.addWeighted(img_np, 0.6, overlay, 0.4, 0)
st.image(blended, caption="🎯 검출 결과 (빨강: 균열)", use_container_width=True)

st.success(
    f"✅ 측정 완료 — 거리 {distance_used}m 기반\n"
    f"📊 균열 픽셀: {pixel_cnt:,}개 | 1px = {scale:.4f}mm"
)