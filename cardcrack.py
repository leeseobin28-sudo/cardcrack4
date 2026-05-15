# cardcrack.py
# 콘크리트 균열 자동 진단 V9.0 - Streamlit Cloud 안정 버전
# Python 3.14 환경에서도 작동 (streamlit-webrtc 사용 안 함)
# 카드 자동 검출 + 원스텝 분석

import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO
import base64
import io

# 이 부분이 수정되어야 합니다!
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration 

# ... 나머지 코드
# ... (이하 작성자님의 원래 코드) ...

st.set_page_config(page_title="균열 자동 진단 V9", layout="wide")
st.title("🔍 콘크리트 균열 자동 진단 V9")
st.caption("📏 촬영 거리 지정 → 가이드박스로 거리 고정 → 카드 제거 후 촬영")

# ════════════════════════════════════════════════════════════════
# 상수 및 거리-비율 매핑
# ════════════════════════════════════════════════════════════════
CARD_W_MM = 85.60
CARD_H_MM = 53.98
CARD_ASPECT = CARD_W_MM / CARD_H_MM

# 거리에 따른 가이드박스의 화면 너비 차지 비율 
# (기준: 0.4m에서 화면의 40%를 차지한다고 가정, 거리에 반비례)
DISTANCE_RATIO_MAP = {
    0.2: 0.80,  # 20cm (가장 큼)
    0.4: 0.40,  # 40cm
    0.6: 0.266, # 60cm
    0.8: 0.20,  # 80cm
    1.0: 0.16,  # 100cm
    1.2: 0.133  # 120cm (가장 작음)
}

# ════════════════════════════════════════════════════════════════
# YOLO 모델 로드
# ════════════════════════════════════════════════════════════════
@st.cache_resource
def load_yolo():
    return YOLO("bestcrack.pt")

# ════════════════════════════════════════════════════════════════
# 가이드박스 오버레이 그리기 (비율에 따라 가변)
# ════════════════════════════════════════════════════════════════
def draw_guide_box(frame, guide_ratio):
    H, W = frame.shape[:2]
    cx, cy = W // 2, H // 2

    # 가이드박스 크기 계산
    box_w = int(W * guide_ratio)
    box_h = int(box_w / CARD_ASPECT)

    x1, y1 = cx - box_w // 2, cy - box_h // 2
    x2, y2 = cx + box_w // 2, cy + box_h // 2

    # 반투명 어두운 배경 (가이드박스 외부)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (W, H), (0, 0, 0), -1)
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)
    
    # 가이드박스 안은 다시 원본으로
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

    # 점선 박스
    color = (102, 255, 102)  # 녹색
    dash_len = max(5, int(box_w * 0.05))
    gap = max(5, int(box_w * 0.03))
    for i in range(x1, x2, dash_len + gap):
        cv2.line(frame, (i, y1), (min(i + dash_len, x2), y1), color, 3)
        cv2.line(frame, (i, y2), (min(i + dash_len, x2), y2), color, 3)
    for i in range(y1, y2, dash_len + gap):
        cv2.line(frame, (x1, i), (x1, min(i + dash_len, y2)), color, 3)
        cv2.line(frame, (x2, i), (x2, min(i + dash_len, y2)), color, 3)

    # 중앙 십자 (파란색)
    cv2.line(frame, (cx - 15, cy), (cx + 15, cy), (255, 200, 0), 3)
    cv2.line(frame, (cx, cy - 15), (cx, cy + 15), (255, 200, 0), 3)

    # 안내 텍스트
    text = "Fit card here, then REMOVE card to shoot"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, 0.6, 2)
    text_y = max(y1 - 15, th + 10)
    cv2.rectangle(frame, (cx - tw // 2 - 8, text_y - th - 8),
                  (cx + tw // 2 + 8, text_y + 8), color, -1)
    cv2.putText(frame, text, (cx - tw // 2, text_y), font, 0.6, (0, 0, 0), 2)

    return frame

# ════════════════════════════════════════════════════════════════
# 비디오 프로세서 (실시간 가이드박스 변경 적용)
# ════════════════════════════════════════════════════════════════
class VideoProcessor:
    def __init__(self):
        self.lock = threading.Lock()
        self.latest_frame = None
        self.guide_ratio = 0.40  # 기본값 0.4m

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        with self.lock:
            self.latest_frame = img.copy()
            current_ratio = self.guide_ratio
        
        img_with_guide = draw_guide_box(img, current_ratio)
        return av.VideoFrame.from_ndarray(img_with_guide, format="bgr24")

# ════════════════════════════════════════════════════════════════
# 사이드바 설정
# ════════════════════════════════════════════════════════════════
st.sidebar.header("⚙️ 옵션")
selected_distance = st.sidebar.selectbox(
    "📏 촬영 거리 선택 (m)", 
    options=[0.2, 0.4, 0.6, 0.8, 1.0, 1.2],
    index=1  # 기본값 0.4m
)
current_guide_ratio = DISTANCE_RATIO_MAP[selected_distance]
conf_thres = st.sidebar.slider("YOLO 신뢰도", 0.05, 0.9, 0.25, 0.05)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**📋 올바른 사용법**
1. 원하는 촬영 거리를 선택합니다.
2. 벽에 신용카드를 대고, 화면의 **녹색 가이드박스**에 카드 크기가 딱 맞도록 물러납니다.
3. 카메라(스마트폰) 위치를 **그대로 고정**합니다.
4. 벽에서 **카드를 치웁니다.** (화면에 균열만 보이게)
5. **📸 촬영 및 분석** 버튼을 누릅니다.
""")

# 세션 상태 초기화
if "captured_img" not in st.session_state:
    st.session_state.captured_img = None
if "analyze" not in st.session_state:
    st.session_state.analyze = False
if "captured_ratio" not in st.session_state:
    st.session_state.captured_ratio = current_guide_ratio

# ════════════════════════════════════════════════════════════════
# 1. 라이브 카메라
# ════════════════════════════════════════════════════════════════
st.markdown("### 📷 카메라")
st.warning("⚠️ 촬영 직전에 **반드시 카드를 화면에서 치운 후** 촬영 버튼을 누르세요!")

RTC_CONFIG = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

ctx = webrtc_streamer(
    key="cardcrack",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIG,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={
        "video": {"facingMode": {"ideal": "environment"}},
        "audio": False
    },
    async_processing=True,
)

# 사이드바에서 선택한 비율을 실시간 영상 프로세서에 전달
if ctx.video_processor:
    ctx.video_processor.guide_ratio = current_guide_ratio

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    capture = st.button(
        "📸 촬영 및 분석 (균열만 보이게!)",
        use_container_width=True,
        type="primary",
        disabled=(ctx.video_processor is None)
    )

if capture and ctx.video_processor is not None:
    with ctx.video_processor.lock:
        if ctx.video_processor.latest_frame is not None:
            # 원본 프레임(오버레이 없는 상태)을 캡처
            frame_rgb = cv2.cvtColor(ctx.video_processor.latest_frame, cv2.COLOR_BGR2RGB)
            st.session_state.captured_img = frame_rgb
            st.session_state.captured_ratio = current_guide_ratio # 촬영 당시의 배율 저장
            st.session_state.analyze = True

# ════════════════════════════════════════════════════════════════
# 2. 분석 결과 (촬영 후 자동 실행)
# ════════════════════════════════════════════════════════════════
if not st.session_state.analyze or st.session_state.captured_img is None:
    st.stop()

img_np = st.session_state.captured_img
ratio_used = st.session_state.captured_ratio
H, W = img_np.shape[:2]

st.markdown("---")
st.markdown("### 🎯 분석 결과")

if st.button("🔁 다시 촬영"):
    st.session_state.captured_img = None
    st.session_state.analyze = False
    st.rerun()

# [핵심] mm/pixel 스케일 자동 계산 (가이드박스 비율 기준 역산)
box_w_px = W * ratio_used
scale = CARD_W_MM / box_w_px  # 1픽셀당 mm

col1, col2 = st.columns(2)
with col1:
    st.markdown("**📷 촬영된 원본 (카드 없음)**")
    st.image(img_np, use_container_width=True)
with col2:
    st.markdown("**📐 측정 기준 정보**")
    st.write(f"📏 설정된 촬영 거리: **{selected_distance} m**")
    st.write(f"🔬 1 픽셀 = **{scale:.4f} mm**")
    st.write(f"🖼️ 이미지 해상도: **{W} × {H} px**")

# 균열 검출
with st.spinner("🔍 균열 탐지 중..."):
    yolo = load_yolo()
    results = yolo.predict(img_np, conf=conf_thres, verbose=False)

if not results or results[0].masks is None:
    st.error("❌ 균열을 찾지 못했습니다. 사이드바에서 신뢰도를 낮춰보거나 조명을 확인하세요.")
    st.stop()

masks = results[0].masks.data.cpu().numpy()
full_mask = np.zeros((H, W), dtype=np.uint8)
for m in masks:
    mr = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
    full_mask = np.maximum(full_mask, (mr > 0.5).astype(np.uint8))

if full_mask.sum() == 0:
    st.warning("⚠️ 균열 마스크가 비어있습니다.")
    st.stop()

# 측정 계산 (스케일 기반)
pixel_cnt = int(full_mask.sum())
area_cm2 = (pixel_cnt * scale * scale) / 100.0
dt = cv2.distanceTransform(full_mask, cv2.DIST_L2, 5)
max_width_mm = 2 * float(dt.max()) * scale

# 지표 표시
c1, c2, c3 = st.columns(3)
c1.metric("📏 단위 픽셀 길이", f"{scale:.4f} mm")
c2.metric("📐 균열 총 면적", f"{area_cm2:.2f} cm²")
c3.metric("📏 최대 균열 폭", f"{max_width_mm:.2f} mm")

# 시각화 (빨간색으로 균열 오버레이)
overlay = img_np.copy()
overlay[full_mask > 0] = [255, 50, 50]
blended = cv2.addWeighted(img_np, 0.6, overlay, 0.4, 0)

st.image(blended, caption="🎯 검출 결과 (빨강: 균열)", use_container_width=True)

st.success(
    f"✅ 측정 완료!\n\n"
    f"지정 거리({selected_distance}m) 기반 계산 | 총 균열 픽셀: {pixel_cnt:,}개"
)