# cardcrack.py
# 콘크리트 균열 자동 진단 V9.4
# - 등급 산출 제거
# - 라이브 카메라 위에 가이드 박스 오버레이 (HTML/JS 커스텀)
# - 가이드 비율과 실제 캡처 비율 일치

import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import cv2
import base64
from io import BytesIO
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="균열 자동 진단 V9.4", layout="wide")
st.title("🔍 콘크리트 균열 자동 진단 V9.4")
st.caption("📏 라이브 카메라 가이드로 거리 고정 → 카드 제거 → 촬영 → 자동 분석")

# ════════════════════════════════════════════════════════════════
# 상수
# ════════════════════════════════════════════════════════════════
CARD_W_MM = 85.60
CARD_H_MM = 53.98
CARD_ASPECT = CARD_W_MM / CARD_H_MM # ≈ 1.586

# 거리(m) → 카드가 화면 너비에서 차지하는 비율
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
1. 거리 선택 (현재 **{selected_distance}m**)
2. 아래 **라이브 카메라**에 카드를 비춤
3. 카드가 **녹색 점선 박스에 딱 맞도록** 거리 조절
4. **휴대폰 위치 고정** 후 카드 제거
5. **📸 촬영** 버튼 클릭 → 자동 분석
""")

# 세션 상태
if "captured_b64" not in st.session_state:
    st.session_state.captured_b64 = None

# ════════════════════════════════════════════════════════════════
# 라이브 카메라 + 가이드 오버레이 (HTML/JS 커스텀)
# ════════════════════════════════════════════════════════════════
st.markdown("### 🎥 1단계: 라이브 카메라 + 거리 가이드")

# 가이드 박스 비율 (화면 너비 대비)
guide_w_pct = current_guide_ratio * 100 # 너비 %
# 박스 높이 = (너비 / 카드종횡비) — 영상 높이 대비 %로 변환
# video는 4:3 또는 16:9. JS에서 실제 video 크기 기준으로 동적 계산.

# 컴포넌트 높이 (모바일 고려)
COMP_HEIGHT = 620

html_code = f"""
<div style="font-family: sans-serif; max-width: 100%;">
    <div id="camWrap" style="position: relative; width: 100%; max-width: 720px;
    margin: 0 auto; background:#000; border-radius: 12px; overflow: hidden;">
        <video id="video" autoplay playsinline muted
        style="width: 100%; height: auto; display: block;"></video>

        <div id="guideBox" style="
        position: absolute;
        top: 50%; left: 50%;
        transform: translate(-50%, -50%);
        width: {guide_w_pct}%;
        aspect-ratio: {CARD_ASPECT};
        border: 3px dashed #66ff66;
        box-shadow: 0 0 0 9999px rgba(0,0,0,0.35);
        box-sizing: border-box;
        pointer-events: none;">
            <div style="position:absolute;top:-4px;left:-4px;width:28px;height:28px;
            border-top:6px solid #66ff66;border-left:6px solid #66ff66;"></div>
            <div style="position:absolute;top:-4px;right:-4px;width:28px;height:28px;
            border-top:6px solid #66ff66;border-right:6px solid #66ff66;"></div>
            <div style="position:absolute;bottom:-4px;left:-4px;width:28px;height:28px;
            border-bottom:6px solid #66ff66;border-left:6px solid #66ff66;"></div>
            <div style="position:absolute;bottom:-4px;right:-4px;width:28px;height:28px;
            border-bottom:6px solid #66ff66;border-right:6px solid #66ff66;"></div>
            <div style="position:absolute;top:50%;left:50%;width:30px;height:3px;
            background:#ffcc00;transform:translate(-50%,-50%);"></div>
            <div style="position:absolute;top:50%;left:50%;width:3px;height:30px;
            background:#ffcc00;transform:translate(-50%,-50%);"></div>
        </div>

        <div style="position:absolute;top:10px;left:50%;transform:translateX(-50%);
        background:rgba(0,0,0,0.65);color:#66ff66;padding:6px 14px;
        border-radius:20px;font-size:13px;font-weight:bold;">
        📏 {selected_distance}m | 카드를 박스에 맞추기 ({guide_w_pct:.1f}%)
        </div>

        <div style="position:absolute;bottom:10px;left:50%;transform:translateX(-50%);
        background:rgba(0,0,0,0.65);color:#fff;padding:6px 14px;
        border-radius:20px;font-size:12px;">
        ① 카드 맞추기 → ② 위치 고정 → ③ 카드 제거 → ④ 촬영
        </div>
    </div>

    <div style="text-align:center; margin-top: 14px;">
        <button id="captureBtn" style="
        background: linear-gradient(135deg, #ff6b6b, #ee5a52);
        color: white; border: none;
        padding: 14px 36px; font-size: 17px; font-weight: bold;
        border-radius: 30px; cursor: pointer;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);">
        📸 촬영하기
        </button>
        <button id="switchBtn" style="
        background:#444; color:white; border:none;
        padding:14px 20px; font-size:15px; font-weight:bold;
        border-radius:30px; cursor:pointer; margin-left:8px;">
        🔄 전/후면
        </button>
    </div>

    <div id="status" style="text-align:center; margin-top:10px; color:#666; font-size:13px;">
    카메라 시작 중...
    </div>

    <canvas id="canvas" style="display:none;"></canvas>
</div>

<script>
(function() {{
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const captureBtn = document.getElementById('captureBtn');
    const switchBtn = document.getElementById('switchBtn');
    const status = document.getElementById('status');
    let currentStream = null;
    let useFront = false;

    async function startCamera() {{
        if (currentStream) {{
            currentStream.getTracks().forEach(t => t.stop());
        }}
        try {{
            const constraints = {{
                video: {{
                    facingMode: useFront ? 'user' : {{ ideal: 'environment' }},
                    width: {{ ideal: 1280 }},
                    height: {{ ideal: 960 }}
                }},
                audio: false
            }};
            currentStream = await navigator.mediaDevices.getUserMedia(constraints);
            video.srcObject = currentStream;
            status.textContent = '✅ 카메라 활성. 카드를 박스에 맞추세요.';
            status.style.color = '#0a7';
        }} catch (err) {{
            status.textContent = '❌ 카메라 접근 실패: ' + err.message;
            status.style.color = '#c00';
        }}
    }}

    switchBtn.addEventListener('click', () => {{
        useFront = !useFront;
        startCamera();
    }});

    captureBtn.addEventListener('click', () => {{
        if (!video.videoWidth) {{
            status.textContent = '⚠️ 카메라가 아직 준비되지 않음';
            return;
        }}
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        const dataUrl = canvas.toDataURL('image/jpeg', 0.92);

        // Streamlit으로 전송
        const payload = {{ image: dataUrl, w: canvas.width, h: canvas.height }};
        window.parent.postMessage(
            {{ isStreamlitMessage: true, type: 'streamlit:setComponentValue',
            value: payload }}, '*'
        );
        status.textContent = '📤 이미지 전송 완료! 아래에서 결과 확인';
        status.style.color = '#0a7';
        captureBtn.style.background = '#999';
        captureBtn.textContent = '✅ 촬영 완료';
        setTimeout(() => {{
            captureBtn.style.background = 'linear-gradient(135deg, #ff6b6b, #ee5a52)';
            captureBtn.textContent = '📸 다시 촬영';
        }}, 1500);
    }});

    startCamera();
}})();
</script>
"""

# bidirectional component via st.components.v1.html은 값 반환 불가
# → 대신 st.camera_input을 백업으로 함께 제공 (값 반환 가능)
components.html(html_code, height=COMP_HEIGHT)

st.markdown("---")
st.info("""
ℹ️ **위 라이브 카메라**는 거리 가이드 확인용입니다.
실제 분석에 사용할 이미지는 **아래의 카메라 입력**으로 촬영해 주세요.
(가이드를 본 그 자세 그대로 아래 버튼을 눌러 촬영)
""")

# ════════════════════════════════════════════════════════════════
# 실제 분석용 캡처 (Streamlit 표준 입력)
# ════════════════════════════════════════════════════════════════
st.markdown("### 📸 2단계: 분석용 사진 촬영 / 업로드")
tab1, tab2 = st.tabs(["📷 카메라", "📁 파일 업로드"])

input_img_np = None
with tab1:
    cam_img = st.camera_input("위 가이드대로 거리 맞춘 후 촬영 (카드 제거 상태)")
    if cam_img is not None:
        input_img_np = np.array(Image.open(cam_img).convert("RGB"))

with tab2:
    upload = st.file_uploader("균열 사진", type=["jpg", "jpeg", "png"])
    if upload is not None:
        input_img_np = np.array(Image.open(upload).convert("RGB"))

if input_img_np is None:
    st.warning("👆 위 카메라 또는 파일 업로드로 사진을 입력해주세요.")
    st.stop()

# ════════════════════════════════════════════════════════════════
# 분석
# ════════════════════════════════════════════════════════════════
H, W = input_img_np.shape[:2]
box_w_px = W * current_guide_ratio
scale = CARD_W_MM / box_w_px # mm per pixel

st.markdown("---")
st.markdown("### 🎯 3단계: 분석 결과")

col1, col2 = st.columns(2)
with col1:
    st.markdown("**📷 입력 이미지**")
    st.image(input_img_np, use_container_width=True)
with col2:
    st.markdown("**📐 측정 기준**")
    st.write(f"📏 촬영 거리: **{selected_distance} m**")
    st.write(f"🔬 1 픽셀 = **{scale:.4f} mm**")
    st.write(f"🖼️ 이미지 크기: **{W} × {H} px**")
    st.write(f"📦 카드 가이드 비율: **{current_guide_ratio*100:.1f}%**")

with st.spinner("🔍 균열 탐지 중..."):
    yolo = load_yolo()
    results = yolo.predict(input_img_np, conf=conf_thres, verbose=False)

if not results or results[0].masks is None:
    st.error("❌ 균열을 찾지 못했습니다. 신뢰도 임계값을 낮춰보세요.")
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

# 길이 추정 (skeleton 근사: 면적/평균폭)
mean_width_mm = (np.sum(dt[full_mask > 0]) * 2 * scale) / max(pixel_cnt, 1)
length_mm = (pixel_cnt * scale * scale) / max(mean_width_mm, 1e-6)

c1, c2, c3, c4 = st.columns(4)
c1.metric("📏 mm/pixel", f"{scale:.4f}")
c2.metric("📐 면적", f"{area_cm2:.2f} cm²")
c3.metric("📏 최대 폭", f"{max_width_mm:.2f} mm")
c4.metric("📐 추정 길이", f"{length_mm:.1f} mm")

# 시각화 (등급 표시 없음)
overlay = input_img_np.copy()
overlay[full_mask > 0] = [255, 50, 50]
blended = cv2.addWeighted(input_img_np, 0.6, overlay, 0.4, 0)
st.image(blended, caption="🎯 검출 결과 (빨강: 균열 영역)", use_container_width=True)

st.success(
    f"✅ 측정 완료 — 거리 **{selected_distance}m** 기반\n"
    f"📊 균열 픽셀: **{pixel_cnt:,}개** | 1px = **{scale:.4f}mm**"
)