import cv2
import time
import threading
import queue
import requests
import streamlit as st
from imageCapturing import EdgeGatekeeper

st.set_page_config(page_title="LiveSight - Image Captioning", layout="wide")
st.title("LiveSight - Live Camera & Transformer Captioning")

# Initialize caption history in session state
if "caption_history" not in st.session_state:
    st.session_state.caption_history = []

# Sidebar: Backend URL, Health Check, and Gatekeeper Thresholds
with st.sidebar:
    st.header("FastAPI Backend")
    api_base_url = st.text_input("Backend Base URL", value="https://call-with-ai-backend.onrender.com")
    predict_url = f"{api_base_url}/predict"
    health_url = f"{api_base_url}/health"

    # Check FastAPI /health endpoint
    if st.button("Check Backend Status"):
        try:
            resp = requests.get(health_url, timeout=3)
            data = resp.json()
            if data.get("status") == "online" and data.get("model_loaded"):
                st.success("Backend Online | Transformer Model Loaded")
            else:
                st.warning(f"Backend reachable, but model_loaded={data.get('model_loaded')}")
        except Exception as e:
            st.error(f"Cannot reach backend: {e}")

    st.subheader("Gatekeeper Settings")
    scene_threshold = st.slider("Scene Diff Threshold", 5.0, 30.0, 12.0, 0.5)
    motion_threshold = st.slider("Motion Threshold", 0.5, 10.0, 2.5, 0.5)
    anchor_threshold = st.slider("Anchor Drift Threshold", 1.0, 15.0, 4.5, 0.5)
    focus_time_required = st.slider("Focus Time Required (s)", 0.5, 3.0, 1.0, 0.1)

# Main UI Layout: Left for Live Camera, Right for Captions
col_cam, col_result = st.columns([3, 2])

with col_cam:
    run_camera = st.toggle("Turn On Live Camera", value=False)
    video_placeholder = st.empty()

with col_result:
    st.subheader("Generated Caption")
    status_placeholder = st.empty()
    caption_placeholder = st.empty()
    metrics_placeholder = st.empty()
    captured_img_placeholder = st.empty()
    st.subheader("Recent Captions")
    history_placeholder = st.empty()


def send_frame_to_fastapi(jpeg_bytes, scene_id, url, result_queue):
    """Posts jpeg_bytes to FastAPI /predict in a background thread."""
    try:
        files = {"file": (f"scene_{scene_id}.jpg", jpeg_bytes, "image/jpeg")}
        response = requests.post(url, files=files, timeout=60)

        if response.status_code == 200:
            data = response.json()
            result_queue.put({
                "scene_id": scene_id,
                "status": "success",
                "caption": data.get("caption", ""),
                "inference_time_ms": data.get("inference_time_ms", 0.0)
            })
        else:
            # Extracts FastAPI HTTPException detail message
            err_detail = response.json().get("detail", response.text)
            result_queue.put({
                "scene_id": scene_id,
                "status": "error",
                "error": f"HTTP {response.status_code}: {err_detail}"
            })
    except Exception as e:
        result_queue.put({
            "scene_id": scene_id,
            "status": "error",
            "error": str(e)
        })


if run_camera:
    cap = cv2.VideoCapture(0)
    gatekeeper = EdgeGatekeeper(
        scene_threshold=scene_threshold,
        motion_threshold=motion_threshold,
        anchor_threshold=anchor_threshold,
        focus_time_required=focus_time_required
    )
    result_queue = queue.Queue()

    if not cap.isOpened():
        st.error("Could not open webcam (source 0). Make sure no other app is using it.")
    else:
        status_placeholder.info("Camera active. Hold steady on a new scene for 1.0s to generate a caption...")

        while run_camera:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.05)
                continue

            # Run gatekeeper on current frame
            display_bgr, triggered, payload = gatekeeper.process_frame(frame)

            # Convert BGR to RGB for Streamlit live display
            display_rgb = cv2.cvtColor(display_bgr, cv2.COLOR_BGR2RGB)
            video_placeholder.image(display_rgb, channels="RGB", use_container_width=True)

            # If gatekeeper triggers, send jpeg_bytes to FastAPI asynchronously
            if triggered and payload is not None:
                scene_id = payload["scene_id"]
                status_placeholder.warning(
                    f"Scene #{scene_id} captured ({payload['payload_kb']:.1f} KB). "
                    f"Sending to Transformer model..."
                )
                captured_img_placeholder.image(
                    payload["rgb_image"],
                    caption=f"Scene #{scene_id} sent to /predict (224x224)",
                    width=224
                )

                worker = threading.Thread(
                    target=send_frame_to_fastapi,
                    args=(payload["jpeg_bytes"], scene_id, predict_url, result_queue),
                    daemon=True
                )
                worker.start()

            # Check if FastAPI returned a caption
            while not result_queue.empty():
                res = result_queue.get_nowait()
                if res["status"] == "success":
                    status_placeholder.success(f"Scene #{res['scene_id']} captioned successfully!")
                    caption_placeholder.markdown(f"### \"{res['caption']}\"")
                    metrics_placeholder.caption(f"Inference Time: **{res['inference_time_ms']} ms**")

                    # Prepend to history list
                    st.session_state.caption_history.insert(
                        0,
                        f"**Scene #{res['scene_id']}** ({res['inference_time_ms']} ms): {res['caption']}"
                    )
                    history_placeholder.markdown("\n\n".join(st.session_state.caption_history[:5]))
                else:
                    status_placeholder.error(f"Backend Error (Scene #{res['scene_id']}): {res['error']}")

            time.sleep(0.01)

        cap.release()
else:
    video_placeholder.info("Toggle 'Turn On Live Camera' above to start the live stream.")