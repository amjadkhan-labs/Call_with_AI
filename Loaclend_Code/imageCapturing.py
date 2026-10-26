# import cv2
# import time
# import numpy as np


# def preprocessing_for_gatekeeper(frame,size=(224, 224)):
    
#     """
#     Lightweight preprocessing ONLY for fast scene-change & motion math.
#     Runs in < 0.5 milliseconds on CPU.
#     """
    
#     # 1. Convert BGR to Grayscale (reduces 3 color channels to 1 intensity channel)
#     gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
#     # 2. Downscale to 64x64 (4,096 pixels instead of 2,073,600 pixels in 1080p)
#     gray_frame_lowpixel = cv2.resize(gray_frame,size,interpolation=cv2.INTER_AREA)
    
#     # 3. Apply Gaussian Blur to smooth out high-frequency mobile sensor noise/flicker
#     blurred_frame = cv2.GaussianBlur(gray_frame_lowpixel,(5,5),0)
    
#     return blurred_frame


# def preprocessing_for_model(image,size=(224,224),jpeg_quality=85):
#     """
#     Heavy preprocessing executed ONLY when a new scene is locked in.
#     Prepares the single frame for our Deep Learning model / Cloud API.
#     """
    
#     # 1. Resize to standard Vision backbone input size (224x224 for ResNet / ViT)
#     image_resize = cv2.resize(image,size,interpolation=cv2.INTER_AREA)
    
#     # 2. Convert OpenCV's default BGR to RGB (required by Tensorflow & Hugging Face models)
#     rgb_image = cv2.cvtColor(image_resize,cv2.COLOR_BGR2RGB)
    
#     # 3. Optional Cloud Optimization: Compress to in-memory JPEG bytes (~12 KB)
#     # This saves 95% network bandwidth when sending the 1 frame to a cloud GPU server
#     encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
#     success, encoded_jpeg = cv2.imencode('.jpg', image_resize, encode_param)
#     jpeg_bytes = encoded_jpeg.tobytes() if success else None
    
#     return rgb_image, jpeg_bytes


    
#     # How different the new view must be from the last sent photo
#     # How still the hand must be (lower = stricter stillness)
#     # Must hold still for ~8 frames (~0.25 sec) before capturing
# def run_edge_gatekeeper(source=0,scene_threshold=15.0,motion_threshold=2.5,steady_frames_required=8):
#     """
#     Captures live video, detects scene changes + camera stillness,
#     and triggers preprocessing for ONLY 1 image per new scene.
#     """
#     cap = cv2.VideoCapture(source)
    
#     if not cap.isOpened():
#         print(f"[ERROR] Could not open camera source: {source}")
#         return

#     prev_gray = None
#     last_sent_gray = None
#     steady_counter = 0
#     scenes_sent_count = 0

#     print("[INFO] Edge Gatekeeper started. Press 'q' to quit.")
    
#     while True:
#         ret, frame = cap.read()
        
#         # Handle dropped mobile Wi-Fi frames gracefully without crashing
#         if not ret or frame is None:
#             print("[WARNING] Dropped frame from stream, retrying...")
#             time.sleep(0.05)
#             continue

#         # Step 1: Fast grayscale thumbnail for gatekeeper math
#         current_gray = preprocessing_for_gatekeeper(frame)

#         # Initialize on the very first frame
#         if prev_gray is None:
#             prev_gray = current_gray
#             continue

#         # Step 2: Calculate Motion Difference (Current frame vs Frame 1/30s ago)
#         motion_diff = np.mean(cv2.absdiff(current_gray, prev_gray))
#         prev_gray = current_gray

#         # Check if the user's hand is holding the camera steady
#         if motion_diff < motion_threshold:
#             steady_counter += 1
#         else:
#             steady_counter = 0  # Reset counter if hand is still moving/panning

#         # Step 3: Calculate Scene Difference (Current frame vs Last Sent Scene)
#         if last_sent_gray is None:
#             # First scene ever: force scene_diff above threshold so it captures once steady
#             scene_diff = 999.0
#         else:
#             scene_diff = np.mean(cv2.absdiff(current_gray, last_sent_gray))

#         # Step 4: THE GATEKEEPER DECISION
#         # Condition A: We are pointing at a new scene (scene_diff > scene_threshold)
#         # Condition B: The camera has stopped moving (steady_counter >= steady_frames_required)
#         is_new_scene = scene_diff > scene_threshold
#         is_camera_steady = steady_counter >= steady_frames_required

#         status_text = "Monitoring (Same Scene)"
#         status_color = (0, 255, 0)  # Green

#         if is_new_scene and not is_camera_steady:
#             status_text = "New Scene Detected -> Waiting for camera to stabilize..."
#             status_color = (0, 165, 255)  # Orange

#         elif is_new_scene and is_camera_steady:
#             # TRIGGER! Preprocess and send ONLY this 1 sharp frame
#             rgb_img, jpeg_bytes = preprocessing_for_model(frame)
            
#             # Lock this scene so we don't send duplicate photos
#             last_sent_gray = current_gray
#             steady_counter = 0
#             scenes_sent_count += 1
            
#             payload_kb = len(jpeg_bytes) / 1024 if jpeg_bytes else 0
#             print(
#                 f"[SCENE #{scenes_sent_count} CAPTURED] "
#                 f"Scene Diff: {scene_diff:.1f} | Motion Diff: {motion_diff:.2f} | "
#                 f"RGB Shape: {rgb_img.shape} | Cloud Payload: {payload_kb:.1f} KB"
#             )
#             status_text = f"SENT SCENE #{scenes_sent_count} TO MODEL!"
#             status_color = (0, 0, 255)  # Red flash on capture

#         # Step 5: Draw live telemetry HUD on display frame (helpful for debugging & demo)
#         display_frame = frame.copy()
#         cv2.putText(display_frame, f"Status: {status_text}", (15, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)
#         cv2.putText(display_frame, f"Scene Diff: {scene_diff:.1f} (Thresh: {scene_threshold})", (15, 60),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1)
#         cv2.putText(display_frame, f"Motion Diff: {motion_diff:.2f} | Steady Frames: {steady_counter}/{steady_frames_required}", (15, 85),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1)
#         cv2.putText(display_frame, f"Total Images Sent to Cloud: {scenes_sent_count}", (15, 115),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

#         cv2.imshow("LiveSight - Edge Gatekeeper", display_frame)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()


# if __name__ == "__main__":
#     # Replace 0 with your mobile IP Webcam URL string e.g. "http://192.168.1.5:8080/video"
#     run_edge_gatekeeper(source=0)


import cv2
import time
import numpy as np


def preprocess_for_gatekeeper(frame, size=(64, 64)):
    """
    Lightweight 64x64 grayscale thumbnail used for:
    1. Fast motion check (prev_gray)
    2. Slow camera drift check (anchor_gray)
    3. New scene check vs the 1 last sent frame (last_sent_gray)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small_gray = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    blurred_gray = cv2.GaussianBlur(small_gray, (5, 5), 0)
    return blurred_gray


def preprocess_for_model(frame, target_size=(224, 224), jpeg_quality=85):
    """
    Executed ONLY when a new scene is focused for >= 1.0 second.
    Prepares the single image for our Deep Learning model.
    """
    resized_bgr = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
    rgb_image = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    success, encoded_jpeg = cv2.imencode('.jpg', resized_bgr, encode_param)
    jpeg_bytes = encoded_jpeg.tobytes() if success else None

    return rgb_image, jpeg_bytes


def run_edge_gatekeeper(
    source=0,
    scene_threshold=12.0,       # How different current view must be from the 1 last sent photo
    motion_threshold=2.5,       # Catches fast hand movement between adjacent frames
    anchor_threshold=4.5,       # Catches slow camera panning away from focus start point
    focus_time_required=1.0     # Must hold steady on scene for 1.0 full second
):
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[ERROR] Could not open camera source: {source}")
        return

    prev_gray = None           # last fram
    anchor_gray = None         # 
    last_sent_gray = None      # last sent gray image
    focus_start_time = None    # focus time
    scenes_sent_count = 0      # how many image sent yet

    print("[INFO] Clean Edge Gatekeeper started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("[WARNING] Dropped frame from stream, retrying...")
            time.sleep(0.05)
            continue

        now = time.monotonic()
        current_gray = preprocess_for_gatekeeper(frame)

        # Initialize on the very first frame
        if prev_gray is None or anchor_gray is None:
            prev_gray = current_gray
            anchor_gray = current_gray
            focus_start_time = now
            continue

        # Step 1: Detect Fast Motion (vs prev_gray) AND Slow Movement (vs anchor_gray)
        instant_diff = np.mean(cv2.absdiff(current_gray, prev_gray))
        anchor_drift = np.mean(cv2.absdiff(current_gray, anchor_gray))
        prev_gray = current_gray

        if instant_diff > motion_threshold or anchor_drift > anchor_threshold:
            # Camera is moving (fast or slow) -> Reset anchor and restart 1.0s timer
            anchor_gray = current_gray
            focus_start_time = now
            focus_duration = 0.0
        else:
            # Camera is holding still -> count up time in seconds
            focus_duration = now - focus_start_time

        is_camera_steady = focus_duration >= focus_time_required

        # Step 2: Simple Scene Check (Compare ONLY against the 1 last sent frame)
        if last_sent_gray is None:
            scene_diff = 999.0  # First capture ever
        else:
            scene_diff = np.mean(cv2.absdiff(current_gray, last_sent_gray))

        is_new_scene = scene_diff > scene_threshold

        # Step 3: Gatekeeper Decision
        status_text = "Holding on Current Scene (Already Captured)"
        status_color = (0, 255, 0)  # Green

        if is_new_scene and not is_camera_steady:
            status_text = f"Focusing on New Scene... {focus_duration:.1f}s / {focus_time_required:.1f}s"
            status_color = (0, 165, 255)  # Orange

        elif is_new_scene and is_camera_steady:
            # TRIGGER! Focused >= 1.0s on a new scene -> Send 1 image
            rgb_img, jpeg_bytes = preprocess_for_model(frame)

            last_sent_gray = current_gray
            anchor_gray = current_gray
            focus_start_time = now
            scenes_sent_count += 1

            payload_kb = len(jpeg_bytes) / 1024 if jpeg_bytes else 0
            print(
                f"[SCENE #{scenes_sent_count} SENT] "
                f"Held: {focus_duration:.2f}s | Scene Diff: {scene_diff:.1f} | "
                f"Drift: {anchor_drift:.2f} | Payload: {payload_kb:.1f} KB"
            )
            status_text = f"CAPTURED SCENE #{scenes_sent_count}!"
            status_color = (0, 0, 255)  # Red

        # Step 4: Live HUD
        display_frame = frame.copy()
        cv2.putText(display_frame, f"Status: {status_text}", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        cv2.putText(display_frame, f"Focus Timer: {min(focus_duration, focus_time_required):.2f}s / {focus_time_required:.1f}s", (15, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(display_frame, f"Scene Diff: {scene_diff:.1f} (Need > {scene_threshold})", (15, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(display_frame, f"Instant Diff: {instant_diff:.2f} | Slow Drift: {anchor_drift:.2f} (Max {anchor_threshold})", (15, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(display_frame, f"Total Images Sent: {scenes_sent_count}", (15, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow("LiveSight - Clean Edge Gatekeeper", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_edge_gatekeeper(source=0)