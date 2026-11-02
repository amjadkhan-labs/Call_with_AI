import streamlit as st
import cv2
import face_recognition
import numpy as np
import os
import sqlite3
import pandas as pd
from datetime import datetime
from ultralytics import YOLO

# ---------------------------------------------------------
# 1. INITIALIZATION & CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(page_title="Smart Home Security", layout="wide")

DIRS = ["known_faces", "alerts", "logs"]
for d in DIRS:
    os.makedirs(d, exist_ok=True)

# Load YOLOv8 Nano for extremely fast human detection
@st.cache_resource
def load_yolo():
    return YOLO("yolov8n.pt")

yolo_model = load_yolo()

# Database Setup
def init_db():
    conn = sqlite3.connect("security_logs.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS events
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  name TEXT,
                  status TEXT,
                  timestamp DATETIME,
                  image_path TEXT)''')
    conn.commit()
    conn.close()

init_db()

# ---------------------------------------------------------
# 2. CORE AI FUNCTIONS
# ---------------------------------------------------------
@st.cache_data
def load_known_faces():
    """Generates 128-d embeddings for all images in the known_faces directory."""
    known_encodings = []
    known_names = []
    
    for filename in os.listdir("known_faces"):
        if filename.endswith((".jpg", ".png", ".jpeg")):
            image_path = os.path.join("known_faces", filename)
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            
            if len(encodings) > 0:
                known_encodings.append(encodings[0])
                known_names.append(os.path.splitext(filename)[0].title())
                
    return known_encodings, known_names

def log_event(name, status, frame):
    """Saves the image and logs the event to SQLite."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    safe_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    folder = "logs" if status == "Family" else "alerts"
    img_path = f"{folder}/{name}_{safe_time}.jpg"
    
    # Save image to disk
    cv2.imwrite(img_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    # Log to DB
    conn = sqlite3.connect("security_logs.db")
    c = conn.cursor()
    c.execute("INSERT INTO events (name, status, timestamp, image_path) VALUES (?, ?, ?, ?)",
              (name, status, timestamp, img_path))
    conn.commit()
    conn.close()

# ---------------------------------------------------------
# 3. STREAMLIT USER INTERFACE
# ---------------------------------------------------------
st.title("🛡️ AI Smart Home Security System")

menu = st.sidebar.selectbox("Navigation", ["Live Security Feed", "Activity Logs", "Manage Family Members"])

# Initialize Face Data
known_encodings, known_names = load_known_faces()

# ---- TAB 1: LIVE VIDEO PROCESSING ----
if menu == "Live Security Feed":
    st.header("Live Camera Feed")
    st.write("Processing video using YOLOv8 (Human Detection) + ResNet (Face Matching).")
    
    run_camera = st.checkbox("Start Camera")
    frame_window = st.image([])
    
    # Cooldown dictionary to prevent logging the same person 100 times a second
    if "cooldowns" not in st.session_state:
        st.session_state.cooldowns = {}
        
    COOLDOWN_SECONDS = 300 # 5 minutes before logging the same person again

    if run_camera:
        cap = cv2.VideoCapture(0) # Change to RTSP stream URL if using IP cameras
        
        while run_camera:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to access webcam.")
                break
                
            # Convert BGR to RGB for Streamlit and Face_Recognition
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Step 1: Detect Humans using YOLO to save compute
            results = yolo_model(frame, classes=[0], verbose=False) # class 0 is person
            
            human_detected = len(results[0].boxes) > 0
            
            if human_detected:
                # Step 2: Extract Face and Match
                face_locations = face_recognition.face_locations(rgb_frame)
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                
                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
                    name = "Unknown"
                    status = "Alert"
                    color = (255, 0, 0) # Red for unknown
                    
                    if True in matches:
                        first_match_index = matches.index(True)
                        name = known_names[first_match_index]
                        status = "Family"
                        color = (0, 255, 0) # Green for family
                        
                    # Draw Bounding Box
                    cv2.rectangle(rgb_frame, (left, top), (right, bottom), color, 2)
                    cv2.putText(rgb_frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
                    
                    # Logic for Logging and Alerts
                    current_time = datetime.now()
                    last_seen = st.session_state.cooldowns.get(name)
                    
                    if last_seen is None or (current_time - last_seen).total_seconds() > COOLDOWN_SECONDS:
                        log_event(name, status, rgb_frame)
                        st.session_state.cooldowns[name] = current_time
                        
                        if status == "Alert":
                            st.toast(f"🚨 UNKNOWN PERSON DETECTED!", icon="🚨")
                        else:
                            st.toast(f"👋 Welcome {name} logged.", icon="✅")

            frame_window.image(rgb_frame)
            
        cap.release()

# ---- TAB 2: ACTIVITY LOGS ----
elif menu == "Activity Logs":
    st.header("Security Event Logs")
    
    conn = sqlite3.connect("security_logs.db")
    df = pd.read_sql_query("SELECT * FROM events ORDER BY timestamp DESC", conn)
    conn.close()
    
    if df.empty:
        st.info("No events logged yet.")
    else:
        # Display as a clean dataframe
        st.dataframe(df[['timestamp', 'name', 'status']], use_container_width=True)
        
        st.subheader("Recent Photographic Evidence")
        cols = st.columns(4)
        for idx, row in df.head(8).iterrows():
            with cols[idx % 4]:
                if os.path.exists(row['image_path']):
                    st.image(row['image_path'], caption=f"{row['name']} ({row['status']}) - {row['timestamp'][11:]}")
                else:
                    st.write("Image missing.")

# ---- TAB 3: MANAGE FAMILY ----
elif menu == "Manage Family Members":
    st.header("Customize Authorized Faces")
    st.write("Upload a clear picture of a family member. The filename will be used as their name.")
    
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        file_path = os.path.join("known_faces", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Successfully added {uploaded_file.name} to authorized family members!")
        st.cache_data.clear() # Clear cache so embeddings recalculate on next run
        
    st.subheader("Currently Authorized")
    if known_names:
        cols = st.columns(5)
        for idx, name in enumerate(known_names):
            with cols[idx % 5]:
                st.write(f"✅ {name}")
    else:
        st.warning("No family members added yet.")
