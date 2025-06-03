import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["YOLO_CONFIG_DIR"] = "/tmp"
import re
import cv2
import av
import easyocr
import spacy
import numpy as np
import streamlit as st
from ultralytics import YOLO
from collections import defaultdict
from supervision import Detections, ByteTrack
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase

# === CONFIGURATION ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best_new.pt')
NER_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'ner_plate_model4')

# === CLASS IDS ===
FACE_ID = 2
BADGE_ID = 1
LICENSE_ID = 0

# === Load models ===
try:
model = YOLO(MODEL_PATH)
except Exception as e:
    st.error(f"Failed to load YOLO model: {e}")
    st.stop()
ocr = easyocr.Reader(['en'], gpu=False)  # Set gpu=True if GPU is available
ner_model = spacy.load(NER_MODEL_PATH)

# === Tracking ===
tracker = ByteTrack()

# === License Plate Rules ===
plate_regex = r"[A-Z0-9]{3,}"

def is_valid_plate(text):
    text = text.strip().replace(" ", "")
    return bool(re.match(plate_regex, text)) or (re.search(r'[A-Za-z]', text) and re.search(r'\d', text))

def custom_ner_license_plate(text):
    doc = ner_model(text)
    for ent in doc.ents:
        if ent.label_.lower() in ["plate", "license_plate"]:
            return True
    return False

# === Blur helper ===
def blur_region(frame, xyxy):
    x1, y1, x2, y2 = map(int, xyxy)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return frame
    blur = cv2.GaussianBlur(crop, (25, 25), 30)
    frame[y1:y2, x1:x2] = blur
    return frame

# === Badge filter ===
def badge_is_valid(xyxy):
    x1, y1, x2, y2 = map(int, xyxy)
    area = (x2 - x1) * (y2 - y1)
    ratio = (x2 - x1) / (y2 - y1 + 1e-5)
    return 800 <= area <= 20000 and 0.4 <= ratio <= 1.5

# === Face filter ===
def face_is_valid(xyxy):
    x1, y1, x2, y2 = map(int, xyxy)
    ratio = (x2 - x1) / (y2 - y1 + 1e-5)
    area = (x2 - x1) * (y2 - y1)
    return 0.6 <= ratio <= 1.4 and area <= 100000

# === Streamlit WebRTC Video Processor ===
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.history = defaultdict(lambda: {"last_seen": 0, "box": None, "class_id": None})
        self.frame_count = 0
        self.MAX_MISSED_FRAMES = 10

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1

        results = model(img)[0]
        boxes = results.boxes

        detections = Detections(
            xyxy=boxes.xyxy.cpu().numpy(),
            confidence=boxes.conf.cpu().numpy(),
            class_id=boxes.cls.cpu().numpy().astype(int)
        )

        tracked = tracker.update_with_detections(detections)
        annotated = img.copy()

        for box, cls_id, track_id in zip(tracked.xyxy, tracked.class_id, tracked.tracker_id):
            track_id = int(track_id)

            self.history[track_id]["last_seen"] = self.frame_count
            self.history[track_id]["box"] = box
            self.history[track_id]["class_id"] = cls_id

            if cls_id == FACE_ID and face_is_valid(box):
                annotated = blur_region(annotated, box)

            elif cls_id == BADGE_ID and badge_is_valid(box):
                annotated = blur_region(annotated, box)

            elif cls_id == LICENSE_ID:
                x1, y1, x2, y2 = map(int, box)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                crop_up = cv2.resize(crop_rgb, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

                try:
                    ocr_result = ocr.readtext(crop_up)
                    plate_found = False
                    for result in ocr_result:
                        text, conf = result[1].strip(), result[2]
                        if conf > 0.3 and is_valid_plate(text) and custom_ner_license_plate(text):
                            plate_found = True
                            print(f"[✅] Detected plate: {text}")
                            break
                    if plate_found:
                        annotated = blur_region(annotated, box)
                except Exception as e:
                    print(f"[⚠️] OCR error: {e}")

        for tid, data in self.history.items():
            if self.frame_count - data["last_seen"] <= self.MAX_MISSED_FRAMES:
                box = data["box"]
                cls_id = data["class_id"]
                if cls_id == FACE_ID and face_is_valid(box):
                    annotated = blur_region(annotated, box)
                elif cls_id == BADGE_ID and badge_is_valid(box):
                    annotated = blur_region(annotated, box)
                elif cls_id == LICENSE_ID:
                    annotated = blur_region(annotated, box)

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

# === Streamlit App ===
st.title("Real-Time Object Detection and Blurring")
st.write("This application uses YOLOv8, ByteTrack, EasyOCR, and spaCy NER to detect and blur sensitive information in real-time.")

webrtc_streamer(
    key="example",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)
