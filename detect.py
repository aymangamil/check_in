import streamlit as st
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
import numpy as np

# ---------------------------
# إعداد واجهة Streamlit
# ---------------------------
st.set_page_config(page_title="👥 People Counter", layout="wide")
st.title("👥 People Counter — Live Stream with Unique Tracking")

# زر لإعادة تعيين العداد
if "unique_ids" not in st.session_state:
    st.session_state["unique_ids"] = set()

if st.button("🔄 Reset Counter"):
    st.session_state["unique_ids"].clear()

# ---------------------------
# تحميل موديل YOLOv8 Nano
# ---------------------------
model = YOLO("yolov8n.pt")  # سريع وخفيف

# ---------------------------
# Video Processor Class
# ---------------------------
class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # كشف وتتبع الأشخاص فقط (class 0 = person)
        results = model.track(img, persist=True, classes=[0])

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
            ids = results[0].boxes.id.cpu().numpy().astype(int)

            for i, person_id in enumerate(ids):
                x1, y1, x2, y2 = boxes[i].astype(int)

                # إذا الشخص جديد، نضيفه للـ set
                if person_id not in st.session_state["unique_ids"]:
                    st.session_state["unique_ids"].add(person_id)

                # رسم bounding box أخضر مع رقم الشخص
                cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(img, f"ID:{person_id}", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        # عداد الأشخاص الفريدين في الأعلى
        cv2.putText(img, f"Unique Count: {len(st.session_state['unique_ids'])}", (20,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ---------------------------
# تشغيل WebRTC Stream
# ---------------------------
webrtc_streamer(
    key="people_counter",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False}
)

# ---------------------------
# عرض عدد الأشخاص في Streamlit sidebar
# ---------------------------
st.sidebar.header("Statistics")
st.sidebar.write("👥 Unique People Count:", len(st.session_state["unique_ids"]))

