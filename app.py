import PIL
import streamlit as st
from pathlib import Path
from ImageDetector import ImageDetector
from VideoDetector import VideoDetector
from YouTubeDetector import YouTubeDetector
from WebcamDetector import WebcamDetector
import settings
import helper

# Setting page layout
st.set_page_config(
    page_title="Object Detection And Tracking using YOLOv8",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Main page heading
st.title("Object Detection And Tracking using YOLOv8")

# Sidebar
st.sidebar.header("ML Model Config")
model_type = st.sidebar.radio("Select Task", ['Detection', 'Segmentation'])
confidence = float(st.sidebar.slider("Select Model Confidence", 25, 100, 40)) / 100

# Selecting Detection Or Segmentation
if model_type == "Detection":
    model_path = Path(settings.DETECTION_MODEL)
elif model_type == "Segmentation":
    model_path = Path(settings.SEGMENTATION_MODEL)

# Load Pre-trained ML Model
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Error loading model. Check the specified path: {model_path}")
    st.error(ex)

# Sidebar
st.sidebar.header("Data Config")
source_radio = st.sidebar.radio("Select Source", settings.SOURCE_LIST)

if source_radio == settings.IMAGE:
    image_detector = ImageDetector(model, confidence)
    image_detector.detect()
elif source_radio == settings.VIDEO:
    video_detector = VideoDetector(model, confidence)
    video_detector.detect()
elif source_radio == settings.YOUTUBE:
    youtube_detector = YouTubeDetector(model, confidence)
    youtube_detector.detect()
elif source_radio == settings.WEBCAM:
    webcam_detector = WebcamDetector(model, confidence)
    webcam_detector.detect()
