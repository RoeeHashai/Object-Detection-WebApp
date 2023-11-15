from ultralytics import YOLO
import time
import streamlit as st
import cv2
from pytube import YouTube
import settings


def load_model(model_path):
    """
    Loads a YOLO object detection model from the specified model_path.

    Parameters:
        model_path (str): The path to the YOLO model file.

    Returns:
        A YOLO object detection model.
    """
    model = YOLO(model_path)
    return model


def display_tracker_options():
    """
    Displays options for enabling object tracking in the Streamlit app.

    Returns:
        Tuple (bool, str): A tuple containing a boolean flag for displaying the tracker and the selected tracker type.
    """
    display_tracker = st.radio("Display Tracker", ("Yes", "No"))
    is_display_tracker = True if display_tracker == "Yes" else False
    if display_tracker:
        tracker_type = st.radio("Tracker", ("bytetrack.yaml", "botsort.yaml"))
        return is_display_tracker, tracker_type
    return is_display_tracker, None


def display_frames(
    model, acc, st_frame, image, is_display_tracker=None, tracker_type=None
):
    """
    Displays detectes objects from a video stream.

    Parameters:
        model (YOLO): A YOLO object detection model.
        acc (float): The model's confidence threshold.
        st_frame (streamlit.Streamlit): A Streamlit frame object.
        image (PIL.Image.Image): A frame from a video stream.
        is_display_tracker (bool): Whether or not to display a tracker.
        tracker_type (str): The type of tracker to display.

    Returns:
        None
    """

    image = cv2.resize(image, (720, int(720 * (9 / 16))))
    if is_display_tracker:
        res = model.track(image, conf=acc, persist=True, tracker=tracker_type)
    else:
        res = model.predict(image, conf=acc)

    res_plot = res[0].plot()
    st_frame.image(
        res_plot,
        caption="Detected Video",
        channels="BGR",
        use_column_width=True,
    )
    return res


def sum_detections(detected_objects_summary_list, model):
    """
    Summarizes detected objects from a list and displays the summary in a Streamlit success message.

    Parameters:
        detected_objects_summary_list (list): List of detected object indices.

    Returns:
        None
    """
    detected_objects_summary = set()
    for obj in detected_objects_summary_list:
        detected_objects_summary.add(model.names[int(obj)])
    name_summary = ", ".join(detected_objects_summary)
    st.success(f"Detected Objects: {name_summary}")
