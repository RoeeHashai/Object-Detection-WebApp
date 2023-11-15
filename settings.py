from pathlib import Path
import sys

# Get the absolute path of the current script file
file_path = Path(__file__).resolve()

# Get the parent directory of the script file
root_path = file_path.parent

# Add the parent directory to the system path if not already present
if root_path not in sys.path:
    sys.path.append(str(root_path))

# Create a relative path from the current working directory to the parent directory
ROOT = root_path.relative_to(Path.cwd())

# Source
IMAGE = 'Image'
VIDEO = 'Video'
YOUTUBE = 'Youtube'
WEBCAM  = 'Webcam'
SOURCE_LIST = [IMAGE, VIDEO, YOUTUBE, WEBCAM]

# Images configuation
IMAGES_DIR = ROOT / 'images'
DEFAULT_IMAGE = IMAGES_DIR / 'default.jpg'

# Videos configuation
VIDEO_DIR = ROOT / 'videos'
DEFAULT_VIDEO = VIDEO_DIR / 'default.mp4'

# Youtube configuation
DEFAULT_URL = "https://www.youtube.com/watch?v=P0wNIsAjht8"


# ML Model configuation
MODEL_DIR = ROOT / 'weights'
DETECTION_MODEL = MODEL_DIR / 'yolov8n.pt'
SEGMENTATION_MODEL = MODEL_DIR / 'yolov8n-seg.pt'