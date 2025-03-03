# Person Detection and Counting with YOLOv8

This repository contains scripts for person detection and counting in real-time camera feeds using the YOLOv8 model.

## Setup

1.  **Create a virtual environment:**

    ```bash
    python -m venv env_name
    ```

2.  **Activate the virtual environment:**

    ```bash
    source env_name/bin/activate  # For Linux/macOS
    # OR
    # .\env_name\Scripts\activate   # For Windows
    ```

3.  **Install dependencies:**

    (It is assumed you have a `requirements.txt` file. If not, create one by running `pip freeze > requirements.txt` after installing the necessary packages.)

    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the scripts:**

    ```bash
    python3 filename.py
    ```

## Scripts

### `detect_web_cam.py`

Detects objects in a webcam feed using a YOLOv8 model and overlays a semi-transparent custom bounding box (`box.png`) on detected objects. Optionally streams the processed frames to an RTMP server using FFmpeg.

*   **Functionality:** Real-time object detection from webcam, custom bounding box overlay, optional RTMP streaming.
*   **Inputs:** Webcam feed, YOLOv8 model, `box.png` (optional), RTMP server URL (optional).
*   **Outputs:** Displayed processed webcam feed with detections, RTMP stream (optional).

### `detect_server_person_count_1.py`

Processes an RTMP stream, detects humans using a YOLOv8 model, overlays a semi-transparent bounding box (`box.png`), and counts the detected persons. The output stream is then re-streamed to another RTMP server using FFmpeg. Automatically retries if the input stream disconnects.

*   **Functionality:** Human detection and counting in an RTMP stream, bounding box overlay, RTMP re-streaming, automatic retry on disconnect.
*   **Inputs:** RTMP input stream URL (`rtmp_input_url`), YOLOv8 model, `box.png`, RTMP output stream URL (`output_rtmp_url`).
*   **Outputs:** RTMP stream re-streamed to the output URL with bounding boxes and person count.

### `Person_count_script_api.py`

Processes RTMP video streams to detect persons using a YOLOv8 model and uploads annotated frames to Azure Blob Storage when detections occur. Sends detection metadata (camera ID, timestamp, image URL, and person count) to a POST API. Supports multiple streams using multithreading and retries if a stream disconnects.

*   **Functionality:** Person detection in RTMP streams, Azure Blob Storage upload, API data posting, multithreading, stream retry.
*   **Inputs:** RTMP video stream URLs, Azure Blob Storage credentials, POST API URL, YOLOv8 model.
*   **Outputs:** Uploaded images to Azure Blob Storage, data sent to the POST API.

### `Person_count_gstremmer.py`

Processes video streams from multiple RTMP/RTSP URLs, detects people using a YOLO model, annotates the frames with bounding boxes, and streams out the annotated videos using GStreamer. If a person is detected, it uploads a frame to Azure Blob Storage and sends data to a POST API.

*   **Functionality:** Person detection in multiple RTMP/RTSP streams, bounding box annotation, GStreamer output streaming, Azure Blob Storage upload, API data posting.
*   **Inputs:** RTMP/RTSP video stream URLs, Azure storage credentials, API endpoint, YOLO model.
*   **Outputs:** Annotated video streams to specified RTMP output URLs, uploaded images to Azure Blob Storage, data sent to a POST API.

### `Azure_API_Only_Threads_2.py`

Processes multiple RTMP video streams to detect people using a YOLO model, drawing bounding boxes around them. If people are detected, and a sufficient time has passed since the last API call, it uploads a frame to Azure Blob Storage and sends data about the detection to a POST API.

*   **Functionality:** Person detection in multiple RTMP streams, rate-limited Azure Blob Storage upload, API data posting.
*   **Inputs:** RTMP video stream URLs, Azure Blob Storage credentials, POST API URL, YOLOv8 model weights.
*   **Outputs:** Annotated video frames (displayed, but not saved directly), images uploaded to Azure Blob Storage when people are detected, data sent to the analytics API.

### `KPT-demo_1.py`

Processes multiple RTSP video streams to detect people using YOLOv8, drawing bounding boxes with confidence scores. When a person is detected and at least 2 seconds have elapsed since the last detection, the frame is uploaded to Azure Blob Storage, and information is sent to an analytics API.

*   **Functionality:** Person detection in multiple RTSP streams, confidence score bounding boxes, rate-limited Azure Blob Storage upload, API data posting.
*   **Inputs:** RTSP video stream URLs, Azure Blob Storage credentials, Analytics API URL, YOLOv8 model, confidence threshold.
*   **Outputs:** Annotated video (displayed but not saved), images uploaded to Azure Blob Storage, data sent to the analytics API, controlled by a minimum time interval between uploads.
