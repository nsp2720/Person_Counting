
detect_web_cam.py   :   This script uses a YOLOv8 model to detect objects in a webcam feed and overlays a semi-transparent custom bounding box (box.png) on detected objects. The processed frames are optionally streamed to an RTMP server using FFmpeg.


detect_server_person_count_1.py  :  This script takes an RTMP stream (rtmp_input_url) as input, processes frames using a YOLOv8 model to detect humans, overlays a semi-transparent bounding box (box.png), and counts the detected persons. The output stream with overlays and person count is then re-streamed to another RTMP server (output_rtmp_url) using FFmpeg. If the input stream disconnects, the script retries automatically.


Person_count_script_api.py :  This script processes RTMP video streams to detect persons using a YOLOv8 model and uploads annotated frames to Azure Blob Storage when detections occur. It also sends detection metadata (camera ID, timestamp, image URL, and person count) to a POST API. It supports multiple streams using multithreading and retries if a stream disconnects.


Person_count_gstremmer.py  : This code processes video streams from multiple RTMP/RTSP URLs, detects people using a YOLO model, annotates the frames with bounding boxes, and streams out the annotated videos. If a person is detected, it uploads a frame to Azure Blob Storage and sends data to a POST API.  Inputs: RTMP/RTSP video stream URLs, Azure storage credentials, API endpoint, YOLO model. Outputs: Annotated video streams to specified RTMP output URLs, uploaded images to Azure Blob Storage, and data sent to a POST API


Azure_API_Only_Threads_2.py : This script processes multiple RTMP video streams to detect people using a YOLO model, drawing bounding boxes around them. If people are detected, and a sufficient time has passed since the last API call, it uploads a frame to Azure Blob Storage and sends data about the detection to a POST API. Inputs: RTMP video stream URLs, Azure Blob Storage credentials, POST API URL, YOLOv8 model weights.
Outputs: Annotated video frames (displayed, but not saved in this version), images uploaded to Azure Blob Storage when people are detected, and data sent to the analytics API.


KPT-demo_1.py : This code processes multiple RTSP video streams to detect people using YOLOv8, drawing bounding boxes with confidence scores. When a person is detected and at least 2 seconds have elapsed since the last detection, the frame is uploaded to Azure Blob Storage, and information is sent to an analytics API.
Inputs: RTSP video stream URLs, Azure Blob Storage credentials, Analytics API URL, YOLOv8 model, confidence threshold.
Outputs: Annotated video (displayed but not saved), images uploaded to Azure Blob Storage, data sent to the analytics API, controlled by a minimum time interval between uploads.



For Running all this files use cmd:

python3 filename.py

##To make environment use requirements.txt file.

python -m venv env_name

##to activate the environment

source env_name/bin/activate


by abobe cmd environment  will be activated then use cmd like:

python3 filename.py
