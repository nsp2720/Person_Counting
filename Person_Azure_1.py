import cv2
import subprocess
import time
import threading
import requests
from ultralytics import YOLO
import numpy as np
from azure.storage.blob import BlobServiceClient
from datetime import datetime

# Function to start FFmpeg process
def start_ffmpeg_process(output_rtmp_url, width, height, fps):
    try:
        ffmpeg_process = subprocess.Popen(
            [
                'ffmpeg', '-re', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
                '-s', f'{width}x{height}', '-r', str(int(fps)),
                '-i', '-', '-c:v', 'libx264', '-preset', 'ultrafast',
                '-maxrate', '3000k', '-bufsize', '7000k', '-f', 'flv', output_rtmp_url
            ],
            stdin=subprocess.PIPE
        )
        return ffmpeg_process
    except Exception as e:
        print(f"Error starting FFmpeg process: {e}")
        return None

# Function to open RTMP stream
def open_rtmp_stream(rtmp_url):
    cap = cv2.VideoCapture(rtmp_url)
    if not cap.isOpened():
        print("Error: Unable to open RTMP stream.")
        return None
    return cap

# Function to upload image to Azure Blob Storage and send data to POST API
def upload_to_azure_and_post(image, azure_connection_string, container_name, cameradid, post_api_url, an_id, img_count):
    try:
        blob_service_client = BlobServiceClient.from_connection_string(azure_connection_string)
        container_client = blob_service_client.get_container_client(container_name)
        
        # Create a dynamic filename with the current date and time
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        blob_name = f"live-record/arista8/persons_{timestamp}.jpg"
        
        # Convert the image to JPEG format
        _, img_encoded = cv2.imencode('.jpg', image)
        img_bytes = img_encoded.tobytes()
        
        # Upload the image to Azure Blob Storage
        blob_client = container_client.get_blob_client(blob_name)
        blob_client.upload_blob(img_bytes, overwrite=True)
        
        print(f"Image uploaded successfully: {blob_name}")
        
        # Construct the image URL
        image_url = f"https://{blob_service_client.account_name}.blob.core.windows.net/{container_name}/{blob_name}"
        send_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.000")
        
        # Send data to POST API
        data = {
            "cameradid": cameradid,
            "sendtime": send_time,
            "imgurl": image_url,
            "an_id": an_id,
            "ImgCount": img_count
        }
        headers = {'Content-Type': 'application/json'}
        response = requests.post(post_api_url, json=data, headers=headers)
        if response.status_code == 200:
            print(f"Data sent to POST API successfully: {response.json()}")
        else:
            print(f"POST API error: {response.status_code}, {response.text}")
    except Exception as e:
        print(f"Error uploading image or sending data to POST API: {e}")

# Set parameters
rtmp_input_url = 'rtmp://ptz.vmukti.com:80/live-record/KKKK-820603-VVVVV'
output_rtmp_url = 'rtmp://ptz.vmukti.com:80/live-record/NILE-103247-DILIP'
azure_connection_string = "BlobEndpoint=https://nvrdatashinobi.blob.core.windows.net/;QueueEndpoint=https://nvrdatashinobi.queue.core.windows.net/;FileEndpoint=https://nvrdatashinobi.file.core.windows.net/;TableEndpoint=https://nvrdatashinobi.table.core.windows.net/;SharedAccessSignature=sv=2022-11-02&ss=bfqt&srt=sco&sp=rwdlacupiytfx&se=2025-01-10T14:44:50Z&st=2025-01-08T06:44:50Z&spr=https,http&sig=7di0L6jZatcA5agFdF8Dm3ozpNKCSRYygmd8JQezKn0%3D"
container_name = "nvrdatashinobi"
post_api_url = "https://tn2023demo.vmukti.com/api/analytics/"
an_id = 2
img_count = 0

# Extract camera DID from RTMP input URL
cameradid = rtmp_input_url.split('/')[-1]

# Load the YOLOv8 model
model = YOLO('best.pt')  # Path to your custom trained model (best.pt)

# Start RTMP stream processing
cap = open_rtmp_stream(rtmp_input_url)
if cap is None:
    print("RTMP feed not available.")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
fps = cap.get(cv2.CAP_PROP_FPS) or 25

ffmpeg_process = start_ffmpeg_process(output_rtmp_url, width, height, fps)
if ffmpeg_process is None:
    print("Error starting FFmpeg process.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Stream disconnected. Retrying in 5 seconds...")
        cap.release()
        time.sleep(5)
        cap = open_rtmp_stream(rtmp_input_url)
        if cap is None:
            continue
        ffmpeg_process = start_ffmpeg_process(output_rtmp_url, width, height, fps)
        if ffmpeg_process is None:
            print("Error starting FFmpeg process.")
            break

    # Run YOLOv8 model for human detection
    results = model(frame)  # This returns a list of results
    
    # Initialize person count
    person_count = 0
    
    # Extract bounding boxes from the results
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get coordinates of the bounding box [xmin, ymin, xmax, ymax]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Increment person count
            person_count += 1
            
            # Draw the bounding box around the detected person
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw green rectangle
            
            # Capture image of the detected person
            person_image = frame[y1:y2, x1:x2]
            
            # Upload image to Azure and send to POST API in a separate thread
            img_count += 1
            threading.Thread(
                target=upload_to_azure_and_post,
                args=(person_image, azure_connection_string, container_name, cameradid, post_api_url, an_id, img_count),
                daemon=True
            ).start()
    
    # Write the processed frame to the output RTMP stream
    ffmpeg_process.stdin.write(frame.tobytes())

# Release resources and cleanup
cap.release()
ffmpeg_process.stdin.close()
ffmpeg_process.wait()
cv2.destroyAllWindows()
print("Video processing complete.")

