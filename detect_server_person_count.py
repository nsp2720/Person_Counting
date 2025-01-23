import cv2
import subprocess
import time
from ultralytics import YOLO
import numpy as np

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

# Set parameters
rtmp_input_url = 'rtmp://ptz.vmukti.com:80/live-record/VSPL-103247-HYQPO'
output_rtmp_url = 'rtmp://ptz.vmukti.com:80/live-record/NILE-103247-DILIP'

# Load the YOLOv8 model
model = YOLO('best.pt')  # Path to your custom trained model (best.pt)

# Set transparency factor (between 0 and 1)
transparency_factor = 0.4  # Adjust this value for desired transparency

# Load the custom bounding box overlay image (box.png) with an alpha channel
overlay_img = cv2.imread('box.png', cv2.IMREAD_UNCHANGED)

# Ensure overlay has an alpha channel; add one if necessary
if overlay_img.shape[2] == 3:
    alpha_channel = np.ones(overlay_img.shape[:2], dtype=overlay_img.dtype) * 255
    overlay_img = np.dstack([overlay_img, alpha_channel])

# Scale the alpha channel by the transparency factor to reduce transparency
overlay_img[:, :, 3] = (overlay_img[:, :, 3] * transparency_factor).astype(overlay_img.dtype)

# Start FFmpeg process
cap = open_rtmp_stream(rtmp_input_url)
if cap is None:
    print("RTMP feed not available.")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
fps = cap.get(cv2.CAP_PROP_FPS) or 25

# Start the FFmpeg process to push the video
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

            # Increment person count for each detected person
            person_count += 1

            # Resize the overlay image to fit the bounding box size
            box_width = x2 - x1
            box_height = y2 - y1
            resized_overlay = cv2.resize(overlay_img, (box_width, box_height))

            # Get the alpha mask for the overlay image (transparency)
            alpha_mask = resized_overlay[:, :, 3] / 255.0

            # Overlay the image with transparency applied
            for c in range(3):  # Loop through each color channel
                frame[y1:y2, x1:x2, c] = (1 - alpha_mask) * frame[y1:y2, x1:x2, c] + alpha_mask * resized_overlay[:, :, c]

            # Add label to the frame
            label = "Person"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    # Display person count on the top-right corner
    text = f"Person Count: {person_count}"
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]  # Get text width and height
    text_x = width - text_size[0] - 10  # Adjust X position to fit text within the frame
    text_y = 30  # Fixed Y position near the top
    cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # Write the processed frame to the output RTMP stream
    ffmpeg_process.stdin.write(frame.tobytes())

# Release resources and cleanup
cap.release()
ffmpeg_process.stdin.close()
ffmpeg_process.wait()
cv2.destroyAllWindows()

print("Video processing complete.")

