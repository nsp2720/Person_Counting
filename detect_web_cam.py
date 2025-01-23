import cv2
import subprocess
import time
from ultralytics import YOLO
import numpy as np

# Function to start FFmpeg process (for streaming, if needed)
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

# Load the YOLOv8 model
model = YOLO('best.pt')  # Path to your custom trained model (best.pt)

# Set transparency factor (between 0 and 1)
transparency_factor = 0.4  # Adjust this value for desired transparency (e.g., 0.5 for 50% opacity)

# Load the custom bounding box overlay image (box.png) with an alpha channel
overlay_img = cv2.imread('box.png', cv2.IMREAD_UNCHANGED)

# Ensure overlay has an alpha channel; add one if necessary
if overlay_img.shape[2] == 3:
    alpha_channel = np.ones(overlay_img.shape[:2], dtype=overlay_img.dtype) * 255
    overlay_img = np.dstack([overlay_img, alpha_channel])

# Scale the alpha channel by the transparency factor to reduce transparency
overlay_img[:, :, 3] = (overlay_img[:, :, 3] * transparency_factor).astype(overlay_img.dtype)

# Open the webcam
cap = cv2.VideoCapture(0)  # 0 is usually the default webcam device ID
if not cap.isOpened():
    print("Error: Unable to access the webcam.")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
fps = cap.get(cv2.CAP_PROP_FPS) or 25

# Start the FFmpeg process (optional for streaming)
# If you're not streaming, you can skip this section
output_rtmp_url = 'rtmp://4.240.85.196:80/live/modified_stream/VSPL-103247-HYQPO'
ffmpeg_process = start_ffmpeg_process(output_rtmp_url, width, height, fps)
if ffmpeg_process is None:
    print("Error starting FFmpeg process.")
    exit()

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Error reading frame from webcam.")
        break

    # Run YOLOv8 model for detection
    results = model(frame)  # This returns a list of results
    
    # Extract bounding boxes and confidences from the results
    for result in results:
        boxes = result.boxes  # The first result contains the detections
        
        for box in boxes:
            # Get coordinates of the bounding box [xmin, ymin, xmax, ymax]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Get the confidence score
            conf = box.conf[0]  # Confidence score for the bounding box
            
            # Create label with confidence
            label = f"Person {conf:.2f}"  # Label with confidence score

            # Resize the overlay image to fit the bounding box size
            box_width = x2 - x1
            box_height = y2 - y1
            resized_overlay = cv2.resize(overlay_img, (box_width, box_height))

            # Get the alpha mask for the overlay image (transparency)
            alpha_mask = resized_overlay[:, :, 3] / 255.0  # Alpha channel for transparency

            # Overlay the image with transparency applied
            for c in range(3):  # Loop through each color channel
                frame[y1:y2, x1:x2, c] = (1 - alpha_mask) * frame[y1:y2, x1:x2, c] + alpha_mask * resized_overlay[:, :, c]

            # Add label to the frame
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the resulting frame with overlay
   # cv2.imshow('Webcam Feed with YOLO', frame)

    # Write the processed frame to the output RTMP stream (if needed)
    if ffmpeg_process:
        ffmpeg_process.stdin.write(frame.tobytes())

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources and cleanup
cap.release()
if ffmpeg_process:
    ffmpeg_process.stdin.close()
    ffmpeg_process.wait()
cv2.destroyAllWindows()

print("Video processing complete.")

