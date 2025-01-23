import cv2
import threading
import requests
from ultralytics import YOLO
from azure.storage.blob import BlobServiceClient
from datetime import datetime
import time

# Function to upload the annotated frame to Azure Blob Storage and send data to POST API
def upload_to_azure_and_post(frame, azure_connection_string, container_name, cameradid, post_api_url, an_id, person_count):
    try:
        blob_service_client = BlobServiceClient.from_connection_string(azure_connection_string)
        container_client = blob_service_client.get_container_client(container_name)
        
        # Create a dynamic filename with the current date and time
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        blob_name = f"live-record/arista8/Person_{timestamp}.jpg"
        
        # Convert the frame to JPEG format
        _, img_encoded = cv2.imencode('.jpg', frame)
        img_bytes = img_encoded.tobytes()
        
        # Upload the annotated frame to Azure Blob Storage
        blob_client = container_client.get_blob_client(blob_name)
        blob_client.upload_blob(img_bytes, overwrite=True)
        
        print(f"Frame uploaded successfully: {blob_name}")
        
        # Construct the image URL
        image_url = f"https://{blob_service_client.account_name}.blob.core.windows.net/{container_name}/{blob_name}"
        send_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.000")
        
        # Send data to POST API
        data = {
            "cameradid": cameradid,
            "sendtime": send_time,
            "imgurl": image_url,
            "an_id": an_id,
            "ImgCount": person_count  # Send the actual person count
        }
        headers = {'Content-Type': 'application/json'}
        response = requests.post(post_api_url, json=data, headers=headers)
        if response.status_code == 200:
            print(f"Data sent to POST API successfully: {response.json()}")
        else:
            print(f"POST API error: {response.status_code}, {response.text}")
    except Exception as e:
        print(f"Error uploading frame or sending data to POST API: {e}")

# Function to open RTMP stream
def open_rtmp_stream(rtmp_url):
    cap = cv2.VideoCapture(rtmp_url)
    if not cap.isOpened():
        print("Error: Unable to open RTMP stream.")
        return None
    return cap

# Set parameters
rtmp_input_url = 'rtmp://ptz.vmukti.com:80/live-record/VSPL-100005-CDBWA'
azure_connection_string = "BlobEndpoint=https://nvrdatashinobi.blob.core.windows.net/;QueueEndpoint=https://nvrdatashinobi.queue.core.windows.net/;FileEndpoint=https://nvrdatashinobi.file.core.windows.net/;TableEndpoint=https://nvrdatashinobi.table.core.windows.net/;SharedAccessSignature=sv=2022-11-02&ss=bfqt&srt=sco&sp=rwdlacupiytfx&se=2025-01-10T14:44:50Z&st=2025-01-08T06:44:50Z&spr=https,http&sig=7di0L6jZatcA5agFdF8Dm3ozpNKCSRYygmd8JQezKn0%3D"
container_name = "nvrdatashinobi"
post_api_url = "https://tn2023demo.vmukti.com/api/analytics/"
an_id = 2

# Extract camera DID from RTMP input URL
cameradid = rtmp_input_url.split('/')[-1]

# Load the YOLOv8 model
model = YOLO('best.pt')  # Path to your custom trained model (best.pt)

# Start RTMP stream processing
cap = open_rtmp_stream(rtmp_input_url)
if cap is None:
    print("RTMP feed not available.")
    exit()

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Stream disconnected. Retrying in 5 seconds...")
        cap.release()
        time.sleep(5)
        cap = open_rtmp_stream(rtmp_input_url)
        if cap is None:
            continue

    # Process every 5th frame
    frame_count += 1
    if frame_count % 5 != 0:
        continue

    # Run YOLOv8 model for human detection
    results = model(frame)  # This returns a list of results
    
    # Initialize person count
    person_count = 0

    # Extract bounding boxes and draw them on the frame
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Extract coordinates and class
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            conf = box.conf.item()  # Confidence score (convert tensor to float)
            cls = int(box.cls)  # Class index

            # Increment person count for class 'person' (assuming 'person' class index is 0)
            if cls == 0:  # Adjust based on your model's class mappings
                person_count += 1
                # Draw bounding box and label on the frame
                label = f"Person"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display person count on the top-right corner of the frame
    cv2.putText(
        frame, f"Person_Count: {person_count}",
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2
    )

    # If persons are detected, upload the annotated frame to Azure and send data to POST API
    if person_count > 0:
        threading.Thread(
            target=upload_to_azure_and_post,
            args=(frame, azure_connection_string, container_name, cameradid, post_api_url, an_id, person_count),
            daemon=True
        ).start()

    # Display the frame for visualization (optional)
   # cv2.imshow("RTMP Stream", frame)
   # if cv2.waitKey(1) & 0xFF == ord('q'):
   #     break

# Release resources and cleanup
cap.release()
cv2.destroyAllWindows()
print("Video processing complete.")

