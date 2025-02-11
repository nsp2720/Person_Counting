import cv2
import threading
import requests
from ultralytics import YOLO
from azure.storage.blob import BlobServiceClient
from datetime import datetime
import time
import gi
import numpy as np

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GObject
import os

# Enable GStreamer debug logs (optional)
os.environ["GST_DEBUG"] = "3"

# Initialize GStreamer
Gst.init(None)

# Dictionary to store GStreamer pipelines for each camera
rtmp_pipelines = {}
appsrcs = {}

def create_rtmp_pipeline(output_rtmp_url, width, height, fps):
    pipeline_str = f"""
    appsrc name=source format=GST_FORMAT_TIME is-live=true do-timestamp=true !
    capsfilter caps=video/x-raw,format=RGB,width={width},height={height},framerate={fps}/1 !
    queue max-size-buffers=5 max-size-time=0 max-size-bytes=0 leaky=downstream !
    videoconvert ! video/x-raw, format=I420 !
    queue max-size-buffers=5 max-size-time=0 max-size-bytes=0 leaky=downstream !
    x264enc speed-preset=veryfast tune=zerolatency bitrate=1500 key-int-max=30 bframes=0 rc-lookahead=0 !
    video/x-h264, profile=high !
    queue max-size-buffers=5 max-size-time=0 max-size-bytes=0 leaky=downstream !
    flvmux streamable=true !
    queue max-size-buffers=5 max-size-time=0 max-size-bytes=0 leaky=downstream !
    rtmp2sink location={output_rtmp_url}
    """
    try:
        pipeline = Gst.parse_launch(pipeline_str)
        source = pipeline.get_by_name("source")
        return pipeline, source
    except Exception as e:
        print(f"Error creating GStreamer pipeline: {e}")
        return None, None
    
# Function to upload the annotated frame to Azure Blob Storage and send data to POST API
def upload_to_azure_and_post(frame, azure_connection_string, container_name, cameradid, post_api_url, an_id, person_count):
    try:
        blob_service_client = BlobServiceClient.from_connection_string(azure_connection_string)
        container_client = blob_service_client.get_container_client(container_name)
        
        # Create a dynamic filename with the current date and time
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        blob_name = f"live-record/arista8/Person_{timestamp}_{cameradid}.jpg"
        
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
        print(f"Error: Unable to open RTMP stream: {rtmp_url}")
        return None
    return cap

# Function to process video stream
def process_stream(rtmp_url, azure_connection_string, container_name, post_api_url, an_id, model, last_api_call_time, rtmp_output_feeds):
    #cameradid = rtmp_url.split('/')[-1] # Removed to fix the issue
    cap = open_rtmp_stream(rtmp_url)
    if cap is None:
        print(f"RTMP feed not available for {rtmp_url}.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(f"Stream {rtmp_url}: Width={frame_width}, Height={frame_height}, FPS={fps}")

    # Retrieve the corresponding output feed URL for the input stream
    output_rtmp_url = rtmp_output_feeds.get(rtmp_url)
    if not output_rtmp_url:
      print(f"No matching RTMP output URL found for {rtmp_url}")
      return
    # Create GStreamer pipeline if not already created
    if rtmp_url not in rtmp_pipelines: # Changed here
        pipeline, source = create_rtmp_pipeline(output_rtmp_url,frame_width,frame_height,fps)
        if pipeline is None:
            print(f"Unable to create GStreamer pipeline for {rtmp_url}")
            return
        rtmp_pipelines[rtmp_url] = pipeline # Changed here
        appsrcs[rtmp_url] = source # Changed here
        pipeline.set_state(Gst.State.PLAYING)
    else:
        pipeline = rtmp_pipelines[rtmp_url] # Changed here
        source = appsrcs[rtmp_url] # Changed here
    
    
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Stream disconnected for {rtmp_url}. Retrying in 5 seconds...") # Changed here
            cap.release()
            time.sleep(5)
            cap = open_rtmp_stream(rtmp_url)
            if cap is None:
                continue
            
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            print(f"Stream {rtmp_url}: Width={frame_width}, Height={frame_height}, FPS={fps}") # Changed here

            if rtmp_url in rtmp_pipelines: # Changed here
                # If the pipeline exists stop it and remove it
                rtmp_pipelines[rtmp_url].set_state(Gst.State.NULL) # Changed here
                del rtmp_pipelines[rtmp_url] # Changed here
                del appsrcs[rtmp_url] # Changed here
                
            # Create GStreamer pipeline with new stream resolution
            pipeline, source = create_rtmp_pipeline(output_rtmp_url,frame_width,frame_height,fps)
            if pipeline is None:
                print(f"Unable to create GStreamer pipeline for {rtmp_url}")
                continue
            rtmp_pipelines[rtmp_url] = pipeline # Changed here
            appsrcs[rtmp_url] = source # Changed here
            pipeline.set_state(Gst.State.PLAYING)
            continue # restart the loop

        try:
          # Run YOLOv8 model for human detection
          results = model(frame)
          
          # Initialize person count
          person_count = 0

          # Extract bounding boxes and draw them on the frame
          for result in results:
              boxes = result.boxes
              for box in boxes:
                  x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                  cls = int(box.cls)  # Class index

                  # Increment person count for class 'person' (assuming 'person' class index is 0)
                  if cls == 0:  # Adjust based on your model's class mappings
                      person_count += 1
                      # Draw bounding box and label on the frame
                      label = f"Person"
                      cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                      cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

          # Convert the frame to RGB for GStreamer
          frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
          # Convert the frame to bytes
          frame_bytes = frame_rgb.tobytes()
          
          # Create a GStreamer buffer and push the frame
          buffer = Gst.Buffer.new_allocate(None,len(frame_bytes),None)
          buffer.fill(0,frame_bytes)
          # Push the buffer into the pipeline
          source.emit("push-buffer", buffer)


          # If persons are detected, check API call timing
          current_time = time.time()
          if person_count > 0 and (rtmp_url not in last_api_call_time or current_time - last_api_call_time[rtmp_url] >= 3): # Changed here
              last_api_call_time[rtmp_url] = current_time # Changed here
              cameradid = rtmp_url.split('/')[-1]
              threading.Thread(
                  target=upload_to_azure_and_post,
                  args=(frame, azure_connection_string, container_name, cameradid, post_api_url, an_id, person_count),
                  daemon=True
              ).start()
        except Exception as e:
           print(f"Error processing frame {rtmp_url}: {e}")

    
    # Release resources and cleanup
    cap.release()
    cv2.destroyAllWindows()
    print(f"Video processing complete for {rtmp_url}.") # Changed here

# Main function to handle multiple cameras
if __name__ == "__main__":
    # Set parameters
    rtmp_input_urls = [
        'rtmp://mediaserver.vmukti.com/live/stream_243',
        'rtsp://admin:@192.168.1.39:554/ch0_0.264'
    ]
    
    # Define corresponding output RTMP URLs for each input URL
    rtmp_output_feeds = {
      'rtmp://mediaserver.vmukti.com/live/stream_243': 'rtmp://ptz.vmukti.com:80/live-record/detection_1',
      'rtsp://admin:@192.168.1.39:554/ch0_0.264': 'rtmp://ptz.vmukti.com:80/live-record/detection_2'
    }
    
    azure_connection_string = "BlobEndpoint=https://nvrdatashinobi.blob.core.windows.net/;QueueEndpoint=https://nvrdatashinobi.queue.core.windows.net/;FileEndpoint=https://nvrdatashinobi.file.core.windows.net/;TableEndpoint=https://nvrdatashinobi.table.core.windows.net/;SharedAccessSignature=sv=2022-11-02&ss=bfqt&srt=sco&sp=rwdlacupiytfx&se=2025-02-20T13:17:35Z&st=2025-01-20T05:17:35Z&spr=https,http&sig=itFAeolMU9KD7OQlxeYLOCMjjFmO1Pc%2FaxMFweniorE%3D"
    container_name = "nvrdatashinobi"
    post_api_url = "https://tn2023demo.vmukti.com/api/analytics/"
    an_id = 2

    # Load the YOLOv8 model
    model = YOLO('best.pt')  # Path to your custom trained model (best.pt)

    # Dictionary to track the last API call time for each camera
    last_api_call_time = {}

    # Start threads for each camera stream
    threads = []
    for rtmp_url in rtmp_input_urls:
        t = threading.Thread(
            target=process_stream,
            args=(rtmp_url, azure_connection_string, container_name, post_api_url, an_id, model, last_api_call_time, rtmp_output_feeds),
            daemon=True
        )
        threads.append(t)
        t.start()

    # Wait for all threads to complete
    for t in threads:
        t.join()

