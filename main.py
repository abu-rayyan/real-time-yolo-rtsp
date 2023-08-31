import subprocess as sp
import cv2
import numpy as np
import math
import csv
from ultralytics import YOLO
from datetime import datetime

# Load YOLO model
model = YOLO("yolo-Weights/yolov8n.pt")

# List of class names for object detection
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]
# Open a CSV file for writing detections
csv_file = open("person_detections.csv", mode="w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Timestamp", "Frame Number", "X1", "Y1", "X2", "Y2"])

# Initialize frame number
frame_number = 0
# Set up the RTSP input stream
rtsp_url = "your_rtsp_input_url or your webcam or your video"
cap = cv2.VideoCapture(rtsp_url)
# Get the actual width and height of the video frames
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# FFmpeg command to encode and stream frames to the RTSP server
ffmpeg_command = [
    "ffmpeg",
    "-f", "rawvideo",
    "-pix_fmt", "bgr24",
    "-video_size", "{}x{}".format(frame_width, frame_height),  # Replace with actual width and height
    "-i", "-",  # Input from pipe
    "-c:v", "libx264",
    "-preset", "ultrafast",
    "-f", "rtsp",
    "-rtsp_transport", "tcp",  # Use TCP transport
    "rtsp://<your_ip_address>:8554/inference_live_stream"  # Replace <your_ip_address> with your system's IP address
]

# Start the FFmpeg process
ffmpeg_process = sp.Popen(ffmpeg_command, stdin=sp.PIPE)

# Capture and process frames
while cap.isOpened():
    success, img = cap.read()
    results = model(img, stream=True)
    frame_number += 1  # Increment frame number

    # coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            cls = int(box.cls[0])
            
            if classNames[cls] == "person":
                # bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values
                # Get current timestamp
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                # Write data to CSV
                csv_writer.writerow([timestamp, frame_number, x1, y1, x2, y2])
                # put box in cam
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                # confidence
                confidence = math.ceil((box.conf[0]*100))/100
                print("Confidence --->",confidence)
                # class name
                cls = int(box.cls[0])
                print("Class name -->", classNames[cls])
                # object details
                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2
                cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)
    # Send the processed frame to FFmpeg for streaming
    if ffmpeg_process.poll() is None:
        ffmpeg_process.stdin.write(img.tobytes())
    else:
        break

# Close the FFmpeg process and release resources
ffmpeg_process.stdin.close()
ffmpeg_process.wait()
cap.release()
cv2.destroyAllWindows()
