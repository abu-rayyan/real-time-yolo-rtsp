# Real-time Object Detection and Streaming with YOLO and RTSP

This project demonstrates real-time object detection using YOLO (You Only Look Once) and streaming the processed video frames over RTSP (Real-Time Streaming Protocol). YOLO is a deep learning model known for its fast and accurate object detection capabilities. The persons detected are marked with bounding boxes and labels, and the results are streamed to an RTSP server for live viewing and also creating a csv for logs.

## Prerequisites

- Python 3.8
- OpenCV
- [Ultralytics YOLO](https://github.com/ultralytics/yolov5) library
- FFmpeg
- Docker

## Setup

1. Clone this repository:

```bash
git clone https://github.com/abu-rayyan/real-time-yolo-rtsp.git
cd real-time-yolo-rtsp/
```

2. Install the required dependencies. It's recommended to use a virtual environment:

```
python -m venv venv
source venv/bin/activate  # On Windows, use "venv\Scripts\activate"
pip install -r requirements.txt
```

3. Open a new terminal and Download and launch the docker image by running the below command:
```
docker run --rm -it --network=host bluenviron/mediamtx:latest
```
Now your RTSP Server is up. You can send data on the rtsp server (on the port 8554).  


## Usage
1. Now go back to your IDE terminal and run the below command to see the magic:
```
python main.py
```
The script will perform real-time object detection on the video stream and stream the processed frames to the RTSP server.

Note: Before that step make sure to replace <your_ip_address> with your system's IP address   

2. Open VLC to check the inference live stream. Navigate to `Media` and click on the `Open Network Streem...` and add your rtsp url there.

Example:
`rtsp://<your_ip_address>:8554/inference_live_stream` 
 
 Note: Replace <your_ip_address> with your system's IP address where rtsp server is configured.
###### If you want to access the inference stream URL globally then you will have to enable port forwarding.

## Output
1. Detected Persons are marked with bounding boxes and labelsLicense
This project is licensed under the MIT License. Feel free to use and modify it for your needs.
2. A CSV file named person_detections.csv is created, containing timestamped data about detected persons.

## Acknowledgments
This project was inspired by the need for real-time object detection and streaming for various applications.
