# NeuroVision — Real-Time AI Object Detection

A real-time computer vision app that runs entirely in your browser. Point your webcam at anything and NeuroVision identifies objects instantly using a neural network — no server, no API keys, no data ever leaves your device.

## Live Demo

Enable GitHub Pages on this repo (Settings → Pages → Deploy from branch `main`) to get a shareable live link, or simply open `index.html` locally in a modern browser.

## Features

- Real-time object detection at interactive frame rates using the COCO-SSD model
- - Runs 100% client-side with TensorFlow.js — your video never touches a server
  - - Live HUD showing FPS, inference latency, and object count
    - - Color-coded bounding boxes with confidence scores
      - - Live detections list updated every frame
        - - Snapshot tool to save an annotated frame as a PNG
          - - Responsive, modern dark UI designed to demo well
           
            - ## Tech Stack
           
            - - HTML5 / CSS3 / Vanilla JavaScript
              - - [TensorFlow.js](https://www.tensorflow.org/js)
                - - [COCO-SSD](https://github.com/tensorflow/tfjs-models/tree/master/coco-ssd) pre-trained object detection model
                 
                  - ## Getting Started
                 
                  - 1. Clone this repository
                    2. 2. Open `index.html` in a modern browser (Chrome, Edge, or Firefox recommended)
                       3. 3. Click "Start Camera" and allow webcam access
                          4. 4. Point your camera at everyday objects and watch NeuroVision identify them in real time
                            
                             5. No build step, no dependencies to install — it just works.
                            
                             6. ## Why This Problem Matters
                            
                             7. Real-time perception is at the core of robotics, autonomous vehicles, accessibility tools, and augmented reality. This project demonstrates how far in-browser machine learning has come: a full object-detection pipeline running smoothly on ordinary consumer hardware, with zero backend infrastructure.
                            
                             8. ## License
                            
                             9. This project is open for learning and demonstration purposes.
