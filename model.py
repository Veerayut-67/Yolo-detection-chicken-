from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image

# Load a model
model = YOLO("yolov8n.pt")  # load an official model

results = model(["https://ultralytics.com/images/bus.jpg"], conf=0.3)  # return a list of Results objects

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    result.show()  # display to screen
    result.save(filename="result.jpg")  # save to disk
