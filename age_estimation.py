import cv2
import numpy as np
import os

# Age buckets for classification
AGE_BUCKETS = ["(0-15)", "(16-22)", "(23-29)", "(30-35)", "(36-42)", "(43-50)", "(51-64)", "(65-80)"]

# Initialize age model
age_net = None

def load_age_model():
    global age_net
    if age_net is not None:
        return
    
    try:
        # Use relative path that works with your structure
        prototxt = "./models/deploy_age.prototxt"  # Note the filename change
        caffemodel = "./models/age_net.caffemodel"
        
        # Check if files exist
        if not os.path.exists(prototxt):
            raise FileNotFoundError(f"Prototxt file not found: {prototxt}")
        if not os.path.exists(caffemodel):
            raise FileNotFoundError(f"Model file not found: {caffemodel}")
            
        age_net = cv2.dnn.readNet(prototxt, caffemodel)
        print("Age model loaded successfully")
    except Exception as e:
        print(f"Error loading age model: {str(e)}")
        age_net = None

def estimate_age(face_img):
    if age_net is None:
        load_age_model()
        if age_net is None:
            return "Age Model Error"
    
    try:
        # Preprocess face for age model - UPDATED MEAN VALUES
        blob = cv2.dnn.blobFromImage(
            face_img, 
            scalefactor=1.0,
            size=(227, 227),
            mean=(78.4263377603, 87.7689143744, 114.895847746),
            swapRB=False  # OpenCV uses BGR by default
        )
        
        # Make prediction
        age_net.setInput(blob)
        preds = age_net.forward()
        i = preds[0].argmax()
        age = AGE_BUCKETS[i]
        return age
        
    except Exception as e:
        print(f"Age estimation error: {str(e)}")
        return "Age Error"