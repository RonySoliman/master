# face_detection.py
import cv2
import face_recognition

def detect_faces_from_image(image, max_dimension=2000):
    height, width = image.shape[:2]
    
    if max(height, width) > max_dimension:
        scale = max_dimension / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        resized_image = cv2.resize(image, (new_width, new_height))
        face_locations = face_recognition.face_locations(resized_image)
        
        original_face_locations = []
        for top, right, bottom, left in face_locations:
            top = int(top / scale)
            right = int(right / scale)
            bottom = int(bottom / scale)
            left = int(left / scale)
            original_face_locations.append((top, right, bottom, left))
        return image, original_face_locations
    else:
        face_locations = face_recognition.face_locations(image)
        return image, face_locations