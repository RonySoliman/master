# image_processing.py
import cv2
import numpy as np
from face_detection import detect_faces_from_image
from feature_extraction import detect_mask
from age_estimation import estimate_age

# Outfit comparison functions
def get_outfit_signature(image, bbox):
    """Generate a simplified outfit signature using average color"""
    x1, y1, x2, y2 = bbox
    height = image.shape[0]
    # Get region below face (upper body/outfit)
    y_start = min(y2 + 10, height)
    y_end = min(y2 + 100, height)  # Smaller region for speed
    
    if y_start >= y_end:
        return None
    
    outfit_region = image[y_start:y_end, x1:x2]
    
    if outfit_region.size == 0:
        return None
    
    # Use average color instead of histogram for speed
    avg_color = np.mean(outfit_region, axis=(0, 1))
    return avg_color

def compare_outfits(sig1, sig2, threshold=20):
    """Compare two outfit signatures using color difference"""
    if sig1 is None or sig2 is None:
        return False
    # Calculate Euclidean distance between colors
    color_diff = np.linalg.norm(sig1 - sig2)
    return color_diff < threshold

def process_frame(frame, known_outfits, frame_index=None, outfit_threshold=20):
    # Detect faces
    _, face_locs = detect_faces_from_image(frame)
    
    annotated_frame = frame.copy()
    frame_has_new_outfit = False
    
    for i, loc in enumerate(face_locs):
        top, right, bottom, left = loc
        face_img = frame[top:bottom, left:right]
        
        # Skip if empty face region
        if face_img.size == 0:
            continue
            
        # Get predictions
        mask_status = detect_mask(face_img)
        age = estimate_age(face_img)
        
        # Get outfit signature (much faster now)
        outfit_signature = get_outfit_signature(frame, (left, top, right, bottom))
        
        # Check if outfit is new
        outfit_is_new = True
        if outfit_signature is not None:
            for known_sig in known_outfits:
                if compare_outfits(outfit_signature, known_sig, outfit_threshold):
                    outfit_is_new = False
                    break
            if outfit_is_new:
                known_outfits.append(outfit_signature)
                frame_has_new_outfit = True
        
        # Draw annotations with RED box
        label = f"{mask_status} | {age}"
        cv2.rectangle(annotated_frame, (left, top), (right, bottom), (0, 0, 225), 2)  # Red color
        cv2.putText(annotated_frame, label, (left, top-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 225), 2)  # Red color
        
        # Save mask images only for new outfits
        if mask_status == "Mask" and frame_index is not None and outfit_is_new:
            mask_path = f"output/masks/mask_{frame_index}_{i}.jpg"
            cv2.imwrite(mask_path, cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))
    
    return annotated_frame, frame_has_new_outfit