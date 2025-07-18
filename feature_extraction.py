# feature_extraction.py
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
import os
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Model performance tracking
class ModelPerformanceTracker:
    def __init__(self):
        self.true_labels = []
        self.pred_labels = []
    
    def add_prediction(self, true_label, pred_label):
        self.true_labels.append(true_label)
        self.pred_labels.append(pred_label)
    
    def generate_report(self):
        if not self.true_labels:
            return "No predictions made yet"
        
        report = classification_report(
            self.true_labels,
            self.pred_labels,
            target_names=["No Mask", "Mask"],
            output_dict=True
        )
        
        # Generate confusion matrix
        cm = confusion_matrix(self.true_labels, self.pred_labels)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=["No Mask", "Mask"],
                    yticklabels=["No Mask", "Mask"])
        plt.title("Mask Detection Confusion Matrix")
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig("mask_confusion_matrix.png", dpi=120, bbox_inches='tight')
        plt.close()
        
        return report

# Initialize performance tracker
performance_tracker = ModelPerformanceTracker()

# Load mask model once at module level
try:
    mask_model = load_model('./models/mask_detector.h5')
    print("Mask detection model loaded successfully")
    
    # Get model input and output details
    _, target_height, target_width, _ = mask_model.input_shape
    output_shape = mask_model.output_shape
    
    # Determine model output type
    if len(output_shape) == 2 and output_shape[1] == 2:
        model_type = "softmax"
    else:
        model_type = "sigmoid"
        
except Exception as e:
    print(f"Error loading mask model: {str(e)}")
    mask_model = None
    target_height, target_width = 224, 224
    model_type = "sigmoid"

def get_augmenter():
    """Create data augmentation generator"""
    return ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest"
    )

def detect_mask(face_image, true_label=None):
    if mask_model is None:
        return "Model Error"
    
    # Resize to model's expected dimensions
    resized_face = cv2.resize(face_image, (target_width, target_height))
    
    # Convert to float and normalize
    normalized = resized_face.astype("float32") / 255.0
    processed = np.expand_dims(normalized, axis=0)
    
    # Make prediction
    try:
        prediction = mask_model.predict(processed, verbose=0)
        
        # Interpret prediction based on model type
        if model_type == "softmax":
            mask_prob = prediction[0][1]  # Assuming index 1 is "Mask"
            no_mask_prob = prediction[0][0]
            pred_label = "No Mask" if mask_prob > 0.5 else "Mask"
        else:
            pred_label = "No Mask" if prediction[0][0] > 0.5 else "Mask"
        
        # Track performance if true label provided
        if true_label is not None:
            performance_tracker.add_prediction(true_label, pred_label)
            
        return pred_label
            
    except Exception as e:
        print(f"No Mask prediction error: {str(e)}")
        return "Prediction Error"

def evaluate_model(test_dir, mask_labels):
    """
    Evaluate model performance on test dataset
    Args:
        test_dir: Directory containing test images
        mask_labels: Dictionary of {image_name: {'mask': true_label}}
    """
    y_true = []
    y_pred = []
    
    for img_name, labels in mask_labels.items():
        img_path = os.path.join(test_dir, img_name)
        image = cv2.imread(img_path)
        if image is None:
            continue
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        true_label = labels['mask']
        
        # Get prediction
        pred_label = detect_mask(image_rgb, true_label=true_label)
        
        if pred_label not in ["Model Error", "Prediction Error"]:
            y_true.append(true_label)
            y_pred.append(pred_label)
    
    # Generate evaluation report
    if y_true:
        report = classification_report(
            y_true, y_pred,
            target_names=["No Mask", "Mask"],
            output_dict=True
        )
        
        # Save confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=["No Mask", "Mask"])
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=["No Mask", "Mask"],
                    yticklabels=["No Mask", "Mask"])
        plt.title("Mask Detection Confusion Matrix")
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig("model_evaluation.png", dpi=120, bbox_inches='tight')
        plt.close()
        
        return report
    return None

def get_performance_metrics():
    """Return current performance metrics as a dictionary"""
    try:
        report = performance_tracker.generate_report()
        
        # Handle case where report might be a string
        if isinstance(report, str):
            return {"error": report}
            
        # Ensure we have the expected structure
        if not isinstance(report, dict):
            return {"error": "Invalid report format"}
            
        return report
        
    except Exception as e:
        return {"error": f"Could not generate metrics: {str(e)}"}