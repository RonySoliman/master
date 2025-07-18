
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from feature_extraction import detect_mask

def evaluate_system(test_dir, mask_labels):
    test_faces = []
    y_true_mask = []
    
    # Load test images and labels
    for img_name, labels in mask_labels.items():
        img_path = os.path.join(test_dir, img_name)
        image = cv2.imread(img_path)
        if image is None:
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # For simplicity, assume one face per test image
        test_faces.append(image_rgb)
        y_true_mask.append(labels['mask'])
    
    # Generate predictions
    y_pred_mask = [detect_mask(face) for face in test_faces]
    
    # Calculate metrics
    mask_acc = accuracy_score(y_true_mask, y_pred_mask)
    mask_cm = confusion_matrix(y_true_mask, y_pred_mask, labels=["Mask", "No Mask"])
    
    # Visualization
    plt.figure(figsize=(8, 6))
    sns.heatmap(mask_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Mask", "No Mask"],
                yticklabels=["Mask", "No Mask"])
    plt.title(f"Mask Detection\nAccuracy: {mask_acc:.2f}")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # Final layout adjustments
    plt.tight_layout(pad=2.0)
    plt.savefig("performance_metrics.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Mask Detection Accuracy: {mask_acc:.2%}")
    print("-"*50)
    print(f"Tested on {len(test_faces)} images")
    print("="*50)

# Example usage (replace with actual labeled data)
if __name__ == "__main__":
    # Mock data structure: {image_name: {'mask': label}}
    test_labels = {
        "person1.jpg": {"mask": "Mask"},
        "person2.jpg": {"mask": "No Mask"},
        "person3.jpg": {"mask": "Mask"}
    }
    
    evaluate_system("../test_data/", test_labels)