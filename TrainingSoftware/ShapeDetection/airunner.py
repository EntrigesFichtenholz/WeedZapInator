import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from PIL import Image
import os
def load_images(directory):
    images = []
    for filename in os.listdir(directory):
        if filename.endswith((".jpg", ".png")):
            image_path = os.path.join(directory, filename)
            image = cv2.imread(image_path)
            if image is not None:
                images.append((image_path, image))
    return images
def detect_objects(model, image):
    # Convert results to the expected format
    results = model(image)
    return results[0]  # Return the first result object
def draw_bounding_boxes(result, image, image_path, output_dir):
    # Get the output filename
    output_filename = os.path.join(
        output_dir,
        f"outputimage{os.path.basename(os.path.splitext(image_path)[0])}_detected.png"
    )
    # Create a copy of the image to draw on
    annotated_image = image.copy()
    # Draw boxes for all detections
    boxes = result.boxes
    for box in boxes:
        # Get coordinates
        x1, y1, x2, y2 = box.xyxy[0]
        # Convert coordinates to integers
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        # Draw rectangle
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Get class name and confidence
        if hasattr(box, 'cls') and hasattr(box, 'conf'):
            class_id = int(box.cls[0])
            class_name = result.names[class_id]  # Get class name from model's names dictionary
            conf = float(box.conf[0])
            # Create label with class name and confidence
            label = f"{class_name}: {conf:.2f}"

            # Add background rectangle for text
            (label_width, label_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(annotated_image,
                        (x1, y1-label_height-10),
                        (x1+label_width, y1),
                        (255, 0, 0),
                        -1)  # Filled rectangle

            # Add text
            cv2.putText(annotated_image,
                       label,
                       (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.5,
                       (255, 255, 255),  # White text
                       2)

    # Save the annotated image
    cv2.imwrite(output_filename, annotated_image)
    # Display the image
    cv2.imshow("Detected Objects", annotated_image)
    cv2.waitKey(1)  # Wait for 1ms instead of indefinitely
    return output_filename
def main():
    # Load images from directory
    images_dir = "./Gen_output_images/validate/"
    images = load_images(images_dir)
    if not images:
        print("No images found in the specified directory!")
        return
    # Initialize YOLO model
    model = YOLO("./savemodels/150ITshapeDetect11n.pt")
    # Create output directory
    output_dir = "detected_images"
    os.makedirs(output_dir, exist_ok=True)
    # Process images
    for i, (image_path, image) in enumerate(images[:10]):
        try:
            # Perform detection
            result = detect_objects(model, image)
            # Draw bounding boxes and save result
            output_filename = draw_bounding_boxes(result, image, image_path, output_dir)
            print(f"Image {i+1} processed and saved as {os.path.basename(output_filename)}")
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
    # Close all windows at the end
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
