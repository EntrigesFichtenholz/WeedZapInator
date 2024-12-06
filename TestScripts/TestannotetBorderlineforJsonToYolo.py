import os
import json
import random
import cv2
import numpy as np
from pathlib import Path

def visualize_random_annotation():
    """
    Selects a random image from the dandelion directory and visualizes its annotations.
    """
    # Get all JSON files from dandelion directory
    data_dir = Path('dandelion')
    json_files = list(data_dir.glob('*.json'))

    if not json_files:
        print("Keine JSON-Dateien im dandelion Verzeichnis gefunden!")
        return

    # Select random JSON file
    json_path = random.choice(json_files)

    # Find matching image file
    image_path = None
    for ext in ['.jpg', '.jpeg', '.png']:
        temp_path = json_path.with_suffix(ext)
        if temp_path.exists():
            image_path = temp_path
            break

    if image_path is None:
        print(f"Kein passendes Bild für {json_path} gefunden!")
        return

    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Konnte Bild nicht lesen: {image_path}")
        return

    # Read annotations
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Fehler beim Lesen der JSON-Datei {json_path}: {e}")
        return

    # Create a copy of the image for visualization
    vis_image = image.copy()

    # Dictionary of colors for different classes
    colors = {
        'loewenzahn': (0, 255, 0),  # Grün
        'loewenzahn_Blüte': (0, 0, 255)  # Rot
    }

    # Process each annotation
    for shape in data.get('shapes', []):
        try:
            label = shape['label']
            points = np.array(shape['points'], dtype=np.int32)

            # Get color for this class (default to white if not in dictionary)
            color = colors.get(label, (255, 255, 255))

            # Calculate bounding box
            x_min = points[:, 0].min()
            y_min = points[:, 1].min()
            x_max = points[:, 0].max()
            y_max = points[:, 1].max()

            # Draw bounding box
            cv2.rectangle(vis_image, (x_min, y_min), (x_max, y_max), color, 2)

            # Add label text
            # Calculate text size for better positioning
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]

            # Draw text background
            cv2.rectangle(vis_image,
                         (x_min, y_min - text_size[1] - 10),
                         (x_min + text_size[0], y_min),
                         color,
                         -1)  # Filled rectangle

            # Draw text
            cv2.putText(vis_image,
                       label,
                       (x_min, y_min - 5),
                       font,
                       font_scale,
                       (255, 255, 255),  # White text
                       thickness)

        except (KeyError, IndexError) as e:
            print(f"Fehler bei der Verarbeitung einer Annotation: {e}")
            continue

    # Calculate new dimensions while maintaining aspect ratio
    max_dimension = 1200
    height, width = vis_image.shape[:2]
    scaling_factor = min(max_dimension / width, max_dimension / height)

    new_width = int(width * scaling_factor)
    new_height = int(height * scaling_factor)

    # Resize image for display
    vis_image = cv2.resize(vis_image, (new_width, new_height))

    # Show image
    window_name = f"Annotationen - {image_path.name}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, vis_image)

    print("\nTasten:")
    print("- 's': Bild speichern")
    print("- 'n': Nächstes zufälliges Bild")
    print("- 'q': Beenden")

    while True:
        key = cv2.waitKey(0) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save visualization
            output_path = data_dir / f"visualized_{image_path.name}"
            cv2.imwrite(str(output_path), vis_image)
            print(f"Bild gespeichert als: {output_path}")
        elif key == ord('n'):
            cv2.destroyWindow(window_name)
            visualize_random_annotation()
            break

    cv2.destroyWindow(window_name)

if __name__ == "__main__":
    try:
        visualize_random_annotation()
    finally:
        cv2.destroyAllWindows()
