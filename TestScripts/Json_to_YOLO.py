import os
import json
import shutil
import logging
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np


# Logging setup
log_file = 'dataset_conversion.log'
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(log_file),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

# Configurations
input_dir = './Trainingsdata'  # Replace with your directory
output_dir = './yolo_dataset'  # Replace with your directory
split_ratio = 0.2  # 20% of the data will go to the validation set

# Ask the user if they want to use a split
use_split = input("Would you like to split the dataset into training and validation sets? (yes/no): ").strip().lower() == 'yes'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)
train_dir = os.path.join(output_dir, 'train')
validate_dir = os.path.join(output_dir, 'validate')

os.makedirs(train_dir, exist_ok=True)
if use_split and split_ratio > 0:
    os.makedirs(validate_dir, exist_ok=True)

# Collect JSON and image files
json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
image_files = [f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
logger.info(f"Found {len(json_files)} JSON files and {len(image_files)} image files in input directory.")

# Train-validate split or use only train directory
if use_split and split_ratio > 0:
    train_images, validate_images = train_test_split(image_files, test_size=split_ratio)
    logger.info(f"Split images into {len(train_images)} train and {len(validate_images)} validate images.")
else:
    train_images = image_files
    validate_images = []
    logger.info("Skipping validation split; all images will be used for training.")

# Automatically find all unique labels in the dataset
label_set = set()

# First pass to identify all unique labels
for filename in json_files:
    with open(os.path.join(input_dir, filename)) as f:
        data = json.load(f)
        for shape in data['shapes']:
            label_set.add(shape['label'])

# Create a class label mapping
class_labels = {label: idx for idx, label in enumerate(sorted(label_set))}
logger.info(f"Automatically found class labels: {class_labels}")

# Copy all images to train and validate directories
for image_file in tqdm(image_files, desc="Copying images"):
    current_output_dir = train_dir if image_file in train_images else validate_dir
    try:
        shutil.copy(os.path.join(input_dir, image_file), current_output_dir)
    except Exception as e:
        logger.error(f"Failed to copy image {image_file}: {e}")

# Convert annotations to YOLO format using tqdm for progress bar
for filename in tqdm(json_files, desc="Converting annotations"):
    try:
        with open(os.path.join(input_dir, filename)) as f:
            data = json.load(f)

        image_filename = filename.replace('.json', '')
        matching_image = next((image_filename + ext for ext in ['.jpg', '.png', '.jpeg']
                               if os.path.isfile(os.path.join(input_dir, image_filename + ext))), None)

        if matching_image:
            current_output_dir = train_dir if matching_image in train_images else validate_dir
            out_file_path = os.path.join(current_output_dir, filename.replace('.json', '.txt'))

            with open(out_file_path, 'w') as out_file:
                for shape in data.get('shapes', []):
                    try:
                        label = shape['label']
                        points = np.array(shape['points'], dtype=np.int32)

                        # Calculate bounding box
                        x_min = points[:, 0].min()
                        y_min = points[:, 1].min()
                        x_max = points[:, 0].max()
                        y_max = points[:, 1].max()

                        # Normalize coordinates
                        dw = 1. / data['imageWidth']
                        dh = 1. / data['imageHeight']

                        x_center = (x_min + x_max) / 2
                        y_center = (y_min + y_max) / 2
                        w = x_max - x_min
                        h = y_max - y_min

                        x = x_center * dw
                        y = y_center * dh
                        w = w * dw
                        h = h * dh

                        # Write to the YOLO format output file
                        out_file.write(f"{class_labels[label]} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
                        logger.debug(f"Processed label {label} for {filename} with coordinates x={x:.4f}, y={y:.4f}, w={w:.4f}, h={h:.4f}")

                    except (KeyError, IndexError) as e:
                        logger.warning(f"Error while Processing Ann {filename}: {e}")
                        continue

        else:
            logger.warning(f"No matching image found for JSON file {filename}")

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON file {filename}: {e}")
    except Exception as e:
        logger.error(f"Error processing file {filename}: {e}")

logger.info("Conversion and split completed successfully!")
