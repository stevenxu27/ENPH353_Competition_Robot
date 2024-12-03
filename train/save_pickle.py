import argparse
from tqdm import tqdm
import pandas as pd
import cv2
import os
import numpy as np
import gc
from multiprocessing import Pool
import pickle

# Paths
csv_path = os.path.expanduser("~/ros_ws/src/controller/train/drive_data_output/velocity_data.csv")
images_path = os.path.expanduser("~/ros_ws/src/controller/train/drive_data_output")
pickle_path = os.path.expanduser("~/ros_ws/src/controller/train/image_data.pkl")

# Load and shuffle dataset
data = pd.read_csv(csv_path, header=None, names=["image", "linear_vel", "angular_vel"])
data = data.sample(frac=1.0).reset_index(drop=True)

def process_image(row):
    img_path = os.path.join(images_path, row["image"])
    try:
        image = cv2.imread(img_path)
        if image is not None:
            image = cv2.resize(image, (224, 224)) / 255.0  # Resizing for 224x224
            return image, [row["linear_vel"], row["angular_vel"]]
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
    return None, None

# Batch processing
batch_size = 500
X = []
y = []

for start_idx in range(0, len(data), batch_size):
    end_idx = min(start_idx + batch_size, len(data))
    batch = data.iloc[start_idx:end_idx]

    with Pool(processes=4) as pool:
        results = list(tqdm(pool.imap(process_image, batch.to_dict(orient="records")), total=len(batch)))

    # Collect results
    batch_images, batch_velocities = zip(*[(img, vel) for img, vel in results if img is not None])
    X.extend(batch_images)
    y.extend(batch_velocities)

    # Cleanup
    del batch_images, batch_velocities
    gc.collect()

# Convert to numpy arrays
X = np.array(X, dtype="float32")
y = np.array(y, dtype="float32")

print(f"Shape of images: {X.shape}")
print(f"Shape of velocities: {y.shape}")

with open(pickle_path, "wb") as f:
    pickle.dump((X, y), f)

print(f"Data saved to {pickle_path}")