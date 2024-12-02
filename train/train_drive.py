from tqdm import tqdm
import pandas as pd
import cv2
import os
import numpy as np
import gc
from multiprocessing import Pool
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping

csv_path = os.path.expanduser("~/ros_ws/src/controller/train/drive_data_output/velocity_data.csv")
images_path = os.path.expanduser("~/ros_ws/src/controller/train/drive_data_output")

data = pd.read_csv(csv_path, header=None, names=["image", "linear_vel", "angular_vel"])
data = data.sample(frac=1.0, random_state=42).reset_index(drop=True)  # Shuffle rows

def process_image(row):
    img_path = os.path.join(images_path, row["image"])
    try:
        image = cv2.imread(img_path)
        if image is not None:
            image = cv2.resize(image, (128, 128)) / 255.0
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

# Path to the .h5 file
model_path = os.path.expanduser("~/ros_ws/src/controller/models/drive_model.h5")

# Load the model
model = load_model(model_path)

# Define the EarlyStopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',      # Metric to monitor (e.g., validation loss)
    patience=10,              # Number of epochs with no improvement before stopping
    verbose=1,               # Print a message when training stops
    restore_best_weights=True  # Restore model weights from the epoch with the best value
)

history = model.fit(
    X, y,  # Use the entire dataset without a separate validation set
    validation_split=0.2,  # Use 20% of the data for validation
    epochs=75,            # Number of training epochs
    batch_size=32,        # Size of each training batch
    verbose=1,            # Display training progress
    callbacks=[early_stopping]  # Add the EarlyStopping callback
)

model.save(os.path.expanduser("~/ros_ws/src/controller/models/drive_model.h5"))