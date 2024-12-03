import argparse
from tqdm import tqdm
import pandas as pd
import cv2
import os
import numpy as np
import gc
from multiprocessing import Pool
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

# Define the function to create a new model
def create_new_model():
    model = Sequential([
        layers.Input(shape=(168, 224, 3)),

        # First block
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        # Second block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        # Third block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        # Fourth block
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        # Global Average Pooling
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.4),

        # Fully Connected Layers
        layers.Dense(128, activation='relu'),  # Increased size
        layers.Dense(64, activation='relu'),
        layers.Dense(2)  # Output layer
    ])

    model.compile(optimizer='adam', loss='mse')
    model.summary()
    return model

# Command-line argument parser
parser = argparse.ArgumentParser(description="Drive Controller Training Script")
parser.add_argument(
    "--new-model",
    action="store_true",
    help="Redefine the model instead of loading the existing one."
)
args = parser.parse_args()

# Paths
csv_path = os.path.expanduser("~/ros_ws/src/controller/train/drive_data_output/velocity_data.csv")
images_path = os.path.expanduser("~/ros_ws/src/controller/train/drive_data_output")
model_path = os.path.expanduser("~/ros_ws/src/controller/models/drive_model.h5")

# Load and shuffle dataset
data = pd.read_csv(csv_path, header=None, names=["image", "linear_vel", "angular_vel"])
data = data.sample(frac=1.0).reset_index(drop=True)

def process_image(row):
    img_path = os.path.join(images_path, row["image"])
    try:
        image = cv2.imread(img_path)
        if image is not None:
            image = cv2.resize(image, (224, 224)) / 255.0  
            
            image = image[56 :, :, :]  # Keep only the bottom 3/4
            
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

reduce_size = 3

X = X[: X.shape[0] // reduce_size, :, :, :]
y = y[: y.shape[0] // reduce_size, :]

print(f"Shape of images: {X.shape}")
print(f"Shape of velocities: {y.shape}")

# Define the model (new or existing)
if args.new_model:
    print("Defining a new model...")
    model = create_new_model()
else:
    print("Loading existing model...")
    model = load_model(model_path)

# Define the EarlyStopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    verbose=1,
    restore_best_weights=True
)

# Train the model
history = model.fit(
    X, y,
    validation_split=0.2,
    epochs=150,
    batch_size=16,
    verbose=1,
    callbacks=[early_stopping]
)

# Save the model
model.save(model_path)
