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
        layers.Input(shape=(224, 224, 3)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),

        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),

        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(2, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
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
    epochs=75,
    batch_size=32,
    verbose=1,
    callbacks=[early_stopping]
)

# Save the model
model.save(model_path)

# from tqdm import tqdm
# import pandas as pd
# import cv2
# import os
# import numpy as np
# import gc
# from multiprocessing import Pool
# from tensorflow.keras.models import load_model
# from tensorflow.keras.callbacks import EarlyStopping

# csv_path = os.path.expanduser("~/ros_ws/src/controller/train/drive_data_output/velocity_data.csv")
# images_path = os.path.expanduser("~/ros_ws/src/controller/train/drive_data_output")

# data = pd.read_csv(csv_path, header=None, names=["image", "linear_vel", "angular_vel"])
# data = data.sample(frac=1.0, random_state=42).reset_index(drop=True)  # Shuffle rows

# def process_image(row):
#     img_path = os.path.join(images_path, row["image"])
#     try:
#         image = cv2.imread(img_path)
#         if image is not None:
#             image = cv2.resize(image, (128, 128)) / 255.0
#             return image, [row["linear_vel"], row["angular_vel"]]
#     except Exception as e:
#         print(f"Error processing {img_path}: {e}")
#     return None, None

# # Batch processing
# batch_size = 500
# X = []
# y = []

# for start_idx in range(0, len(data), batch_size):
#     end_idx = min(start_idx + batch_size, len(data))
#     batch = data.iloc[start_idx:end_idx]

#     with Pool(processes=4) as pool:
#         results = list(tqdm(pool.imap(process_image, batch.to_dict(orient="records")), total=len(batch)))

#     # Collect results
#     batch_images, batch_velocities = zip(*[(img, vel) for img, vel in results if img is not None])
#     X.extend(batch_images)
#     y.extend(batch_velocities)

#     # Cleanup
#     del batch_images, batch_velocities
#     gc.collect()

# # Convert to numpy arrays
# X = np.array(X, dtype="float32")
# y = np.array(y, dtype="float32")

# print(f"Shape of images: {X.shape}")
# print(f"Shape of velocities: {y.shape}")

# # Path to the .h5 file
# model_path = os.path.expanduser("~/ros_ws/src/controller/models/drive_model.h5")

# # Load the model
# model = load_model(model_path)

# # Define the EarlyStopping callback
# early_stopping = EarlyStopping(
#     monitor='val_loss',      # Metric to monitor (e.g., validation loss)
#     patience=10,              # Number of epochs with no improvement before stopping
#     verbose=1,               # Print a message when training stops
#     restore_best_weights=True  # Restore model weights from the epoch with the best value
# )

# history = model.fit(
#     X, y,  # Use the entire dataset without a separate validation set
#     validation_split=0.2,  # Use 20% of the data for validation
#     epochs=75,            # Number of training epochs
#     batch_size=32,        # Size of each training batch
#     verbose=1,            # Display training progress
#     callbacks=[early_stopping]  # Add the EarlyStopping callback
# )

# model.save(os.path.expanduser("~/ros_ws/src/controller/models/drive_model.h5"))