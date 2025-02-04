import os
import tensorflow as tf
from mediapipe_model_maker import gesture_recognizer
from time import time  # Import time for elapsed time calculation
import warnings
import argparse  # Import argparse for command-line arguments

# Suppress specific warnings and TensorFlow logs
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Check TensorFlow version
assert tf.__version__.startswith('2'), "TensorFlow version 2.x is required."

# Set up argument parser
parser = argparse.ArgumentParser(description="Train a gesture recognition model.")
parser.add_argument(
    "--dataset_path",
    type=str,
    required=True,
    help="Path to the dataset directory containing gesture classes."
)
args = parser.parse_args()

# Get dataset path from arguments
dataset_path = args.dataset_path

# Verify the dataset by printing the labels
print(f"Dataset path: {dataset_path}")
labels = [label for label in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, label))]
print(f"Labels: {labels}")

# Start the overall timer
overall_start_time = time()

# Load the dataset
print("Loading dataset...")
data = gesture_recognizer.Dataset.from_folder(
    dirname=dataset_path,
    hparams=gesture_recognizer.HandDataPreprocessingParams()
)
print("Dataset loaded.")

# Split dataset
print("Splitting dataset...")
train_data, rest_data = data.split(0.8)
validation_data, test_data = rest_data.split(0.5)
print("Dataset split.")

# Train the model
print("Training the model...")
hparams = gesture_recognizer.HParams(export_dir="exported_model")
options = gesture_recognizer.GestureRecognizerOptions(hparams=hparams)
model = gesture_recognizer.GestureRecognizer.create(
    train_data=train_data,
    validation_data=validation_data,
    options=options
)
print("Model trained.")

# Evaluate the model
print("Evaluating the model...")
loss, acc = model.evaluate(test_data, batch_size=1)
print(f"Test loss: {loss}, Test accuracy: {acc}")

# Export the model
print("Exporting the model...")
model.export_model()
print("Model exported to the 'exported_model' directory.")

# Stop the overall timer
overall_elapsed_time = time() - overall_start_time
print(f"Total time elapsed: {overall_elapsed_time:.2f} seconds.")