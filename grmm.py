import os
import tensorflow as tf
from mediapipe_model_maker import gesture_recognizer
from time import time
import argparse

# check tensorflow version
assert tf.__version__.startswith('2'), "TensorFlow version 2.x is required."

# for the argument
parser = argparse.ArgumentParser(description="Train a model like the MediaPipe Hand Gesture Classification model.")
parser.add_argument(
    "--dataset_path",
    type=str,
    required=True,
    help="Path to the dataset directory containing gesture classes."
)
args = parser.parse_args()

# get dataset path from argument
dataset_path = args.dataset_path

# show user gesture labels
print(f"Dataset path: {dataset_path}")
labels = [label for label in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, label))]
print(f"Labels: {labels}")

# for timer
overall_start_time = time()

# process dataset
print("Loading dataset...")
data = gesture_recognizer.Dataset.from_folder(
    dirname=dataset_path,
    hparams=gesture_recognizer.HandDataPreprocessingParams()
)
train_data, rest_data = data.split(0.8)
validation_data, test_data = rest_data.split(0.5)
print("Dataset loaded.")

# training
print("Training the model...")
hparams = gesture_recognizer.HParams(export_dir="exported_model")
options = gesture_recognizer.GestureRecognizerOptions(hparams=hparams)
model = gesture_recognizer.GestureRecognizer.create(
    train_data=train_data,
    validation_data=validation_data,
    options=options
)
print("Model trained.")

# evaluation
print("Evaluating the model...")
loss, acc = model.evaluate(test_data, batch_size=1)
print(f"Test loss: {loss}, Test accuracy: {acc}")

# export model
print("Exporting the model...")
model.export_model()
print("Model exported to the 'exported_model' directory.")


overall_elapsed_time = time() - overall_start_time
print(f"Total time elapsed: {overall_elapsed_time:.2f} seconds.")