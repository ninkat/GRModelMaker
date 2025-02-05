# GRModelMaker

GRModelMaker processes a directory of training data and generates a directory containing a `.task` file trained on the data. These `.task` files are designed for use with [Google's MediaPipe Hand Gesture Classification model](https://ai.google.dev/edge/mediapipe/solutions/vision/gesture_recognizer).

---

## Getting Started

### 🔹 Virtual Environment Setup
It is recommended to use a **virtual environment** to manage dependencies.
You can make a virtual environment with the following:
```sh
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies from requirements.txt
Once the virtual environment is activated, install dependencies:
```sh
pip install -r requirements.txt
```
Remember you can deactivate/leave the virtual environment with the following:
```sh
deactivate
```
---

## Data Specifications

The dataset for gesture recognition must follow the structure specified by Google:

```sh
<dataset_path>/<label_name>/<img_name>.*
```

### Additional Notes:

- **Label Requirements**: One of the labels in your dataset must be `none`. This label represents any gesture that doesn't fall under the other defined labels.

---

## Running the Program

To execute the program, use the following command:

```sh
python3 grt.py –dataset_path <your_dataset_path>
```

### Example:

```sh
python3 grt.py –dataset_path sample
```

Replace `<your_dataset_path>` with the path to your dataset.

### Outputs

After running the program, the resulting `.task` file will be available in a directory named 'exported_model'.

--- 

## Code Structure
The code is organized as such:
GRModelMaker/
│── .gitignore             # Files ignored by Git
│── grt.py                 # Main script to train the model
│── README.md              # Project documentation
│── requirements.txt       # Dependency list
│── sampledata/            # Sample data, ready-to-go
│── venv/                  # Virtual environment (excluded from Git)

