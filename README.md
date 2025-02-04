# GRModelMaker

GRModelMaker takes a directory of training data and returns a directory containing a `.task` file trained on the data.

---

## Getting Started

### Requirements

- **Python Version**: This program has only been tested on **Python 3**.
- **Dependencies**: Ensure you have the following libraries installed:
  - **TensorFlow**
  - **MediaPipe Model Maker**

Install them via pip if necessary:

```sh
pip install tensorflow mediapipe-model-maker
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
python3 grt.py –dataset_path hagrid15A
```

Replace `<your_dataset_path>` with the path to your dataset.

---

## Outputs

After running the program, the resulting `.task` file will be available in the specified output directory.
