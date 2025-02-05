# GRModelMaker

GRModelMaker processes a directory of training data and generates a directory containing a `.task` file trained on the data. These `.task` files are designed for use with [Google's MediaPipe Hand Gesture Classification model](https://ai.google.dev/edge/mediapipe/solutions/vision/gesture_recognizer). This program is an offline implementation of the [Google Colab Notebook](https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/customization/gesture_recognizer.ipynb) that performs a similar task. It provides the same functionality while allowing users to train a gesture recognition model entirely on their local machine.

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
python3 -W ignore grmm.py -–dataset_path <your_dataset_path>
```

### Example:

```sh
python3 -W ignore grmm.py --dataset_path sampledata
```

Replace `<your_dataset_path>` with the path to your dataset.

### Outputs

After running the program, the resulting `.task` file will be available in a directory named 'exported_model'.

--- 

## Code Structure

    .
    ├── sampledata/       # Sample data, ready-to-go
    ├── exported_model/   # Script output (excluded from Git)
    ├── venv/             # Virtual environment (excluded from Git)
    ├── .gitignore        # Files ignored by Git
    ├── README.md         # Project documentation
    ├── grmm.py           # Script to train the model 
    └── requirements.txt  # Dependency list

--- 

## Additional Notes
* I recommend using the [HaGRIDv2 (HAnd Gesture Recognition Image Dataset)](https://github.com/hukenovs/hagrid) for hand gesture classification training data. The dataset is huge and features a variety of distinct gestures. If you are running this program locally or are low on space, I would recommend using subsets of HaGRIDv2. I personally can vouch for [hagrid_subsets](https://huggingface.co/datasets/GestureDetectionConnoisseurs/hagrid_subsets) on Hugging Face.
* For reference: on an M1 Macbook Air, the program takes around 25 seconds using the sample data. On a 1.4GB dataset, it took around 9 minutes.
* If you want to customize the model and training process, refer to [Google's official documentation] for a detailed explanation.
