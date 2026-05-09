# PRISM

PRISM: Physics-Guided Multimodal Vehicle Trajectory Prediction with Driving Style Awareness

---

## Environment Setup

- Python 3.11
- Install required dependencies:

```bash
pip install -r requirements.txt
```

---

## Directory Structure

```python
PRISM
├─data
│  └─tool
├─models
├─modul
├─project
└─trajectory_compared_test
    └─experiment
```

---

## Dataset

The original highD dataset can be obtained from the official website:

- [highD Dataset Official Website](https://levelxdata.com/highd-dataset/?utm_source=chatgpt.com)

A total of 60 vehicle trajectory files are required for this project.

---

## Feature Extraction

1. Extract the required features from the original dataset:

```bash
python extract_feature_data.py
```

2. Perform data cleaning and preprocessing:

```bash
python handler_data.py
```

The processed data generated after these steps serves as the final feature dataset required for model training.

---

## Training

Configure the relevant parameters and run the following script to start training:

```bash
python train.py
```

The folders `trajectory_compared_test` and `trajectory_compared_val` contain the generated trajectory comparison results for the test and validation sets, respectively.

---

## Acknowledgements

- highD-Dataset
- HighD-NGSIM-lane-change-feature-extraction