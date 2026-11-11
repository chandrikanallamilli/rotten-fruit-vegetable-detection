# Transfer Learning for Identifying Rotten Fruits and Vegetables
Full-stack web app that classifies an uploaded photo as **fresh** or **rotten** using transfer learning (MobileNetV2).

**Stack:** Python, TensorFlow/Keras, Flask, SQLite (database), HTML, CSS, JavaScript

## Setup
```bash
pip install -r requirements.txt
```
1. Put images in `dataset/train/{fresh,rotten}` and `dataset/val/{fresh,rotten}`
   (e.g. the "Fruits fresh and rotten for classification" dataset on Kaggle).
2. Train: `python train.py`  -> saves `model/fruit_model.keras`
3. Run the web app: `python app.py` -> http://localhost:5000

## How it works
- `train.py`: MobileNetV2 (ImageNet weights, frozen) + new classification head; data augmentation; fine-tuned on the dataset.
- `app.py`: Flask upload endpoint, prediction, and every prediction is logged to SQLite (`predictions.db`) and shown in the history table.
