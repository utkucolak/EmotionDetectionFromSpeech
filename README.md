
# Emotion Detection from Speech – Transformer-Based Model

This project implements a transformer-based model for emotion recognition from speech signals, built entirely from scratch without relying on pre-trained libraries like HuggingFace.

## 🔍 Overview

The system classifies emotional states from speech using mel-spectrogram features and a transformer encoder architecture. It has been tested with two datasets:
- [RAVDESS](https://zenodo.org/record/1188976)
- [CREMA-D](https://www.kaggle.com/datasets/ejlok1/cremad)

## 📁 Project Structure

```
├── data/
│   ├── AudioWAV/                 # Raw CREMA-D audio files
│   └── ravdess/                  # Raw RAVDESS audio files
│
├── datasetmetadata/
│   └── ravdess_metadata.csv      # Metadata for RAVDESS
│
├── figures/
│   ├── training_curves_cremad.png
│   ├── training_curves_ravdess.png
│   ├── confusion_matrix.png
│   ├── data_dist_cremad.png
│   ├── multi_head_attention.png
│   └── transformer_from_scratch.png
│   └── example_output.png
│   └── model_summary.png
│
├── model/ #contains model weights
│
├── utils/
│   ├── constants.py
│   ├── create_mel_spectrogram.py
│   ├── dataset.py                # Dataset loaders for RAVDESS & CREMA-D
│   ├── data_dist_cremad.py      # Class distribution visualization
│   ├── generate_data_files_for_input.py
│   ├── generate_cremad_metadata.py
│   └── visualize.py              # Training & evaluation visualizations
│
├── train.py                      # Main training script
├── inference.py                  # Real-time microphone inference
│── model.py                  # Transformer model components
├── download_dataset.py
├── config.py                     # Global config
├── val_predictions.csv           # CSV containing predicted validation results
├── LICENSE
├── README.md
└── requirements.txt
```

## 🧠 Model Architecture

- Transformer encoder (built from scratch)
- Multi-head self-attention
- Positional encoding
- Residual connections and layer normalization
- Classification head (linear + softmax)

## 🧪 Training

- Optimizer: Adam
- Loss: CrossEntropy with label smoothing
- Dropout for regularization
- On-the-fly data augmentation for audio inputs (e.g., pitch/time shifting)

Training and validation performance are visualized in `figures/`.

## 📊 Evaluation

Confusion matrix and accuracy/loss curves can be found in:
- `figures/confusion_matrix.png`
- `figures/training_curves_cremad.png`
- `figures/data_dist_cremad.png`

## 🎙️ Real-Time Inference

Run the following to test with live microphone input:

```bash
python inference.py
```

> Press `ESC` to stop recording loop.

## 🧩 Supported Emotions

For CREMA-D:  
`[angry, disgust, fearful, happy, neutral, sad]`

## ✅ Status

- ✅ Custom transformer encoder
- ✅ Dataset parsing & preprocessing
- ✅ On-the-fly audio augmentation
- ✅ Real-time microphone inference
- ✅ CREMA-D support (speaker-independent split)
- ✅ Visualization of metrics

## 📌 Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

## ✍️ Author

Mehmet Utku Çolak
