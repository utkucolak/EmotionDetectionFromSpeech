# Emotion Classification from Speech Using Transformers

This project implements an end-to-end Transformer-based neural network to classify emotions from raw audio speech data. It uses the RAVDESS dataset and extracts mel spectrograms as input features for a custom-built Transformer encoder architecture.

---

## 📚 Description

The model is built entirely from scratch, mimicking the encoder structure from the Transformer paper: _"Attention Is All You Need" (Vaswani et al.)_. It includes positional encoding, multi-head attention, residual connections, layer normalization, and feedforward layers. It performs emotion classification without any decoder or NLP tokenizer components.

---

## 🧠 Key Features

- Custom Transformer Encoder (no HuggingFace or pretrained models)
- Manual positional encoding implementation
- Multi-head attention with residuals and layer norm
- Audio input preprocessed to mel spectrograms
- Emotion classification with final linear classifier
- Dataset: RAVDESS (speech-only subset)

---

## 📁 Project Structure

```
EmotionDetectionFromSpeech/
├── data/                    # Raw and preprocessed dataset (excluded from Git)
├── utils/                   # Helper scripts (mel spectrogram, constants)
├── train.py                 # Training loop
├── transformer.py           # Full Transformer architecture
├── ravdess_metadata.csv     # Parsed metadata (filepaths + emotion labels)
├── download_dataset.py      # Script to fetch and extract dataset
├── README.md                # This file
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/utkucolak/EmotionDetectionFromSpeech.git
cd EmotionDetectionFromSpeech
```

### 2. Install Requirements
```bash
pip install -r requirements.txt
```
Or manually:
```bash
pip install torch torchaudio librosa pandas scikit-learn
```

---

## 🎧 Preprocess Audio

Download and extract the RAVDESS dataset:
```bash
python download_dataset.py
```

Generate metadata:
```bash
# You must run your mel spectrogram and metadata builder script here
```

---

## 🏋️‍♀️ Train the Model

Run the training loop:
```bash
python train.py
```

Make sure your model is built using:
```python
from transformer import build_transformer
model = build_transformer(time_steps=300, d_model=128)
```

---

## 📈 Evaluation

Evaluation is currently based on validation accuracy. Future work will include:
- Confusion matrix
- Precision/Recall/F1 metrics

---

## 🔖 Emotion Classes

| ID | Emotion   |
|----|-----------|
| 01 | Neutral   |
| 02 | Calm      |
| 03 | Happy     |
| 04 | Sad       |
| 05 | Angry     |
| 06 | Fearful   |
| 07 | Disgust   |
| 08 | Surprised |

---

## 🧩 TODO

- [x] Build custom Transformer encoder
- [x] Implement positional encoding and attention manually
- [x] Train model on RAVDESS dataset
- [ ] Evaluate with confusion matrix
- [ ] Add live audio prediction support
- [ ] Export model and demo notebook

---

## 👤 Author

Utku Çolak – [@utkucolak](https://github.com/utkucolak)
