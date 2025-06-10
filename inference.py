import torch
import sounddevice as sd
import librosa
import numpy as np
import keyboard

from config import get_config, get_weights_file_path
from utils.create_mel_spectogram import create_mel_spectogram_for_inference
from train import get_model

# CREMA-D has 6 emotion classes
EMOTION_MAP = {
    0: "angry",
    1: "disgust",
    2: "fearful",
    3: "happy",
    4: "neutral",
    5: "sad"
}

def record_audio(duration=2, sr=22050):
    print("Recording...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()
    print("Done.")
    return audio.flatten()

def is_silent(waveform, threshold=0.01, min_energy_duration=0.2, sr=22050):
    energy = np.abs(waveform)
    above_thresh = energy > threshold
    return np.sum(above_thresh) < sr * min_energy_duration

def prepare_input(audio, config):
    mel = create_mel_spectogram_for_inference(audio, sr=22050)  # (n_mels, time_steps)
    time_steps = config["max_len"]

    if mel.shape[1] < time_steps:
        pad_width = time_steps - mel.shape[1]
        mel = np.pad(mel, ((0, 0), (0, pad_width)), mode='constant')
    elif mel.shape[1] > time_steps:
        mel = mel[:, :time_steps]

    mel_tensor = torch.tensor(mel).transpose(0, 1).unsqueeze(0).float()  # (1, time_steps, n_mels)
    return mel_tensor

def run_inference():
    config = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(config)
    weights_path = get_weights_file_path(config, "best")
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval().to(device)

    print("Press ESC to stop inference...\n")

    while True:
        if keyboard.is_pressed('esc'):
            print("ESC pressed. Exiting...")
            break

        audio = record_audio()
        if is_silent(audio):
            print("Silence detected. Skipping...\n")
            continue

        mel_tensor = prepare_input(audio, config).to(device)

        with torch.no_grad():
            output = model(mel_tensor)
            pred = torch.argmax(output, dim=1).item()
            probs = torch.softmax(output, dim=1).cpu().numpy()[0]

            for i, p in enumerate(probs):
                print(f"{EMOTION_MAP[i]}: {p:.4f}")
            print(f"Predicted Emotion: {EMOTION_MAP[pred]}\n")

if __name__ == "__main__":
    run_inference()