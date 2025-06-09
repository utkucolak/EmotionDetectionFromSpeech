import torch
import sounddevice as sd
import librosa
import numpy as np
from model import build_transformer
from config import get_config, get_weights_file_path
from utils.create_mel_spectogram import create_mel_spectogram_for_inference
import keyboard
from train import get_model
EMOTION_MAP = {
    0: "neutral", 1: "calm", 2: "happy", 3: "sad",
    4: "angry", 5: "fearful", 6: "disgust", 7: "surprised"
}

def record_audio(duration=2, sr=22050):
    print("Recording...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()
    print("Done.")
    return audio.flatten()

def prepare_input(audio, config):
    mel = create_mel_spectogram_for_inference(audio, sr=22050)  # (n_mels, time_steps)

    time_steps = config["max_len"]

    # Pad or truncate the time axis (axis=1)
    if mel.shape[1] < time_steps:
        pad_width = time_steps - mel.shape[1]
        mel = np.pad(mel, ((0, 0), (0, pad_width)), mode='constant')  # pad along time dimension
    elif mel.shape[1] > time_steps:
        mel = mel[:, :time_steps]  # trim time dimension

    mel_tensor = torch.tensor(mel).transpose(0, 1).unsqueeze(0).float()  # (1, time_steps, n_mels)
    return mel_tensor

def run_inference():
    config = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(config, config["max_len"])
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
        mel_tensor = prepare_input(audio, config).to(device)

        with torch.no_grad():
            output = model(mel_tensor)
            pred = torch.argmax(output, dim=1).item()
            print(f"Predicted Emotion: {EMOTION_MAP[pred]}")

if __name__ == "__main__":
    run_inference()