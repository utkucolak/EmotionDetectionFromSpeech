import os
import pandas as pd
from pathlib import Path

# Path to CREMA-D WAV files
AUDIO_DIR = Path("data/AudioWAV")

# Emotion code to label mapping
EMOTION_MAP = {
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fearful",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad"
}

def parse_filename(filename):
    """
    Parses a filename like 1001_IEO_DIS_MD.wav
    Returns actor_id, emotion_label
    """
    parts = filename.stem.split("_")  # remove .wav and split
    if len(parts) != 4:
        return None, None
    actor_id = parts[0]
    emotion_code = parts[2]
    emotion_label = EMOTION_MAP.get(emotion_code)
    return actor_id, emotion_label

def generate_metadata(audio_dir):
    data = []
    for wav_file in audio_dir.glob("*.wav"):
        actor_id, emotion = parse_filename(wav_file)
        if emotion:  # skip unknown emotion codes
            data.append({
                "filepath": str(wav_file),
                "actor_id": actor_id,
                "emotion": emotion
            })

    return pd.DataFrame(data)

if __name__ == "__main__":
    df = generate_metadata(AUDIO_DIR)
    df.to_csv("cremad_metadata.csv", index=False)
    print(f"Saved metadata with {len(df)} samples to cremda_metadata.csv")