from create_mel_spectogram import *
from constants import *
import pandas as pd
import os

DATA = "RAVDESS"
data_path = os.path.join("data", DATA.lower())

data = []

for actor_folder in os.listdir(data_path):
    voice_path = os.path.join(data_path, actor_folder)
    for voice_file_path in os.listdir(voice_path):
        mel_spec = create_mel_spectogram(os.path.join(voice_path, voice_file_path))
        print("mel_spec shape: ", mel_spec.shape)
        voice_file_path = voice_file_path.split(".")[0] # we shall remove .wav
        data_parts = voice_file_path.split("-")
        emotion_id = data_parts[2]
        emotion = EMOTION_MAP[emotion_id]
        actor_id = int(data_parts[-1])
        gender = GENDER[str(actor_id % 2)]
        modality = MODALITY[data_parts[0]]
        vocal_channel = VOCAL_CHANNEL[data_parts[1]]
        intensity = EMOTIONAL_INTENSITY[data_parts[3]]

        data.append(
            {
                "filepath" : os.path.join(voice_path, voice_file_path+".wav"),
                "emotion" : emotion,
                "modality" : modality,
                "vocal_channel" : vocal_channel,
                "emotional_intensity" : intensity,
                "actor_id" : str(actor_id),
                "gender" : gender
            }
        )

df = pd.DataFrame(data)
df.to_csv("ravdess_metadata.csv", index=False)
        

