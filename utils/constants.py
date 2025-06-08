## Dataset mappings for the RAVDESS dataset. 
## its ordered as modality - vocal channel - emotion - emotional intensity - statement - repetition - actor

MODALITY = {
    "01" : "full-AV",
    "02" : "video-only",
    "03" : "audio-only"
}

VOCAL_CHANNEL = {
    "01" : "speech",
    "02" : "song"
}

EMOTIONAL_INTENSITY = {
    "01" : "normal",
    "02" : "strong"
}

EMOTION_MAP = {
    "01" : "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05" : "angry", "06": "fearful", "07": "disgust", "08": "surprised"
}

EMOTION_TO_IDX = {v: int(k) - 1 for k, v in EMOTION_MAP.items()}

STATEMENT = {
    "01" : "Kids are talking by the door",
    "02" : "Dogs are sitting by the door"
}

REPETITION = {
    "01" : "1st repetition",
    "02" : "2nd repetition"
}

GENDER = {
    "0" : "female",
    "1" : "male"
}