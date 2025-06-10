import pandas as pd
import matplotlib.pyplot as plt

EMOTION_MAP = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"
}

# Load the dataset
df = pd.read_csv("train_temp.csv")

# If the 'emotion' column is already string-labeled (like 'happy'), skip this.
# If it's numeric codes, map them:
if df["emotion"].dtype != object:
    df["emotion"] = df["emotion"].astype(str).str.zfill(2)
    df["emotion"] = df["emotion"].map(EMOTION_MAP)

# Count occurrences
emotion_counts = df["emotion"].value_counts().sort_index()

# Plot
plt.figure(figsize=(10, 5))
emotion_counts.plot(kind='bar', color='skyblue')
plt.title("Emotion Distribution in Training Set")
plt.xlabel("Emotion")
plt.ylabel("Number of Samples")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()