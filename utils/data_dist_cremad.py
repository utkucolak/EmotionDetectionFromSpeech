import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("cremad_metadata.csv")

# Count emotion frequencies
emotion_counts = df["emotion"].value_counts().sort_index()

# Plot
plt.figure(figsize=(8, 5))
emotion_counts.plot(kind="bar", color="skyblue", edgecolor="black")
plt.title("CREMA-D Emotion Class Distribution")
plt.xlabel("Emotion")
plt.ylabel("Number of Samples")
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()