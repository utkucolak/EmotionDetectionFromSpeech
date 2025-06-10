from utils.visualize import *
import pandas as pd

csv_path = "val_predictions.csv"
df = pd.read_csv(csv_path)

true = df["true"]
pred = df['pred']

plot_confusion_matrix(true, pred)