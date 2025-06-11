from utils.visualize import *
import pandas as pd
from torchinfo import summary
from train import get_model
from config import get_config

config = get_config()

csv_path = "val_predictions.csv"
df = pd.read_csv(csv_path)

true = df["true"]
pred = df['pred']

#plot_confusion_matrix(true, pred)

print(summary(get_model(config)))