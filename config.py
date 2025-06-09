from pathlib import Path

def get_config():
    return {
        "csv_path": "ravdess_metadata.csv",
        "batch_size": 8,
        "num_epochs": 1000,
        "lr": 1e-4,
        "max_len": 300,
        "d_model": 512,
        "model_folder": "model",
        "model_basename": "tmodel_",
        "preload": None,
        "experiment_name": "runs/tmodel"
    }

def get_weights_file_path(config, epoch: str):
    model_folder = config["model_folder"]
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}{epoch}.pt"

    return str(Path(".") / model_folder / model_filename)