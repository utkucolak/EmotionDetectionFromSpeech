from pathlib import Path

def get_config():
    dataset_type = "cremad"  # Change to "ravdess" or "cremad"

    if dataset_type == "ravdess":
        csv_path = "ravdess_metadata.csv"
        num_classes = 8
    elif dataset_type == "cremad":
        csv_path = "cremad_metadata.csv"
        num_classes = 6
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

    return {
        "dataset_type": dataset_type,
        "csv_path": csv_path,
        "num_classes": num_classes,
        "batch_size": 16,
        "num_epochs": 50,
        "lr": 1e-4,
        "max_len": 300,
        "d_model": 64,
        "model_folder": "model",
        "model_basename": "tmodel_",
        "preload": None,
        "experiment_name": "runs/tmodel",
        "n_mels": 64,
        "h": 2,
        "d_ff": 128,
        "N": 2
    }

def get_weights_file_path(config, epoch: str):
    model_folder = config["model_folder"]
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}{epoch}.pt"

    return str(Path(".") / model_folder / model_filename)