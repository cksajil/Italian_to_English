import os
import yaml
import requests
from tqdm import tqdm


def load_config(config_name):
    """
    A function to load and return config file in YAML format
    """
    CONFIG_PATH = "./config/"
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)

    return config


config = load_config("config.yaml")


def create_models_folder():
    """Function to create a folder in a location if it does not exist"""
    if not os.path.exists("models"):
        os.makedirs("models")


def download_model():
    """
    Download pretrained models if not already downloaded
    """
    if not os.path.exists(os.path.join(config["model_loc"], config["model_name"])):
        print("Downloading pretrained Attention dot model if not exists")
        doi = config["model_doi"]
        response = requests.get(f"https://zenodo.org/api/records/{doi}")
        data = response.json()
        files = data["files"]
        formats = (".hdf5", ".pickle")
        model_files = [file for file in files if file["key"].endswith(formats)]

        if len(model_files) == 0:
            print("No model files found.")
        else:
            model_file = model_files[0]
            file_url = model_file["links"]["self"]
            model_filename = model_file["key"]

            response = requests.get(file_url, stream=True)
            segments = response.iter_content()
            with open(os.path.join(config["model_loc"], model_filename), "wb") as file:
                for chunk in tqdm(segments):
                    file.write(chunk)

            print(f"Model downloaded as {model_filename}")
