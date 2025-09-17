import os
import yaml
import sys
from pathlib import Path
from dotenv import load_dotenv
from roboflow import Roboflow
from ultralytics import YOLO


# Directories - кроссплатформенные пути
project_root = Path(__file__).parent.parent
data_dir = project_root / 'datasets'
config_dir = project_root / 'configs' / 'yolo'
config_dir.mkdir(parents=True, exist_ok=True)
data_dir.mkdir(parents=True, exist_ok=True)


def load_config(config_path: str) -> dict:
    """Loads the YAML configuration file and converts space arrays to tuples."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Convert space arrays to tuples for hyperparameter tuning
        if 'tuning_params' in config and 'space' in config['tuning_params']:
            space = config['tuning_params']['space']
            for key, value in space.items():
                if isinstance(value, list) and len(value) == 2:
                    space[key] = tuple(float(v) for v in value)
            
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        sys.exit(1)


def download_dataset(config: dict, api_key: str) -> Path:
    """
    Downloads the dataset from Roboflow if it doesn't exist locally.
    Returns the path to the dataset's data.yaml file.
    """
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(config['workspace']).project(config['project_id'])
    
    # Construct the local path for the dataset
    dataset_name = f"{config['project_id']}-{config['version']}"
    dataset_path = data_dir / dataset_name

    # Проверяем существование датасета и файла data.yaml
    data_yaml_path = dataset_path / 'data.yaml'

    if not data_yaml_path.exists():
        print(f"Dataset not found locally. "
              f"Downloading version {config['version']}...")

        version = project.version(config['version'])
        # Используем параметр location для указания пути загрузки
        dataset = version.download("yolov8", location=str(dataset_path))
        return Path(dataset.location) / 'data.yaml'
    else:
        print("Dataset already exists locally.")
        return data_yaml_path


def main():
    """Main function to run the hyperparameter tuning process."""
    load_dotenv()
    config = load_config(str(config_dir / 'tune_config.yaml'))

    # --- 1. Get Roboflow API Key ---
    api_key = os.getenv('ROBOFLOW_API_KEY')
    if not api_key:
        print("Error: ROBOFLOW_API_KEY not found.")
        print("Please create a .env file in the project root and add your Roboflow API key to it.")
        print("Example: ROBOFLOW_API_KEY='YOUR_API_KEY'")
        sys.exit(1)

    # --- 2. Download Dataset ---
    print("--- Checking for dataset ---")
    roboflow_config = config.get('roboflow', {})
    required_keys = ['workspace', 'project_id', 'version']
    if not all(k in roboflow_config for k in required_keys):
        print("Error: Roboflow configuration is incomplete "
              "in tune_config.yaml")
        sys.exit(1)

    data_yaml_path = download_dataset(roboflow_config, api_key)

    # Update tuning params with the correct data path
    config['tuning_params']['data'] = str(data_yaml_path)
    print(f"Using dataset config: {data_yaml_path}")

    # --- 3. Tune Hyperparameters ---
    print("\n--- Starting YOLOv8 Hyperparameter Tuning ---")
    tuning_params = config.get('tuning_params', {})
    if not tuning_params.get('model'):
        print("Error: 'model' not specified in tuning_params "
              "in tune_config.yaml")
        sys.exit(1)

    try:
        model = YOLO(tuning_params['model'])
        model.tune(**tuning_params)
        print("\n--- Tuning complete! ---")
        # The results are automatically saved in the runs/segment/tune directory
    except Exception as e:
        print(f"An error occurred during tuning: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
