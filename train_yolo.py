from clearml import Dataset
from ultralytics import YOLO


def get_model_name_from_choice(model_name: str, model_variant: str) -> str:
    mapping = {
        ("YOLOv5", "small"): "yolov5s",
        ("YOLOv5", "medium"): "yolov5m",
        ("YOLOv5", "large"): "yolov5l",
        ("YOLOv8", "small"): "yolov8s",
        ("YOLOv8", "medium"): "yolov8m",
        ("YOLOv8", "large"): "yolov8l",
    }

    return mapping.get((model_name, model_variant), "")


def get_dataset_from_storage(dataset_id: str) -> str:
    """
        Extract dataset from ClearML storage & return yaml filepath
        ```
        yolov5/
            temp/
                {dataset_name}.zip     
            datasets/ <- NOTE: this is requried in ultraalytics config 
                {dataset_name}/    
                    train/
                    test/
                    val/
                    *.yaml -> return filepath of *.yaml
        ```
    """
    import os
    import zipfile
    from pathlib import Path

    def _unzip_file(file_path: str, dest_dir: str) -> Path:
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(dest_dir)
        # Get the folder name inside the zip file
        folder_name = os.path.splitext(os.path.basename(file_path))[0]
        return Path(folder_name)

    def _extract_dataset(dataset: Dataset) -> Path:
        # Create temp folder
        temp_dir = os.path.join(os.getcwd(), "temp")
        # NOTE: follow YOLO config
        dataset_dir = os.path.join(os.getcwd(), "datasets")
        os.makedirs(temp_dir, exist_ok=True)
        os.makedirs(dataset_dir, exist_ok=True)

        # Download from storage, will raise exception it if exists the folder
        try:
            dataset.get_mutable_local_copy(target_folder=temp_dir)
        except ValueError as e:
            pass

        # Get the zip filename
        files = dataset.list_files()
        zip_file = Path(temp_dir) / files[0]

        # Unzip
        folder_name = _unzip_file(zip_file, dataset_dir)
        return Path(dataset_dir) / Path(folder_name)

    def _get_yaml_files(folder_path: Path) -> list[Path]:
        yaml_files = folder_path.glob("*.yaml")
        return list(yaml_files)

    dataset = Dataset.get(dataset_id=dataset_id)
    dataset_dir = _extract_dataset(dataset)

    # Assumpe there only 1 *.yaml file
    yaml_filepath: Path = _get_yaml_files(dataset_dir)[0]
    yaml_filepath: str = str(yaml_filepath.absolute())

    return yaml_filepath


def train_yolo(
        dataset_id: str,
        model_version: str = "yolov5s",
        batch_size: int = 16,
        imgsz: int = 640,
        epochs: int = 10,
) -> None:

    yaml_filepath = get_dataset_from_storage(dataset_id=dataset_id)

    print(f"Dataset is stored at {yaml_filepath}")
    print("Complete prepared dataset, continue to training the model...")

    model = YOLO(f"{model_version}.pt")
    model.train(
        data=yaml_filepath,
        imgsz=imgsz,
        epochs=epochs,
        cache='ram',
        batch=batch_size
    )


if __name__ == "__main__":
    import argparse
    args = argparse.ArgumentParser()
    args.add_argument(
        "--dataset_id", default="yolov5s", help="ClearML dataset id"
    )
    args.add_argument(
        "--model_version", default="yolov5s", help="Model version"
    )
    args.add_argument(
        "--batch_size", default=16, help="Batch size"
    )
    args.add_argument(
        "--imgsz", default=640, help="Image size"
    )
    args.add_argument(
        "--epochs", default=10, help="Epochs"
    )
    args = args.parse_args()
    train_yolo(
        dataset_id=args.dataset_id,
        model_version=args.model_version,
        batch_size=args.batch_size,
        imgsz=args.imgsz,
        epochs=args.epochs)
