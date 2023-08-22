import os 
from ultralytics import YOLO 
from clearml import Dataset

from utils.clearml_utils import download_model
   

def get_dataset_from_storage(dataset_id: str) -> str:
    """
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
    from pathlib import Path

    def _get_yaml_files(folder_path: str) -> list[Path]:
        folder_path = Path(folder_path)
        yaml_files = folder_path.glob("*.yaml")
        return list(yaml_files)

    dataset = Dataset.get(dataset_id=dataset_id)

    dataset_dir = os.path.join(os.getcwd(), "datasets")
    folderpath = os.path.join(dataset_dir,dataset.name) 

    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(folderpath, exist_ok=True)

    dataset.get_mutable_local_copy(
        target_folder=folderpath,
        overwrite=True
    )

    # Assumpe there only 1 *.yaml file
    yaml_filepath: Path = _get_yaml_files(folderpath)[0]
    yaml_filepath: str = str(yaml_filepath.absolute())

    return yaml_filepath



def finetune_yolo(
        dataset_id: str,
        model_id: str,
        batch_size: int = 16,
        imgsz: int = 640,
        epochs: int = 10,
) -> None:

    # yaml_filepath = get_dataset_zip_from_storage(dataset_id=dataset_id)
    yaml_filepath = get_dataset_from_storage(dataset_id=dataset_id)

    print(f"Dataset is stored at {yaml_filepath}")
    print("Complete prepared dataset, continue to training the model...")

    model_path = download_model(args.model_id)
    model = YOLO(model_path)
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
        "--model_id", help="Model from ClearML Registry", type=str, default=None
    )

    args.add_argument(
        "--batch_size", default=16, help="Batch size"
    )
    args.add_argument(
        "--imgsz", default=640, help="Image size"
    )
    args.add_argument(
        "--epochs", default=10, help="Epochs", type=int
    )
    args = args.parse_args()
    
    finetune_yolo(
        dataset_id=args.dataset_id,
        model_id=args.model_id,
        batch_size=args.batch_size,
        imgsz=args.imgsz,
        epochs=args.epochs)