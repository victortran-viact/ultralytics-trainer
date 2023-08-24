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



def test_yolo(
    dataset_id: str,
    model_id: str,
) -> None: 
    """
    Sample output: ultralytics.utils.metrics.DetMetrics object with attributes:
    ```
        ap_class_index: array([0])
        box: ultralytics.utils.metrics.Metric object
        confusion_matrix: <ultralytics.utils.metrics.ConfusionMatrix object at 0x7fd3a588fac0>
        fitness: 0.12331301833487544
        keys: ['metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)']
        maps: array([    0.11055,     0.11055])
        names: {0: 'canvas', 1: 'person'}
        plot: True
        results_dict: {'metrics/precision(B)': 0.05601317957166392, 'metrics/recall(B)': 1.0, 'metrics/mAP50(B)': 0.23813998138422993, 'metrics/mAP50-95(B)': 0.11055446688494716, 'fitness': 0.12331301833487544}
        save_dir: PosixPath('runs/detect/val2')
        speed: {'preprocess': 0.11606216430664062, 'inference': 8.380484580993652, 'loss': 0.00152587890625, 'postprocess': 1.4193296432495117}
    ```
    """
    yaml_filepath = get_dataset_from_storage(dataset_id=dataset_id)

    print(f"Dataset is stored at {yaml_filepath}")
    print("Complete prepared dataset, continue to training the model...")

    model_path = download_model(model_id)
    model = YOLO(model_path)
    
    metrics = model.val(
        data=yaml_filepath,
    )

 
    return metrics 



if __name__ == "__main__":
    import argparse
    args = argparse.ArgumentParser()
    args.add_argument(
        "--dataset_id", default="yolov5s", help="ClearML dataset id"
    )
    args.add_argument(
        "--model_id", help="Model from ClearML Registry", type=str, default=None
    )

    args = args.parse_args()
    
    test_yolo(
        dataset_id=args.dataset_id,
        model_id=args.model_id,
    )