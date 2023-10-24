import argparse
import json
import tempfile
import zipfile
from pathlib import Path

from clearml import Task

from convert_to_onnx import download_model, export_to_onnx
from utils.security import encrypt_model


def label_list_to_txt(label_list: list[str], labels_txt_path: Path) -> None:
    labels_txt = "\n".join(label_list)
    with open(labels_txt_path, "w") as f:
        f.write(labels_txt)


def default_engine_config(
    config_path: Path,
    imgsz=[640, 640],
    model_arch="yolov5s",
    classes=[0]
) -> None:
    config = {
        "model": {
            "arch": model_arch,
            "train": {},
            "inference": {
                "imgsz": imgsz,
                "conf_thres": 0.45,
                "iou_thres": 0.45,
                "max_det": 1000,
                "device": "0",
                "classes": classes,
                "inference_bs": 1,
                "agnostic_nms": False,
                "augment": False,
                "visualize": False,
                "line_thickness": 3,
                "hide_labels": False,
                "hide_conf": False,
                "half": False,
                "dnn": False,
            },
        },
        "alerts": {"alert_string": "ALERT: Detect object inside Danger-zone"},
        "exp": {"exp": "./exp"},
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)


def package_ops(
    model_arch: str,
    version: str,
    label_list:
    list[str],
    model_path: Path,
) -> tuple[str, str]:
    '''
        Return: zip_filepath, zip_filepath
            full_path, filename
    '''
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)

        output_dir = temp_dir_path / f"{model_arch}_{version}"

        configs_dir = output_dir / "configs"
        configs_dir.mkdir(parents=True, exist_ok=True)
        weights_dir = output_dir / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        labels_txt_path = output_dir / "labels.txt"
        config_path = configs_dir / "default_config.json"

        label_list_to_txt(label_list, labels_txt_path)
        default_engine_config(
            config_path=config_path,
            model_arch=model_arch,
        )
        onnx_model_filepath = model_path,

        zip_filepath = Path(f"{model_arch}_{version}.zip")
        with zipfile.ZipFile(zip_filepath, "w") as zipf:
            zipf.write(labels_txt_path, arcname="labels.txt")
            zipf.write(
                config_path, arcname=f"configs/default_config.json")
            zipf.write(onnx_model_filepath,
                       arcname=f"weights/best.onnx")

        return str(zip_filepath.absolute()), str(zip_filepath)


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()

    args_parser.add_argument(
        "--model_id", help="Model from ClearML Registry", type=str, default=None)
    args_parser.add_argument(
        "--model_path", help="Model from local storage", default=None, type=str)
    args_parser.add_argument(
        "--model_arch",
        help="Model type",
        default=None,
        type=str,
        choices=[
            "yolov5s",
            "yolov5m",
            "yolov5l",
            "yolov8s",
            "yolov8m",
            "yolov8l",
        ],
    )
    args_parser.add_argument(
        "--label_list", help="List of labels", nargs="+", default=None, type=str)
    args_parser.add_argument(
        "--version", help="Model version", default="1.0.0", type=str)
    args_parser.add_argument(
        "--encrypt",
        help="Encrypt model",
        action="store_true",
    )
    args = args_parser.parse_args()

    if not args.model_id and not args.model_path:
        raise ValueError("`model_id` or `model_path` must be provided")

    model_path = args.model_path or download_model(args.model_id)

    output_path = export_to_onnx(model_path=model_path)

    print(f"ONNX model stored at: {output_path}")

    print(f"ONNX model stored at: {output_path}")

    if args.encrypt:
        # Encrypt model
        encrypt_model(
            input_path=output_path,
            output_path=output_path,
        )
        print(f"Encrypted model stored at: {output_path}")

    zip_filepath, name = package_ops(
        model_arch=args.model_arch,
        version=args.version,
        label_list=args.label_list,
        output_path=output_path,
    )
    print(f"Package stored at: {name} {zip_filepath}")

    # to upload with clearML
    task = Task.current_task()
    if task:
        print(f"Found current task {task.id}")
        print(f"Uploading to clearML server")

        task.upload_artifact(
            name="exported_model",
            artifact_object=zip_filepath,
        )
        print("Complete upload package to clearML server")
    else:
        print("Current clearML task not found, will not sync artifact to clearML")
