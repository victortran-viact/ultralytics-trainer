from pathlib import Path

from clearml import Task
from ultralytics import YOLO

from utils.clearml_utils import download_model


def export_to_onnx(model_path: str, format: str = 'onnx', imgsz: str = 640) -> str:
    '''
    Ultralytics ONNX args: imgsz, half, dynamic, simplify, opset
    Reference to the document for more details
    Example saved position:
        {workspace}
            /utils
                {model_weights}.onnx <- Ultralytics YOLO stored convert model here
                train_yolo.py
                ...
    '''
    model = YOLO(model=model_path)
    output_name = model.export(
        format=format,
        imgsz=imgsz
    )
    print(f"ONNX model stored at: {output_name}")
    return output_name

if __name__ == "__main__":
    import argparse
    args = argparse.ArgumentParser()

    args.add_argument(
        "--model_id", help="Model from ClearML Registry", type=str, default=None
    )

    args.add_argument(
        "--model_path", help="Model form local storage", default=None, type=str
    )

    args.add_argument(
        "--export_format", default="onnx", help="Export to model format, default is ONNX"
    )

    args.add_argument(
        "--imgsz", default=640, help="Image size"
    )

    args = args.parse_args()

    if not args.model_id and not args.model_path:
        raise ValueError("`model_id` or `model_path` must be provided")

    model_path = None
    if args.model_path:
        model_path = args.model_path
    elif args.model_id:
        model_path = download_model(args.model_id)

    output_path = export_to_onnx(
        model_path=model_path,
    )

    print(f"ONNX model stored at: {output_path}")
    # with open(output_path,"rb") as f:
    #     model = f.read()

    # to upload with clearML
    print(f"Uploading to clearML server")
    task = Task.current_task()

    # Move output_path to tmp dir
    temp_dir = Path(f"/tmp/{task.id}")

    temp_dir.mkdir(parents=True, exist_ok=True)

    onnx_model_path = temp_dir / Path(output_path).name
    Path(output_path).rename(onnx_model_path)

    task.upload_artifact(
        name="onnx_model",
        artifact_object=str(onnx_model_path),

    )
