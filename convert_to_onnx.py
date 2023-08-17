from clearml import Model, Task
from ultralytics import YOLO


def download_model(model_id: str) -> str:
    '''Download model from ClearML Registry'''
    print(f"Download `model_id` {model_id}")
    model = Model(model_id=model_id)
    tmp_path = model.get_local_copy(extract_archive=True, force_download=True)
    if not tmp_path: 
        raise ValueError("Could not download model, you must mistake InputModel & OutputModel")
    print(f"Model stored at {tmp_path}")
    return tmp_path
    

# TODO: 
def upload_model_to_clearml() -> str:
    return


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
    
    output_name = export_to_onnx(
        model_path=model_path,
    )

    print(f"ONNX model stored at: {output_name}")
    with open("./yolov5su.onnx","rb") as f: 
        model = f.read()

    # to upload with clearML
    print(f"Uploading to clearML server")
    task = Task.current_task()
    task.upload_artifact(
        name=output_name,
        # artifact_object=f"./{output_name}"
        artifact_object=model
    )