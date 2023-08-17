from clearml import Model
from ultralytics import YOLO


def download_model(model_id: str) -> str:
    '''Download model from ClearML Registry'''
    model = Model(model_id=model_id)
    tmp_path = model.get_local_copy(
        extract_archive="./"
    )
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
        "--model_id", help="Model from ClearML Registry", default=None
    )

    args.add_argument(
        "--model_path", help="Model form local storage", default=None
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
    elif args.model_id :
        model_path = download_model(args.model_path)
    
    export_to_onnx(
        model_path=model_path,
    )
