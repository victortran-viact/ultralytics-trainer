from clearml import Model

def download_model(model_id: str) -> str:
    '''Download model from ClearML Registry'''
    print(f"Download model_id {model_id} from clearML Model Registry")
    model = Model(model_id=model_id)
    tmp_path = model.get_local_copy(extract_archive=True, force_download=True)
    # if not tmp_path:
    #     raise ValueError(
    #         "Could not download model, you must mistake InputModel & OutputModel")
    print(f"Model stored at {tmp_path}")
    return tmp_path