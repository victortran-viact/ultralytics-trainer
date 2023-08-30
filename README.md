# ultralytics-trainer
Ultralytics Scripts


## Install requirements: 
```
pip install -r requirements.txt
```


## Train: 
```
python train_yolo.py \
    --dataset_id 845f63796967477eb2896a11a487a19e \
    --model_version "yolov5s" \
    --batch_size 16 \
    --imgsz 640 \
    --epochs 2
```

## Finetune: 
```
python finetune_yolo.py \ 
    --dataset_id e5bd66f0631d4aec914ba542e7eedcd3 \
    --model_id 2166c61eaa2749d0ba68111ae3faf9bc \
    --batch_size 16 \
    --imgsz 640 \
    --epochs 2
```


## Export ONNX: 
```
python convert_to_onnx.py \ 
    --model_id 2166c61eaa2749d0ba68111ae3faf9bc 
```


## Evaluate only: 
```
python test_yolo.py \ 
    --dataset_id e5bd66f0631d4aec914ba542e7eedcd3 \ 
    --model_id 2166c61eaa2749d0ba68111ae3faf9bc
```

## Package OPS: 
```
python package_ops.py \ 
    --model_id 2166c61eaa2749d0ba68111ae3faf9bc \
    --model_type danger_zone \ 
    --version 1.0.0 \
    --label_list Person Car 
```



## Benchmark (TODO)