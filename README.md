# MDST-Dehaze

## Environment

```
python=3.12
torch=2.5
torchvision=0.20.0
numpy=1.26.0 (Error when <1.26.0)
pillow=11.0.0
opencv-python=4.11.0.86
visdom=0.2.4 (Optional)
```

## Quick Inference

### Step 1: Prepare dataset

Download the datasets from [BauduPan](https://pan.baidu.com/s/1yInNV9FilBj1tEBAGZ1t3w?pwd=mdst)

```
datasets
 |-- DATASET_NAME
 |    |-- test
 |    |    |-- clear
 |    |    |-- haze
 |    |-- train
 |    |    |-- clear
 |    |    |-- haze
```

### Step 2: Prepare pretrained checkpoints

Download the pretrained model weights from [BauduPan](https://pan.baidu.com/s/1yInNV9FilBj1tEBAGZ1t3w?pwd=mdst)

```
results
 |-- checkpoints
 |    |-- DATASET_NAME
 |    |    |-- Tde.pth
 |    |    |-- Sen.pth
```

### Step 3: Prepare config

Edit `./configs/configs_inferenceH2C.py`, where `DATASET_NAME` is the name of your dataset.

```
category = DATASET_NAME
haze_dir = '../datasets/DATASET_NAME/test/haze/'
saved_model_Sen_path = '../results/checkpoints/DATASET_NAME/Sen.pth'
saved_model_Tde_path = '../results/checkpoints/DATASET_NAME/Tde.pth'
```

### Step 4: Run

```
python inference_H2C.py
```