#!/bin/bash

python train_C2H.py # output Stylized-Database, which is used for dehazing of Stage-2
python train_C2C.py # output Ten.pth and Tde.pth, which serve as Teacher-Encoder and Shared-Decoder
python train_H2C.py # output dehazing images using Stylized-Database, Ten.pth and Tde.pth