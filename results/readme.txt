results are stored here.

./results
 |-- checkpoints   # Models
 |    |-- DATASET_NAME1
 |    |    |-- C2H.pth     # MSDT
 |    |    |-- Ten.pth     # Teacher encoder
 |    |    |-- Tde.pth     # Shared/Teacher decoder
 |    |    |-- Ten_Best_SSIM.pth
 |    |    |-- Tde_Best_SSIM.pth
 |    |    |-- Sen.pth     # Student encoder
 |    |    |-- Sen_Best_PSNR.pth   # (only when semi-supervised)
 |    |    |-- Sen_Best_SSIM.pth   # (only when semi-supervised)
 |    |-- DATASET_NAME2
 |    |-- ...
 |-- dehaze        # Dehazing images
 |    |-- DATASET_NAME1
 |    |    |-- inference   # inference images
 |    |    |    |-- IMG1.png
 |    |    |    |-- IMG2.png
 |    |    |    |-- ...
 |    |    |-- val_psnr    # (only when semi-supervised)
 |    |    |-- val_ssim    # (only when semi-supervised)
 |    |-- DATASET_NAME2
 |    |-- ...
 |-- gehaze        # Generated haze images
 |    |-- source-DATASET_NAME1
 |    |    |-- target-DATASET_NAME1
 |    |    |    |-- IMG1.png
 |    |    |    |-- IMG2.png
 |    |    |-- target-DATASET_NAME2
 |    |    |-- target-...
 |    |-- source-DATASET_NAME2
 |    |-- source-...
 |-- recimg        # Reconstructed images
 |    |-- DATASET_NAME1
 |    |    |-- IMG1.png
 |    |    |-- IMG2.png
 |    |    |-- ...
 |    |-- DATASET_NAME2
 |    |-- ...
 |-- logs          # Log files
 |    |-- logs.txt                 # Log
 |    |-- DATASET_NAME1
 |    |    |-- train_loss_C2H.json # Loss log
 |    |    |-- train_loss_C2C.json
 |    |    |-- train_loss_H2C.json
 |    |    |-- val_loss_H2C.json   # (only when semi-supervised)
 |    |-- DATASET_NAME2
 |    |-- ...
 |-- stylized_database             # Stylized database
 |    |-- DATASET_NAME1
 |    |    |-- reliable_style_bank # Best version
 |    |    |    |-- IMG1.png
 |    |    |    |-- IMG2.png
 |    |    |    |-- ...
 |    |    |-- stylizedEPOCH1      # Version1
 |    |    |    |-- IMG1.png
 |    |    |    |-- IMG2.png
 |    |    |    |-- ...
 |    |    |-- stylizedEPOCH2      # Version2
 |    |    |-- stylizedEPOCH3
 |    |    |-- stylizedEPOCH4
 |    |    |-- stylizedEPOCH5
 |    |    |-- stylizedEPOCH6
 |    |-- DATASET_NAME2
 |    |-- ...