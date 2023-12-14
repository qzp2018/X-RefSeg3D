# X-RefSeg3D
NEWS:🔥 **X-RefSeg3D** is accepted at **AAAI2024**!🔥  
X-RefSeg3D: Enhancing Referring 3D Instance Segmentation via Structured Cross-Modal Graph Neural Networks  
Zhipeng Qian, Yiwei Ma, Jiayi Ji, Xiaoshuai Sun*  

# Installation
## 0. Package Versions
* Packages
    ```
    conda install -c conda-forge tqdm
    conda install -c anaconda scipy
    conda install -c conda-forge scikit-learn
    conda install -c open3d-admin open3d
    conda install pytorch==1.4.0 torchvision==0.5.0 -c pytorch
    conda install -c huggingface transformers
    ```
* Follow instructions from https://github.com/facebookresearch/SparseConvNet to download SparseConvNet.
##  1. Dataset Download

### Scannet Download
For the Scannet Dataset please go to https://github.com/ScanNet/ScanNet and fill out the agreement form to download the dataset.

### ScanRefer Download
For the ScanRefer Dataset please go to https://github.com/daveredrum/ScanRefer and fill out the agreement form to download the dataset.

### Glove Embeddings
Download the [*preprocessed glove embeddings*](http://kaldir.vc.in.tum.de/glove.p) from [*ScanRefer*](https://github.com/daveredrum/ScanRefer).

## 2. Data Organization
```
scannet_data
|--scans

This Repository
|--glove.p
|--ScanRefer
    |--Files from ScanRefer download
```

## 3. Data Preprocessing
First store the point cloud data for each scene into pth files.
```
python prepare_data.py
```
Split the files into train and val folders.
```
python split_train_val.py
```
## 4. Pretrained Models
Please download the [*pretrained instance segmentation model*](https://www.dropbox.com/sh/u2mozpyzycwomwc/AABbYCbZPKGu8foT3bQc_jdna?dl=0) and place into the folder like this.
```
This Repository
|--GRU
    |--checkpoints
        |--model_insseg-000000512.pth
|--BERT
    |--checkpoints
        |--model_insseg-000000512.pth
```
Pretrained model for [*referring model with gru encoder*](https://www.dropbox.com/sh/u2mozpyzycwomwc/AABbYCbZPKGu8foT3bQc_jdna?dl=0) and place into the folder like this.
```
This Repository
|--GRU
    |--checkpoints
        |--gru
            |--models
                |--gru-000000032.pth
```
Pretrained model for [*referring model with bert encoder*](https://www.dropbox.com/sh/u2mozpyzycwomwc/AABbYCbZPKGu8foT3bQc_jdna?dl=0) and place into the folder like this.
```
This Repository
|--BERT
    |--checkpoints
        |--bert
            |--models
                |--bert-000000064.pth
```
## 5. Training and Validation
We pre-save the visual features of validation, resulting in significant time savings. Additionally, we concurrently perform both training and validation steps.  
For example, we pre-save the visual features in GRU mode.
```
cd GRU/
python feat_save.py
```
Then the visual features for validation are saved as ```val_feat.pkl```.  

Train the referring model with GRU encoder. 
```
cd GRU/
python X-RefSeg3D_gru.py
```
Train the referring model with BERT encoder.
```
cd BERT/
python X-RefSeg3D_bert.py
```
## 6. Acknowledgements
Our dataloader and training implementations are modified from https://github.com/hanhung/TGNN, which is obtained by reference to https://github.com/facebookresearch/SparseConvNet and https://github.com/daveredrum/ScanRefer, please go check out their repositories for sparseconvolution and 3D referring object localization implementations respectively. We would also like to thank the teams behind TGNN, Scannet and ScanRefer for providing their pre-trained models and dataset.
