# SFDA-CellSeg
This project was originally developed for the paper: **[Towards Source-Free Cross Tissues Histopathological Cell Segmentation via Target-Specific Finetuning](https://ieeexplore.ieee.org/abstract/document/10087318/)**. If you use this project in your research, please cite the following works:  

    @article{li2023towards,
    title={Towards Source-Free Cross Tissues Histopathological Cell Segmentation via Target-Specific Finetuning},
    author={Li, Zhongyu and Li, Chaoqun and Luo, Xiangde and Zhou, Yitian and Zhu, Jihua and Xu, Cunbao and Yang, Meng and Wu, Yenan and Chen, Yifeng},
    journal={IEEE Transactions on Medical Imaging},
    year={2023},
    publisher={IEEE}

# DataSet

The TNBC dataset with mask annotations can be downloaded from: **[TNBC](https://ieee-dataport.org/documents/segmentation-nuclei-histopathology-images-deep-regression-distance-map#files)**.

The TCIA dataset with mask annotations can be downloaded from: **[TCIA](https://www.nature.com/articles/s41597-020-0528-1#Sec15)**.

The KIRC dataset with mask annotations can be downloaded from: **[KIRC](https://www.worldscientific.com/doi/abs/10.1142/9789814644730_0029)**.

# Requirements

torch>=1.9.0

opencv-python>=4.5.1.10

SimpleCRF==0.1.0

matplotlib>=3.3.1

Python >= 3.6

TensorBoardX

Some basic python packages such as Numpy, Scikit-image, Scipy ......

# Usage

Train the model

`python train_with_UNet.py --epochs=200 --batch-size=4 --mode=Source --batch-size=2 --save-every=100`

Get evaluate results images

Firstly, move the evaluate result masks to the eval folder in data folder.

Then run `python canny.py`

# Acknowledgement
* The GatedCRFLoss is adapted from [GatedCRFLoss](https://github.com/LEONOB2014/GatedCRFLoss) for medical image segmentation.
