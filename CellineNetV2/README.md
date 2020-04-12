# CellineNetV2 in PyTorch
CellineNetV2 is a convolutional neural network(CNN) built on top of an efficiently scaled MobileNetV2.
`MobileNetv2` is an efficient CNN architecture for mobile devices. For more information check the paper:
[Inverted Residuals and Linear Bottlenecks: Mobile Networks for Classification, Detection and Segmentation](https://arxiv.org/abs/1801.04381) 
MobileNetv2 Implementation courtesy proceeds to: Evgeniy Zheltonozhskiy; https://github.com/Randl


## Usage

Clone the repo:
```bash
git clone https://github.com/Randl/MobileNetV2-pytorch
pip install -r requirements.txt
```

Use the model defined in `model.py` to run CelllineNetV2 example:
```bash
python celline_net.py --dataroot "/path/"
```

## Results

For x1.0 model Evgeniy achieved 0.3% higher top-1 accuracy than claimed.
 
|Classification Checkpoint| MACs (M)   | Parameters (M)| Top-1 Accuracy| Top-5 Accuracy|  Claimed top-1|  Claimed top-5|
|-------------------------|------------|---------------|---------------|---------------|---------------|---------------|
|   [mobilenet_v2_1.0_224]|300         |3.47           |          72.10|          90.48|           71.8|           91.0|
|   [mobilenet_v2_0.5_160]|50          |1.95           |          60.61|          82.87|           61.0|           83.2|


