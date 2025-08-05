
---

# SeRI: Efficient Gradient-Free Sensitive Region Identification via Boundary-Guided Search in Decision-Based Black-Box Attacks

## Requirements

- **Python Version**: 3.11.5
- **Libraries**:
  - PyTorch 2.3.0
  - Torchvision 0.18.0

## Installation

1. **Install Python**: Ensure you have Python 3.11.5 installed. If not, download it from the official [Python website](https://www.python.org/downloads/release/python-3115/).

2. **Install Libraries**: Install the required Python libraries using the following command:
   ```bash
   pip install torch==2.3.0 torchvision==0.18.0
   ```

## Models

Download the following pre-trained models and place them in the `/code/model/` directory:

- [VGG19](https://download.pytorch.org/models/vgg19-dcbb9e9d.pth)
- [ResNet50](https://download.pytorch.org/models/resnet50-11ad3fa6.pth)
- [Inception V3](https://download.pytorch.org/models/inception_v3_google-0cc3c7bd.pth)
- [Vision Transformer (ViT) B_32](https://download.pytorch.org/models/vit_b_32-d86f8d99.pth)

## Dataset Setup


### ImageNet
Download the ImageNet dataset from the following Kaggle link:
- [ImageNet Mini](https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000/data)

## Usage

To run SeRI, use the following command structure. 'norm' is the base attacker such as CGBA and ADBA; budget and budget2 are base attacker budget and SeRI budget.

```bash
python main.py --dataset=mnist-cnn --targeted=0 --norm=CGBA --zip=SeRI --budget=800 --budget2=200 --epsilon=1.0 --early=0 --beginIMG=0 --imgnum=10 --remember=1
```

