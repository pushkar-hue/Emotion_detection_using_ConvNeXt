#  ConvNeXt Tiny - Image Classification with Live Detection

This project is a full image classification pipeline built using **PyTorch** and **ConvNeXt Tiny** architecture, enhanced with real-time prediction using **OpenCV**. The model is fine-tuned to classify images into 6 categories and is optimized for performance using techniques like transfer learning, class balancing, early stopping, and data augmentation.

---

##  Features

-  Fine-tuned **ConvNeXt Tiny** model (`torchvision.models`)
-  Classification into **6 custom classes**
-  Handles **imbalanced datasets** via `WeightedRandomSampler`
-  Uses extensive **image augmentation**
-  Supports **early stopping** and **model checkpointing**
-  **Dynamic LR scheduling** with `ReduceLROnPlateau`
- **Live prediction using OpenCV** via webcam stream

---

##  Model Overview

- **Base**: ConvNeXt Tiny pretrained on ImageNet
- **Output Layer**: Modified to output 6 classes
- **Trained Components**:
  - Last 2 blocks of the feature extractor
  - Full classifier

```python
model.classifier[2] = nn.Linear(in_features, 6)
