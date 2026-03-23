# Brain_tumor_mri_CNN
Developed a CNN-based model to detect and classify brain tumors from MRI scans. Performed image preprocessing and augmentation to improve model accuracy, demonstrating effective use of deep learning for medical imaging applications.
# Brain Tumor Detection (MRI) using CNN (VGG16)

## Description
Developed a **CNN model using VGG16** to classify brain tumors from MRI scans into 4 categories. Applied preprocessing, data augmentation, and transfer learning for improved accuracy and robust predictions.

## Motivation
Early detection of brain tumors can save lives. Automated deep learning models reduce manual diagnostic effort and support radiologists in decision-making.

## Tech Stack
Python, TensorFlow, Keras, VGG16, NumPy, Pandas, Matplotlib

## Features
- Transfer learning using **VGG16**  
- Frozen base layers for efficient training  
- Added custom layers: Flatten, Dropout, Dense  
- Output layer with 4-class softmax classification  
- MRI image preprocessing and data augmentation  

## Example Code Snippet
```python
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout

# Load VGG16 base model
base_model = VGG16(
    include_top=False,
    input_shape=(128,128,3)
)

# Freeze base layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers
x = base_model.output
x = Flatten()(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
output = Dense(4, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
