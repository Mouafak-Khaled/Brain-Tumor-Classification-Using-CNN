# Brain-Tumor-Classification-Using-CNN
Brain Tumor MRI Image Classification Using Convolution Neural Networks 


# The Architecture of BrainTumorCNN:
## 1. First Layer:

* A 2D convolution layer:
  * Kernel size of 3 x 3.
  * Stride of 2
  * 128 Filters
* Batch normalization
## 2. Second Layer:
* A 2D convolution layer:
  * Kernel size of 3 x 3.
  * Stride of 2
  * 256 Filters
* Max Poooling Layer:
  * Kernel size of 3 x 3
  * Stride of 2
* Batch normalization
## 3. Third Layer:
* A 2D convolution layer:
  * Kernel size of 3 x 3.
  * Stride of 1
  * 128 Filters
* Max Poooling Layer:
  * Kernel size of 3 x 3
  * Stride of 2
* Batch normalization
## 4. Fourth Layer:
* A 2D convolution layer:
  * Kernel size of 3 x 3.
  * Stride of 1
  * 64 Filters
* Max Poooling Layer:
  * Kernel size of 3 x 3
  * Stride of 2
* Batch normalization
## 5. Fifth Layer:
* Flatten Layer
## 6. Sixth Layer:
* Fully Connected Layer:
  * 1024 - 512 neurons
  * Dropout with 0.25 probability
  * Layer normalization
## 7. Seventh Layer:
* Fully Connected Layer:
  * 512 - 256 neurons
  * Dropout with 0.25 probability
  * Layer normalization
## Output Layer
  * Fully connected layer
    * 4 classes
## Activation Function:
  * LeakyReLU with negative slope of 0.1
  * The weights are initialized using kaiming normal.
  

# Training:

## 1. Optimizer:
* AdamW.
* With a learning rate of 1e-3.
* And a weight decay of 1e-2.
* With amsgrad.

## 2. Criterion (Loss function):
* Cross entropy loss.

## 3. Epochs:
* The model was trained for 45 epochs

## 4. Learning rate scheduler:
* StepLR.
* With a step size of 15.
* And a gamma of 0.1


# Evaluation:
* Accuracy:

| Train | Validation | Test |
|-------|-------------|------|
| 97.72%| 93.39%      | 93.82%|

* Precision, recall, and f1-score:

| classes | precision | recall | f1-score | support |
|---------|-----------|--------|----------|---------|
| Glioma       | 0.96      | 0.87   | 0.92     | 300    |
| Meningioma      | 0.91      | 0.90   | 0.90     | 306     |
| No Tumor       | 0.95      | 0.98   | 0.96     | 405     |
| Pituitery      | 0.93      | 0.98   | 0.96     | 300     |



# Dataset:
The dataset used for training this CNN is Brain Tumor MRI Classification.
The dataset was downloaded using Kaggle.
* The dataset consists of images of three types of brain tumors and images of brains with no tumors.
* There are 4 different classes in the dataset.
* A transform is applied on the dataset to make sure that all images are of the same size which is 224 x 224

