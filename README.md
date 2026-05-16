1. Problem Description
This project implements a Convolutional Neural Network (CNN) to classify handwritten digits (0-9) using the MNIST dataset. The goal is to build a deep learning model using PyTorch, apply enhancement and regularization techniques, and compare the performance of different optimizers (Adam vs. SGD) to analyze their effect on convergence and accuracy.

2. Dataset Link
The MNIST dataset is used for this project. (It is automatically downloaded via the torchvision library.)

Official Link: http://yann.lecun.com/exdb/mnist/
Note: The project code automatically downloads this dataset, so no manual download is required 

3. Data Preprocessing
Normalization: Applied using transforms.Normalize((0.1307,), (0.3081,)) based on the global mean and standard deviation of the MNIST dataset to speed up convergence.
Resizing: Not required as all MNIST images are uniformly 28x28 pixels.
Encoding: Labels are already integer-encoded (0-9), which is directly compatible with PyTorch's CrossEntropyLoss.
Partitioning: Data is split into 80% Training (48,000 images), 20% Validation (12,000 images), and the official Test set (10,000 images).

5. Model Architecture (CNN)
The model is a Convolutional Neural Network built using PyTorch:

Conv Block 1: Conv2d (1->32 channels) + BatchNorm + ReLU + MaxPool2d.
Conv Block 2: Conv2d (32->64 channels) + BatchNorm + ReLU + MaxPool2d.
Fully Connected: Flatten layer -> Linear (6477 -> 128) + ReLU + Dropout -> Linear (128 -> 10).
Loss Function: nn.CrossEntropyLoss

5. Model Enhancement Techniques (Justified)
Data Augmentation (transforms.RandomRotation(10)): Randomly rotates training images by up to 10 degrees. Justification: Handwritten digits can be slightly tilted; this forces the model to learn tilt-invariant features and improves generalization.
Batch Normalization (nn.BatchNorm2d): Normalizes the activations of the convolutional layers. Justification: Reduces internal covariate shift, allowing for higher learning rates, faster convergence, and more stable training.
Dropout (nn.Dropout(0.5)): Randomly zeros 50% of the neurons in the dense layer during training. Justification: Prevents the network from relying too heavily on specific neurons, thereby mitigating overfitting.

7. Experimentation & Results
Two experiments were conducted by varying the Optimizer while keeping the architecture and other parameters constant.

Model	Optimizer	Learning Rate	Test Accuracy	Test Loss
Model A	Adam	0.001	99.30%	0.0232
Model B	SGD (momentum=0.9)	0.01	99.21%	0.0242
Analysis & Comparison
Both models achieved exceptional accuracy (>99%). The Adam optimizer converged slightly faster and reached a marginally higher accuracy compared to SGD with momentum in just 10 epochs. Adam combines the benefits of momentum and adaptive learning rates, making it highly efficient for this task, whereas SGD required a carefully tuned learning rate and momentum to achieve similar results.

7. Visualizations
The training and validation curves for Loss and Accuracy are plotted to monitor the training process and check for overfitting.

Training vs Validation Curves

8. Instructions for Running the Project
Prerequisites: Ensure you have Python 3.x installed.
Clone the repository:
git clone (https://github.com/bobahany/deep-project/edit/main/README.md)
Install required libraries: It is recommended to use the provided requirements file.
pip install -r requirements.txt
Run the code: Open the Jupyter Notebook / Google Colab file (.ipynb) and run all cells sequentially. The dataset will download automatically.
