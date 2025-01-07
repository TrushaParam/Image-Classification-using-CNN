# Image Classification using Convolutional Neural Networks

This repository contains an implementation of an image classification system using Convolutional Neural Networks (CNNs). The project focuses on training a deep learning model to classify images from a dataset, leveraging the power of CNNs for feature extraction and accurate predictions.

## Overview
This project demonstrates the process of building an image classification system using Convolutional Neural Networks (CNNs), a state-of-the-art deep learning architecture designed for image recognition tasks. The goal is to classify images into their respective categories with high accuracy by training a model on the CIFAR-10 dataset. The CIFAR-10 dataset consists of 60,000 32x32 colour images divided into 10 classes, such as aeroplanes, cars, birds, and more.

Key aspects of this project include:
- Preprocessing the dataset for optimal model training.
- Designing a CNN architecture tailored to the dataset's characteristics.
- Experimenting with different regularization techniques like batch normalization and dropout.
- Comparing the custom-designed CNN with well-known architectures such as AlexNet and ResNet.

This project serves as an excellent starting point for understanding and applying CNNs to image classification problems and provides insights into optimizing model performance.

## Features
- **Dataset Handling:** Prepares the CIFAR-10 dataset for training and testing, including normalization and data augmentation.
- **Custom CNN Design:** Implements a tailored CNN architecture for the classification task.
- **Performance Optimization:** Includes experiments with hyperparameters, optimizers, and regularization techniques.
- **Model Comparison:** Benchmarks the custom CNN against popular architectures for a comprehensive performance analysis.
- **Visualization:** Provides metrics such as accuracy and loss curves to evaluate model performance.

## Prerequisites
To run this project, ensure the following dependencies are installed:

- Python 3.8 or above
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Jupyter Notebook (optional for running the provided notebook)

## Dataset
The project uses the CIFAR-10 dataset, which is automatically downloaded through the TensorFlow/Keras datasets API. Ensure your environment has internet access to fetch the dataset.

## Results
The custom-designed CNN achieves competitive accuracy on the CIFAR-10 dataset. Experiments demonstrate the impact of different optimizers, learning rates, and regularization techniques on model performance. Detailed evaluation metrics, confusion matrices, and visualizations of predictions are included in the notebook.

## Contribution
Contributions are welcome! If you have ideas to improve the model or add features, feel free to fork the repository and submit a pull request.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments
- This project was done during my master's degree at the University of Adelaide

