# Signal Denoising Techniques: A Comparative Analysis

## Overview
This study compares three signal denoising approaches—Classical Fourier Transform (CFT), Fast Fourier Transform (FFT), and Deep Learning—using a synthetic dataset of 500,000 points to evaluate performance, accuracy, and noise reduction effectiveness.


## Methods

### Classical Fourier Transform (CFT)
- Computes continuous Fourier Transform via numerical integration
- Applies low-pass filtering to attenuate high-frequency noise
- **Limitation:** Computationally intensive for large datasets

### Fast Fourier Transform (FFT)
- Efficiently computes Discrete Fourier Transform using FFT algorithm
- Implements frequency domain filtering (low-pass/high-pass)
- **Advantage:** Significantly faster than CFT for discrete signals

### Deep Learning
- **Architecture:** Convolutional Neural Network (CNN) or Denoising Autoencoder
- **Training:** 50-100 epochs, Adam optimizer (lr=0.001), MSE loss
- **Process:** Maps noisy signals to clean signals through learned patterns
- **Limitation:** Requires substantial training resources upfront

-----

**Contact:** Adrita Khan  
[Email](mailto:adrita.khan.official@gmail.com) | [LinkedIn](https://www.linkedin.com/in/adrita-khan) | [Twitter](https://x.com/Adrita_)
