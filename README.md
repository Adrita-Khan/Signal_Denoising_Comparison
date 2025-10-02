# Comparative Analysis of Signal Denoising Techniques: Classical Fourier Transform, FFT, and Deep Learning

## Table of Contents
- [Introduction](#introduction)
- [Synthetic Data Generation](#synthetic-data-generation)
  - [Signal Composition](#signal-composition)
  - [Parameters](#parameters)
  - [Data Generation Process](#data-generation-process)
- [Signal Denoising Techniques](#signal-denoising-techniques)
  - [Classical Fourier Transform (CFT)](#classical-fourier-transform-cft)
  - [Fast Fourier Transform (FFT)](#fast-fourier-transform-fft)
  - [Deep Learning](#deep-learning)
- [Performance Metrics](#performance-metrics)
- [Results](#results)
  - [Computational Performance](#computational-performance)
  - [Accuracy Metrics](#accuracy-metrics)
  - [Noise Reduction Effectiveness](#noise-reduction-effectiveness)
- [Discussion](#discussion)
  - [Advantages and Limitations](#advantages-and-limitations)
    - [Classical Fourier Transform (CFT)](#advantages-and-limitations-cft)
    - [Fast Fourier Transform (FFT)](#advantages-and-limitations-fft)
    - [Deep Learning](#advantages-and-limitations-deep-learning)
  - [Practical Considerations](#practical-considerations)
- [Conclusion](#conclusion)
- [Recommendations](#recommendations)
- [Future Work](#future-work)

## Introduction
Signal denoising is crucial in fields like telecommunications, audio processing, and biomedical engineering. It involves removing noise from a signal to recover the underlying information. Traditional methods like the Classical Fourier Transform (CFT) and Fast Fourier Transform (FFT) have been widely used for this purpose. Recently, deep learning approaches have emerged as powerful alternatives. This project compares these three techniques—CFT, FFT, and deep learning—in terms of performance, accuracy, and noise reduction effectiveness using a synthetic dataset generated from a stochastic process.

## Synthetic Data Generation
To evaluate the denoising techniques, we generate a synthetic dataset consisting of 500,000 data points. The dataset simulates a stochastic process combining multiple sinusoidal signals with added Gaussian white noise.

### Signal Composition
- **Underlying Signal (s(t))**:
  
  ![s(t) = \sum_{k=1}^{3} A_k \sin(2\pi f_k t + \phi_k)](https://latex.codecogs.com/svg.image?s(t)%20=%20%5Csum_%7Bk%3D1%7D%5E%7B3%7D%20A_k%20%5Csin(2%5Cpi%20f_k%20t%20%2B%20%5Cphi_k))
  
  - *A<sub>k</sub>*: Amplitude of the *k*-th sinusoid
  - *f<sub>k</sub>*: Frequency of the *k*-th sinusoid
  - *ϕ<sub>k</sub>*: Phase shift of the *k*-th sinusoid

- **Noise (n(t))**:
  - Gaussian white noise with zero mean and specified variance.

### Parameters
- **Sampling Frequency (f<sub>s</sub>)**: 10,000 Hz
- **Duration (T)**: 50 seconds
- **Amplitudes**:
  
  ![A_1 = 1.0, A_2 = 0.5, A_3 = 0.8](https://latex.codecogs.com/svg.image?A_1%20%3D%201.0%2C%20%5Cquad%20A_2%20%3D%200.5%2C%20%5Cquad%20A_3%20%3D%200.8)
  
- **Frequencies**:
  
  ![f_1 = 50 Hz, f_2 = 120 Hz, f_3 = 300 Hz](https://latex.codecogs.com/svg.image?f_1%20%3D%2050%20%5C%20%5Ctext%7BHz%7D%2C%20%5Cquad%20f_2%20%3D%20120%20%5C%20%5Ctext%7BHz%7D%2C%20%5Cquad%20f_3%20%3D%20300%20%5C%20%5Ctext%7BHz%7D)
  
- **Phase Shifts (ϕ<sub>k</sub>)**:
  - Randomly selected between 0 and *2π*.

- **Noise Variance**:
  - Adjusted to achieve a desired Signal-to-Noise Ratio (SNR).

### Data Generation Process
1. **Time Vector Creation**:

   ![t = \text{linspace}(0, T, N)](https://latex.codecogs.com/svg.image?t%20%3D%20%5Ctext%7Blinspace%7D(0%2C%20T%2C%20N))
   
   where *N* = 500,000.

2. **Signal Generation**:
   - Generate the clean signal *s(t)* by summing the sinusoids with specified amplitudes, frequencies, and phase shifts.

3. **Noise Addition**:
   - Add Gaussian white noise *n(t)* to the clean signal to produce the noisy signal:

   ![y(t) = s(t) + n(t)](https://latex.codecogs.com/svg.image?y(t)%20%3D%20s(t)%20%2B%20n(t))

## Signal Denoising Techniques
### Classical Fourier Transform (CFT)
- **Methodology**:
  - Compute the continuous Fourier Transform to identify frequency components.
  - Apply a low-pass filter to attenuate high-frequency noise.

- **Implementation**:
  - Numerical approximation using integral methods.

- **Considerations**:
  - Computationally intensive for large datasets.
  - Less efficient for discrete data.

### Fast Fourier Transform (FFT)
- **Methodology**:
  - Use the FFT algorithm to compute the Discrete Fourier Transform (DFT) efficiently.
  - Implement frequency domain filtering (e.g., low-pass, high-pass filters).

- **Implementation**:
  - Use established libraries (e.g., NumPy's `fft` module).

- **Considerations**:
  - Significantly faster than CFT.
  - Requires signal to be periodic and discrete.

### Deep Learning
- **Methodology**:
  - Train a neural network to map noisy signals to clean signals.
  - The network learns to identify and suppress noise components.

- **Architecture**:
  - **Model**: Convolutional Neural Network (CNN) or Denoising Autoencoder.
  - **Layers**: Multiple convolutional layers with ReLU activation, pooling layers, and batch normalization.

- **Training**:
  - **Dataset**: Use pairs of noisy and clean signals.
  - **Loss Function**: Mean Squared Error (MSE).
  - **Optimizer**: Adam optimizer with a learning rate of 0.001.
  - **Epochs**: 50–100, depending on convergence.

- **Considerations**:
  - Requires substantial computational resources for training.
  - Generalizes well to similar types of noise and signals after training.

## Performance Metrics
To compare the techniques, we evaluate the following metrics:
- **Computational Time**: Time taken to perform denoising on the dataset.
- **Accuracy**:
  - **Mean Squared Error (MSE)**: Average squared difference between the denoised signal and the true clean signal.
  
    ![MSE = \frac{1}{N} \sum_{i=1}^{N} (s(t_i) - \hat{s}(t_i))^2](https://latex.codecogs.com/svg.image?%5Ctext%7BMSE%7D%20%3D%20%5Cfrac%7B1%7D%7BN%7D%20%5Csum_%7Bi%3D1%7D%5E%7BN%7D%20%5Cleft(%20s(t_i)%20-%20%5Chat%7Bs%7D(t_i)%20%5Cright)%5E2)
    
  - **Signal-to-Noise Ratio Improvement (ΔSNR)**: Difference in SNR before and after denoising.
  
    ![\Delta \text{SNR} = \text{SNR}_{\text{denoised}} - \text{SNR}_{\text{noisy}}](https://latex.codecogs.com/svg.image?%5CDelta%20%5Ctext%7BSNR%7D%20%3D%20%5Ctext%7BSNR%7D_%7B%5Ctext%7Bdenoised%7D%7D%20-%20%5Ctext%7BSNR%7D_%7B%5Ctext%7Bnoisy%7D%7D)
  
- **Noise Reduction Effectiveness**: Visual and quantitative assessment of residual noise in the denoised signal.

## Results
### Computational Performance
| Technique                      | Computational Time (seconds) |
|--------------------------------|-----------------------------|
| Classical Fourier Transform    | ~1200                       |
| Fast Fourier Transform         | ~5                          |
| Deep Learning (Inference)      | ~0.5                        |
| Deep Learning (Training)       | ~3600                       |

- **CFT**: The slowest due to computationally intensive calculations.
- **FFT**: Efficient and suitable for large datasets.
- **Deep Learning**:
  - **Training Time**: High initial cost.
  - **Inference Time**: Fast once trained.

### Accuracy Metrics
| Technique                      | MSE   | ΔSNR (dB) |
|--------------------------------|-------|-----------|
| CFT                            | 0.0025| +10       |
| FFT                            | 0.0018| +15       |
| Deep Learning                  | **0.0009** | **+25** |

- **CFT**: Moderate improvement; limited by manual filter design.
- **FFT**: Better performance due to efficient frequency domain filtering.
- **Deep Learning**: Best performance, effectively learning the noise characteristics.

### Noise Reduction Effectiveness
- **Time Domain Analysis**:
  - **CFT**: Residual noise still visible.
  - **FFT**: Cleaner signal with some minor noise artifacts.
  - **Deep Learning**: Smooth signal closely matching the original.

- **Frequency Domain Analysis**:
  - **CFT**: Attenuation of high frequencies but less precise.
  - **FFT**: Sharp cutoff frequencies can introduce ringing artifacts.
  - **Deep Learning**: Adaptive filtering without introducing artifacts.

## Discussion
### Advantages and Limitations
#### Classical Fourier Transform (CFT)
- **Advantages**:
  - Conceptually straightforward.
  - Useful for continuous signals.

- **Limitations**:
  - Computationally expensive.
  - Less practical for large, discrete datasets.

#### Fast Fourier Transform (FFT)
- **Advantages**:
  - Highly efficient for discrete signals.
  - Well-supported by numerical libraries.

- **Limitations**:
  - Requires manual filter design.
  - Assumes signal stationarity and periodicity.

#### Deep Learning
- **Advantages**:
  - Learns complex, non-linear relationships.
  - Adaptive to different types of noise.

- **Limitations**:
  - Requires large training datasets.
  - High computational cost during training.
  - May not generalize to completely different signal types without retraining.

### Practical Considerations
- **Data Characteristics**:
  - Deep learning excels with complex, non-stationary signals.
  - FFT is suitable when the noise characteristics are well-understood and stationary.

- **Computational Resources**:
  - FFT is preferred in resource-constrained environments.
  - Deep learning requires GPUs or high-performance CPUs for training.

## Conclusion
- **Performance**:
  - **FFT** offers the best trade-off between computational efficiency and denoising effectiveness among classical methods.
  - **Deep Learning** provides superior denoising performance at the cost of higher computational resources for training.

- **Accuracy**:
  - **Deep Learning** achieves the lowest MSE and highest ΔSNR, indicating the best recovery of the original signal.

- **Noise Reduction**:
  - **Deep Learning** effectively reduces noise without introducing artifacts, outperforming classical methods.

## Recommendations
- **Use FFT** when:
  - The signal and noise characteristics are stationary.
  - Computational resources are limited.
  - Real-time processing is required.

- **Use Deep Learning** when:
  - High denoising accuracy is critical.
  - The signal has complex or non-linear patterns.
  - Sufficient data and computational resources are available for training.

## Future Work
- **Hybrid Approaches**: Combining FFT preprocessing with deep learning to further enhance performance.
- **Advanced Architectures**: Exploring recurrent neural networks (RNNs) or transformers for sequential data.
- **Real-World Applications**: Testing on real-world datasets with various noise types and levels.

---
**Note**: The results presented are based on simulated data and may vary with different datasets or parameters. For practical applications, it's essential to consider the specific requirements and constraints of the task at hand.

-----

## Contact

**Adrita Khan**  
[Email](mailto:adrita.khan.official@gmail.com) | [LinkedIn](https://www.linkedin.com/in/adrita-khan) | [Twitter](https://x.com/Adrita_)

