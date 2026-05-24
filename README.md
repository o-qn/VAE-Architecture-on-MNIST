# MNIST Variational Autoencoder (VAE) Generative Model

This repository contains my final project for the university Computer Vision course. The objective of this project was to design, implement, and train a Variational Autoencoder (VAE) using the MNIST dataset of handwritten digits. By mapping the input data to a smooth, continuous latent space, the model is capable of synthesizing entirely new, realistic digit images.

The complete implementation—including network architecture design, custom loss functions, training loops, and generation scripts—is available in `Vae_mnist_FULLcode FINAL.py`.

## What I Did in This Project

Unlike standard deterministic autoencoders used for simple compression or denoising, this project required establishing a probabilistic latent space to make the model truly generative. The core phases of the project include:

* **Data Loading and IDX Handling:** Developed routines to read and parse the standard MNIST binary byte files (`idx1-ubyte` and `idx3-ubyte` formats) for training and evaluation.
* **Architecture Implementation (Encoder/Decoder):** Constructed an asymmetric neural network architecture. The encoder compresses input images into a low-dimensional bottleneck, while the decoder reconstructs the compressed vectors back into full-resolution $28 \times 28$ pixel images.
* **The Reparameterization Trick:** Implemented a stochastic layer that outputs the mean ($\mu$) and log variance ($\log \sigma^2$) of a Gaussian distribution. By applying the reparameterization trick, the network introduces random sampling while preserving a differentiable path for backpropagation during training.
* **Custom Loss Function Design:** Created a joint loss function optimizing two distinct mathematical targets: Reconstruction Loss (ensuring the output matches the input) and Kullback-Leibler (KL) Divergence (regularizing the latent space to adhere strictly to a standard normal distribution).
* **Generative Sampling & Visualization:** Programmed sampling mechanics to bypass the encoder entirely post-training, drawing random vectors directly from the regularized latent space to synthesize crisp, new handwritten digits.

## Key Insights and Methodology

The fundamental challenge of a VAE is managing the delicate balance between its two loss components. If the reconstruction loss dominates, the model acts like a standard autoencoder, resulting in a fractured latent space with poor generative qualities. If the KL divergence dominates, the latent space is perfectly uniform, but the generated images lack structural definition. Finding the right training equilibrium allowed this network to smoothly transition between digits (e.g., morphing a 3 into an 8) across the continuous latent landscape.

## Want to Run it Yourself?

If you want to view the network training or generate some digits yourself, follow these steps:

### 1. Grab the dependencies
Ensure you have the required Python data science and machine learning packages installed:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn torch torchvision
```
### 2. Run the pipeline
Ensure the MNIST dataset files are extracted and available in your working directory path as expected by the script, then execute:
```bash 
python "Vae_mnist.py"
```
