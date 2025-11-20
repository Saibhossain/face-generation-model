# 🧠 From Zero to Faces: My GAN Learning Roadmap

This repository documents my journey into Deep Learning and Generative Adversarial Networks (GANs). My ultimate goal is to build, train, and deploy a custom model capable of generating photorealistic human faces.

## 📌 Project Overview
* **Goal:** Create a generative model for human faces and deploy it as a web app.
* **Framework:** PyTorch
* **Current Status:** Moving from Vanilla GANs to Deep Convolutional GANs (DCGAN).

---

## 🗺️ The Roadmap

### Phase 1: Foundations & Prerequisites
*Focus: Mastering the tools and math required for Deep Learning.*
- [x] **Python & NumPy:** Matrix operations and data handling.
- [x] **PyTorch Basics:** Tensors, Autograd, and basic Neural Networks.
- [x] **Math Concepts:** Understanding Probability Distributions (Gaussian) and Gradient Descent.
- [x] **Checkpoint Project:** Build a simple Neural Network to classify MNIST digits.

### Phase 2: The "Hello World" of GANs (Vanilla GAN)
*Focus: Understanding the Adversarial Game.*
- [x] **Concept:** The Minimax Game (Generator vs. Discriminator).
- [x] **Architecture:** Using Linear (Dense) layers.
- [x] **Loss Function:** Binary Cross Entropy (BCELoss).
- [x] **Checkpoint Project:** Generate handwritten digits (0-9) using the MNIST dataset.
    - *Outcome:* Successfully generated recognizable digits from random noise.

### Phase 3: Generating Images (DCGAN) 👈 **(I Am Here)**
*Focus: Moving from flat vectors to spatial image generation.*
- [ ] **Concept:** Transposed Convolutions (Upsampling) vs. Standard Convolutions.
- [ ] **Architecture:** Implementing the DCGAN paper guidelines (BatchNorm, LeakyReLU).
- [ ] **Data Prep:** Pre-processing the CelebA dataset (Resize to 64x64, Normalize).
- [ ] **Checkpoint Project:** Train a DCGAN to generate 64x64 color faces.

### Phase 4: High-Fidelity Generation (ProGAN / StyleGAN)
*Focus: Photorealism and high resolution.*
- [ ] **Concept:** Progressive Growing (4x4 → 1024x1024).
- [ ] **Concept:** Style Transfer and AdaIN (Adaptive Instance Normalization).
- [ ] **Checkpoint Project:** Load a pre-trained StyleGAN2/3 model and generate HD faces.

### Phase 5: Control & Deployment
*Focus: Turning the model into a usable product.*
- [ ] **Latent Space:** Manipulation (Aging, Gender swapping, Interpolation).
- [ ] **App Logic:** Wrapping the model in a Python script (`model.py` & `app.py`).
- [ ] **Deployment:** Hosting the model on Hugging Face Spaces using Streamlit.

---

## 📂 Repository Structure

```text
├── 01_mnist_vanilla_gan/   # My first GAN (Simple digits)
│   ├── train.py
│   └── output_images/
├── 02_celeba_dcgan/        # Current Work (Faces)
│   ├── model.py            # Generator & Discriminator classes
│   ├── train.py            # Training loop
│   └── utils.py            # Image plotting helper functions
├── 03_deployment/          # Web App code
│   ├── app.py
│   └── requirements.txt
└── README.md# face-generation-model
