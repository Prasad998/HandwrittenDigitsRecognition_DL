# HandwrittenDigitsRecognition_DL
Deep Learning based project with Simple Feed Forward Neural Network for recognition of handwritten digits.
---

# 🖊️ Handwritten Digits Recognition (MNIST)

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Keras](https://img.shields.io/badge/Keras-DeepLearning-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A simple deep learning project for recognizing **handwritten digits (0–9)** from the [MNIST dataset](http://yann.lecun.com/exdb/mnist/).
Built using **TensorFlow/Keras**, the model achieves high accuracy (>97%) on the test dataset.

---

## 📊 Dataset

* **MNIST**: 70,000 grayscale images of handwritten digits.

  * Training: 60,000 images
  * Testing: 10,000 images
* Each image is **28×28 pixels**.

---

## 🧠 Model Architectures

This project explores multiple **Keras Sequential Models**:

### 1️⃣ Single Dense Layer

```python
model = keras.Sequential([
    keras.layers.Dense(10, input_shape=(784,), activation='sigmoid')
])
```

### 2️⃣ Two-Layer Dense Network

```python
model = keras.Sequential([
    keras.layers.Dense(100, input_shape=(784,), activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
])
```

### 3️⃣ Flatten + Dense (Best Performing)

```python
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
])
```

---

## ⚙️ Installation

Clone this repository and install dependencies:

```bash
git clone https://github.com/yourusername/HandwrittenDigitsRecognition.git
cd HandwrittenDigitsRecognition
pip install -r requirements.txt
```

Or install manually:

```bash
pip install tensorflow matplotlib seaborn numpy
```

---

## ▶️ Usage

Run the Jupyter notebook:

```bash
jupyter notebook HandwrittenDigitsRecognition.ipynb
```

## 📈 Results

* Achieved **97%+ accuracy** on the MNIST test dataset.
* Below is the **confusion matrix** visualization:

<p align="center">
  <img src="https://github.com/user-attachments/assets/26879998-4c62-4cf5-8580-debf155400ce" alt="Confusion Matrix" width="500">
</p>
---

## 🔮 Future Work

* Use **Softmax** instead of Sigmoid for multi-class classification.
* Implement **Convolutional Neural Networks (CNNs)** for higher accuracy.
* Add **Dropout/BatchNorm** for better generalization.

---

## 🙌 Acknowledgements

* [TensorFlow/Keras](https://www.tensorflow.org/)
* [MNIST Dataset](http://yann.lecun.com/exdb/mnist/) by Yann LeCun

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---
