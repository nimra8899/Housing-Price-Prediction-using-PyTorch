# Housing-Price-Prediction-using-PyTorch

## 📌 Project Overview

This project uses a neural network to predict house prices based on features like income, location, and number of rooms. The model is trained on the California Housing dataset.

---

## 🚀 Features

* Data preprocessing and normalization
* Train-validation split
* Neural network with PyTorch
* Loss tracking using TensorBoard
* Best model saving based on validation loss

---

## 🧠 Model Architecture

* Input Layer: 8 features
* Hidden Layer 1: 64 neurons (ReLU)
* Hidden Layer 2: 32 neurons (ReLU)
* Output Layer: 1 neuron (price prediction)

---

## 📊 Dataset

* Source: California Housing Dataset (Scikit-learn)
* Features include:

  * Median income
  * House age
  * Number of rooms
  * Population
  * Latitude & Longitude

---

## ⚙️ Technologies Used

* Python
* PyTorch
* Scikit-learn
* TensorBoard

---

## ▶️ How to Run

1. Clone the repository:

```
git clone https://github.com/your-username/housing-price-prediction.git
```

2. Navigate to the folder:

```
cd housing-price-prediction
```

3. Install dependencies:

```
pip install -r requirements.txt
```

4. Run the project:

```
python train.py
```

---

## 📈 Results

* Model improves over epochs
* Training and validation loss tracked
* Best model saved as `best_model.pt`

---

## 📸 TensorBoard (Optional)

To visualize training:

```
tensorboard --logdir=runs
```

---

## 👩‍💻 Author
Nimra Jabbar