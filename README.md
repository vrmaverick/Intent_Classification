# 🧠 Intent Classification using Utterances

This project aims to predict customer **intents** based on **utterances** (text inputs) using a variety of machine learning and deep learning models. The dataset contains customer service-related queries, and the goal is to classify these into specific intent categories.

---

## 📂 Project Structure
├── data/
│ ├── X_train.csv
│ ├── X_val.csv
│ ├── y_train.csv
│ └── y_val.csv
├── train/
│ ├── main.py # Model architecture definitions
│ ├── evaluate.py # Evaluation metrics and reports
│ ├── data.py # Data loading and preprocessing
│ ├── vectorize.py # Text vectorization and label encoding
│
├── intent.ipynb # Exploratory experiments and results
├── README.md

---

## 📊 Dataset Overview

- **Input:** `utterance` (short user queries or messages)
- **Target:** `intent` (corresponding action or query type)

> The dataset is already pre-split into training (6539 samples) and validation (818 samples) sets.

---

## 🔧 Models Implemented

| Model              | Type               | Description                                      |
|-------------------|--------------------|--------------------------------------------------|
| **Naive Bayes**    | Classical ML       | Baseline using TF-IDF + BernoulliNB              |
| **1D CNN**         | Deep Learning      | Text CNN model with convolutional layers         |
| **LSTM**           | Deep Learning      | Sequence modeling with Long Short-Term Memory    |
| **GRU**            | Deep Learning      | Gated Recurrent Unit-based model                 |
| **Bi-RNN**         | Deep Learning      | Bidirectional RNN for context from both sides    |
| **USE + Dense**    | Transfer Learning  | Universal Sentence Encoder + Dense layer         |

---
## 💾 Pretrained Models from this project

Trained models for all architectures are saved and can be downloaded here:

🔗 [Google Drive - Trained Models](https://drive.google.com/drive/folders/1lQyII07bK66LO7k9x0-fODNTSpP6QA5i?usp=sharing)

---
## 🚀 To Run the Project

```bash
pip install -r requirements.txt
cd training/
python main.py       # Define and train model
```
---
# In Future Updates : 
I want to Build a web API for real-time intent classification
---
# Contact : [vedantranade2612@gmail.com](vedantranade2612@gmail.com)
# Portfolio : [My Portfolio](https://vedant-ranade.netlify.app/)

