# SMART PREDICT

*Next Word Prediction using LSTM RNN*

## 📌 Project Objective

The project aims to develop a deep learning model that can **predict the next word** in a given sequence of words.
We have used **LSTM (Long Short-Term Memory)** — a type of Recurrent Neural Network (RNN) — which is well-suited for sequence prediction tasks.
Also used **GRU RNN** for better accuracy.

**Example:**

```
Input: I am a good  
Output: [predicted word]
```

## 📂 Dataset

* **Dataset Name**: *Shakespeare’s Hamlet*
* **Source**: Inbuilt in the **`Gutenberg`** corpus from the **NLTK** library

## 🧾 Input/Output Data

* **Input**: Text sequences ( previous words )
* **Output**: Predicted next word
* **Type**: Multi-class Classification
  → Because any word from the vocabulary could be the output

## ⚙️ Project Pipeline

```bash
Data Collection
   ↓
Data Preprocessing
   ↓
Model Building (LSTM RNN)
   ↓
Model Training
   ↓
Streamlit Web App
   ↓
Deployment
```

## 🧠 Model Highlights

* Sequence tokenization and padding
* Embedding Layer ( text -> vectors )
* LSTM layer to capture context from sequences
* Dense output layer with softmax activation
* Trained for multiple epochs using categorical crossentropy

## 🧪 Technologies Used

* TensorFlow and Keras
* NLTK
* Streamlit ( for building the web app and deployment )

## 🚀 Deployment

The model is wrapped in a **Streamlit Web App** for interactive next-word predictions.

