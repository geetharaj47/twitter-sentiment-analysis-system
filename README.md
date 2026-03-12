<div align="center">

# Twitter Sentiment Analysis System

### Transformer-Powered NLP Application for Real-Time Tweet Sentiment Prediction

</div>

<p align="center">

<a href="#technologies">Technologies</a> | <a href="#features">Features</a> | <a href="#layout">Layout</a> | <a href="#installation">Installation</a>

</p>

---

A full machine learning application that analyzes tweet sentiment using **BERT Transformers**.
The system provides a **FastAPI backend for model inference** and a **Streamlit interface for interactive predictions**.

The project demonstrates a **complete NLP pipeline** from classical ML to deep learning and transformer models.

---

## 💻 Technologies

### Machine Learning

* **Python** – Core programming language
* **PyTorch** – Deep learning framework
* **HuggingFace Transformers** – BERT implementation
* **Scikit-learn** – Baseline machine learning models
* **Pandas** – Data processing and manipulation

### Backend

* **FastAPI** – REST API for model inference
* **Uvicorn** – ASGI server for FastAPI

### Frontend

* **Streamlit** – Interactive web application for sentiment prediction

### Development Tools

* **Git & GitHub** – Version control
* **VS Code** – Development environment

---

## 🚀 Features

* Transformer-based sentiment classification using **BERT**
* Baseline comparison using **TF-IDF + Logistic Regression**
* Deep learning implementation using **Bi-LSTM**
* REST API for real-time predictions
* Interactive Streamlit web interface
* Clean modular machine learning project structure
* Production-style model loading and inference pipeline

---

## 🎨 Layout

### Streamlit Web Application

![Streamlit UI](assets/streamlit_ui.png)

### FastAPI Interactive Documentation

![FastAPI Docs](assets/api_docs.png)

---

## ⚙️ System Architecture

```
User Input
     ↓
Streamlit Web Interface
     ↓
FastAPI API
     ↓
BERT Transformer Model
     ↓
Sentiment Prediction
```

---

## 📦 Installation

Clone the repository

```
git clone https://github.com/YOUR_USERNAME/twitter-sentiment-analysis-system.git
cd twitter-sentiment-analysis-system
```

Install dependencies

```
pip install -r requirements.txt
```

---

## ▶ Running the API

```
uvicorn src.api.app:app --reload
```

Open:

```
http://127.0.0.1:8000
```

API documentation:

```
http://127.0.0.1:8000/docs
```

---

## ▶ Running the Web Interface

```
streamlit run src/ui/streamlit_app.py
```

---

## 📊 Example Prediction

Input

```
I love this phone
```

Output

```
Sentiment: Positive 😀
```

---

## 👨‍💻 Author

**GeethaRaj**

Engineering Student
Aspiring Data Analyst / Machine Learning Engineer
