
# Iran–Israel Tweet Sentiment Classification using Advanced NLP Vectorization

## Overview

This project presents a **comprehensive comparative study of word vectorization techniques** for **multi-class sentiment classification** on real-world social media data related to the Iran–Israel conflict.

The objective is to evaluate how different text representation methods influence model performance in capturing **context, semantics, and sentiment polarity** in noisy, short-form text like tweets.

## Problem Statement

> **Comparative Analysis of Word Vectorization Techniques for Multi-Class Sentiment Classification on Social Media Text**

Social media text is:

* noisy
* context-dependent
* short and ambiguous

This project explores how different vectorization techniques perform in extracting meaningful representations for classification.

## Dataset

* **Source**: Self-collected via web scraping
* **Domain**: Tweets related to the Iran–Israel geopolitical conflict
* **Preprocessing**:
  * Lowercasing
  * Stopword removal
  * Tokenization
  * Cleaning URLs, mentions, and hashtags

### Classes

* 🇮🇷 **Pro-Iran**
* ⚖️ **Neutral**
* 🇮🇱 **Pro-Israel**

## Methodology

# 🔹 Word Vectorization Techniques

This project implements and compares:

| Technique                   | Description                             |
| --------------------------- | --------------------------------------- |
| **Bag of Words (BoW)**      | Frequency-based representation          |
| **TF-IDF**                  | Importance-weighted word frequency      |
| **Word2Vec**                | Context-based embeddings                |
| **PPMI + SVD (GloVe-like)** | Matrix factorization-based embeddings   |
| **BERT (DistilBERT)**       | Transformer-based contextual embeddings |

---

### 🔹 Classification Models

| Model                   | Description                                       |
| ----------------------- | ------------------------------------------------- |
| **Logistic Regression** | Baseline linear classifier                        |
| **Linear SVM**          | Margin-based classifier for high-dimensional data |

---

## Project Architecture

```text
iran-israel-vectorization/
├── data/                  # Raw and processed datasets
├── notebooks/             # Jupyter notebooks (experiments, BERT model)
├── src/                   # Core implementation (vectorizers, preprocessing)
├── results/               # Evaluation metrics, plots, outputs
├── docs/                  # Supporting documentation
├── README.md              # Project documentation
├── requirements.txt       # Dependencies
└── .gitignore

This project demonstrates the **evolution from classical NLP to modern transformer-based AI**, highlighting the importance of representation learning in real-world classification problems.


