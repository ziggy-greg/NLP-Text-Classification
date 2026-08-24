# Resume Classification Using NLP

## Overview

An NLP-based **multi-class resume classification system** that automatically categorises resumes into **24 professional job categories**.

The project compares traditional machine learning and deep learning approaches, while evaluating **TF-IDF** and **GloVe** text representations.

## Models

* Multinomial Naïve Bayes
* XGBoost
* Bidirectional LSTM (BiLSTM)
* GRU

## Dataset

* **4,966 labelled resumes**
* **24 job categories**
* Combination of CSV and PDF resume data
* Moderate class imbalance

## NLP Pipeline

**Text Cleaning → Tokenisation → Stopword Removal → Lemmatisation → Feature Extraction → Model Training → Evaluation**

## Results

| Model       | Representation     |   Accuracy |
| ----------- | ------------------ | ---------: |
| **XGBoost** | **TF-IDF**         | **95.98%** |
| BiLSTM      | Learned Embeddings |     93.36% |
| XGBoost     | GloVe              |     90.34% |
| GRU         | GloVe              |     87.73% |
| MNB         | TF-IDF             |     75.05% |

**Best Model:** XGBoost + TF-IDF
**Accuracy:** 95.98%
**Weighted F1:** 95.95%
**Balanced Accuracy:** 94.76%

## Key Findings

* XGBoost + TF-IDF achieved the strongest overall performance.
* Hyperparameter tuning substantially improved Naïve Bayes performance.
* GloVe did not outperform TF-IDF across the evaluated models.
* BiLSTM achieved the strongest deep learning performance but showed signs of overfitting.
* Model performance depended strongly on the interaction between the representation and classifier.

## Technologies

**Python · Pandas · NumPy · Scikit-learn · NLTK · spaCy · XGBoost · TensorFlow/Keras · GloVe · Matplotlib · Seaborn · pdfplumber**

## Repository Structure

```text
NLP-Text-Classification/
├── notebooks/
│   └── nlp_project.ipynb
├── report/
│   └── FINAL NLP REPORT-ZIGGY.docx
└── .gitignore
```

## Author

**Ziggy Greg Uwas**
MSc Artificial Intelligence & Data Science — University of Hull

