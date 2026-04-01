<!-- ````markdown -->
# 🔍 Duplicate Question Pair Detector

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg?logo=streamlit&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-F9AB00.svg?logo=huggingface&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine_Learning-NLP-brightgreen.svg)

An interactive Natural Language Processing (NLP) web application designed to detect semantic duplication in question pairs. 

Traditional string-matching often fails on edge cases (e.g., *"What is good for eyes?"* vs *"What is not good for eyes?"*). This project solves that by combining the vector-space distance of **Bi-Encoders** with the logical entailment/contradiction analysis of **Cross-Encoders**.

---

## ✨ Features

* **Two-Stage NLP Pipeline:** Uses a Bi-Encoder for semantic similarity and a DeBERTa-based Cross-Encoder for Natural Language Inference (NLI).
* **Interactive UI:** Built with Streamlit for real-time analysis.
* **Adjustable Thresholds:** Tweak the Cosine Similarity and Contradiction Logit thresholds on the fly via the sidebar.
* **Explainable Results:** Doesn't just give a "Yes" or "No"—it explains *why* based on similarity scores and NLI metrics (Contradiction, Neutral, Entailment).
* **Model Caching:** Optimized model loading ensures the heavy transformer models are only loaded into memory once.

---

## 🧠 How It Works

The core logic relies on `sentence-transformers` and operates in a logical cascade:

1.  **Semantic Similarity (Bi-Encoder):** * Model: `all-MiniLM-L6-v2`
    * Computes the dense vector embeddings of both questions and calculates their **Cosine Similarity**. 
    * *If the similarity is below the threshold (C1), they are immediately flagged as NOT duplicates.*
2.  **Natural Language Inference (Cross-Encoder):** * Model: `cross-encoder/nli-deberta-v3-base`
    * If similarity is high, the model checks for logical contradiction. 
    * *If the contradiction logit exceeds the threshold (C2), they are flagged as NOT duplicates (handling the "not" edge cases).*
3.  **Final Verdict (C3):**
    * If similarity is high AND contradiction is low, the questions are confirmed as **Duplicates**.

---

## 🚀 Installation & Setup

### Prerequisites
* Python 3.8 or higher
* Git

<!-- ```` -->
### Step-by-Step Guide

1. **Clone the repository:**
   ```bash
   git clone https://github.com/HARSH-GOHIL-git/Duplicate-Questions-Pair-Detection.git
   cd Duplicate-Questions-Pair-Detection
    ```
2.  **Create a virtual environment (Recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

    *(Note: Ensure your `requirements.txt` includes `streamlit`, `sentence-transformers`, `torch`, and `numpy`)*

4.  **Run the Streamlit application:**

    ```bash
    streamlit run app.py
    ```

5.  **Open in Browser:** The app will automatically open in your default browser at `http://localhost:8501`.

-----

## 🛠️ Tech Stack

  * **Language:** Python
  * **Frontend Framework:** Streamlit
  * **NLP Libraries:** `sentence-transformers`, `transformers`
  * **Models:** \* `all-MiniLM-L6-v2` (Microsoft / HuggingFace)
      * `nli-deberta-v3-base` (Cross-Encoder / NLI)


-----

## 👨‍💻 Author

**Harsh Lokeshbhai Gohil**

  * GitHub: [https://github.com/HARSH-GOHIL-git](https://github.com/HARSH-GOHIL-git)
  * LinkedIn: [https://www.linkedin.com/in/harsh-gohil-02723a22a](https://www.linkedin.com/in/harsh-gohil-02723a22a)

-----

### ⭐ Show your support

If you found this project helpful or interesting, please consider giving it a star\!
