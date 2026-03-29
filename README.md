# 🎬 Hybrid Movie Recommendation System

## 📌 Overview

This project implements a **Hybrid Movie Recommendation System** that combines:

* 🎯 Content-Based Filtering (TF-IDF + Cosine Similarity)
* 👥 Collaborative Filtering (Matrix Factorization using SVD)
* 💡 Explainable AI (feature-based explanations)

The system recommends movies based on:

* Movie similarity (content)
* User preferences (ratings)

---

## 🚀 Features

* 🔍 Content-based recommendations using:

  * Genres
  * Keywords
  * Overview
  * Cast & Director
* 👤 Personalized recommendations using collaborative filtering (SVD)
* 🧠 Hybrid ranking (content + CF)
* 💬 Explainable recommendations (why a movie is recommended)
* ⚡ Efficient similarity computation using TF-IDF

---

## 🧠 Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn (TF-IDF, Cosine Similarity)
* Surprise Library (SVD)
* Streamlit (Frontend)

---

## 📂 Dataset Used

This project combines **MovieLens** and **TMDB datasets**.

### 🔹 MovieLens Dataset

* `ratings.dat`
* `movies.dat`

📥 Download:
https://grouplens.org/datasets/movielens/

---

### 🔹 TMDB Dataset

* `movies_metadata.csv`
* `credits.csv`
* `keywords.csv`
* `links.csv`

📥 Download:
https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset

---

## 🔄 Data Pipeline

1. Merge MovieLens and TMDB datasets using `movieId` and `tmdbId`
2. Clean and preprocess:

   * Genres → normalized
   * Keywords → extracted
   * Overview → cleaned + stopwords removed
   * Cast → top actors extracted
   * Crew → director extracted
3. Create `combined_features`
4. Apply TF-IDF vectorization
5. Compute cosine similarity matrix
6. Train SVD model on user ratings

---

## ⚙️ How It Works

### 🔹 Content-Based Filtering

* Converts movie features into TF-IDF vectors
* Uses cosine similarity to find similar movies

---

### 🔹 Collaborative Filtering (SVD)

* Learns user preferences from ratings
* Predicts ratings for unseen movies

---

### 🔹 Hybrid Recommendation

1. Generate candidate movies using content similarity
2. Rank them using SVD predicted ratings
3. Return top-N personalized recommendations

---

## 💡 Explainability (Key Feature)

The system explains **why a movie is recommended** using shared features.

### 🔍 Approach

* Convert features into sets:

  * Genres
  * Keywords
  * Cast
  * Director

* Compute overlap:

```python
common_features = set(movie1) & set(movie2)
```

* Generate explanation:

```text
"Recommended because it shares genres like animation and keywords like friendship."
```

---

### 🧠 Example

Input Movie: **Toy Story (1995)**

Recommendation: **A Bug’s Life (1998)**

Explanation:

* Same genre: Animation
* Shared keywords: friendship, adventure
* Similar audience

---

### 🎯 Importance

* Improves transparency
* Builds user trust
* Enhances user experience
* Adds Explainable AI capability

---

## 📊 Evaluation Metrics

The system was evaluated using both **ranking metrics** and **prediction error metrics**.

### 🔹 Collaborative Filtering (SVD)

* **RMSE:** 0.8737
  → Measures how accurately the model predicts user ratings
  → Lower is better

---

### 🔹 Ranking Metrics (Top-K Recommendations)

* **Precision@10:** 0.685
  → ~68.5% of recommended movies are relevant

* **Recall@10:** 0.641
  → ~64.1% of relevant movies are successfully retrieved

* **NDCG@10:** 0.8738 ⭐
  → Measures ranking quality (higher = better ordering)
  → Values close to 1 indicate highly relevant items are ranked at the top

---

### 🧠 Interpretation

* High **NDCG@10 (~0.87)** indicates excellent ranking quality
* Balanced **Precision and Recall** show good recommendation coverage
* Low **RMSE (~0.87)** indicates accurate rating predictions

👉 Overall, the hybrid system produces **accurate and well-ranked personalized recommendations**

---

## 🧪 Example

**Input:**

```text
Movie: Toy Story (1995)
```

**Output:**

* Toy Story 2 (1999)
* A Bug’s Life (1998)
* Small Soldiers (1998)

---

## 🛠️ Installation

```bash
pip install pandas numpy scikit-learn scikit-surprise streamlit
```

---

## ▶️ Run the Project

```bash
streamlit run app.py
```

---

## 📌 Key Concepts Used

* TF-IDF Vectorization
* Cosine Similarity
* Matrix Factorization (SVD)
* Hybrid Recommendation Systems
* Explainable AI

---

## 📈 Results

* NDCG@10: 0.87
* RMSE: 0.87

---

## 📌 Future Improvements

* Improve explanation quality (SHAP / attention models)
* Add deep learning-based recommenders
* Deploy on cloud
* Add real-time user interaction

---

## 👨‍💻 Author

Your Name

---

## ⭐ If you found this project useful, give it a star!
