# BookFlix – Movie-to-Book Recommendation System

## Overview

BookFlix is a content-based recommendation system that suggests books based on movies a user enjoys. The system leverages Natural Language Processing (NLP) techniques to identify similarities between movie descriptions and book summaries, enabling meaningful cross-domain recommendations.

This project demonstrates practical applications of:

* Text preprocessing
* Feature engineering
* Vectorization (TF-IDF)
* Similarity-based recommendation systems

---

## Methodology

### 1. Data Sources

The project uses three datasets:

* **Movies dataset** – Contains popular movies with metadata such as overview, genres, and popularity.
* **Books dataset** – Includes book titles, authors, descriptions, ratings, and awards.
* **Movie adaptations dataset** – Used to identify and remove books already adapted into movies.

### 2. Data Preprocessing

* Removed books that have existing movie adaptations
* Handled missing values in critical columns
* Cleaned textual data (lowercasing, removing punctuation)
* Tokenized and structured text fields

### 3. Feature Engineering

* Combined:

  * Movie overview
  * Movie genres
* Created a unified **tags** column for similarity comparison

### 4. Vectorization

* Applied **TF-IDF (Term Frequency–Inverse Document Frequency)** to convert text into numerical vectors
* Generated:

  * Movie vectors
  * Book vectors

### 5. Similarity Computation

* Used **cosine similarity** to measure the closeness between movie and book vectors
* Ranked books based on similarity scores

### 6. Recommendation Output

For a given movie, the system returns the top N recommended books including:

* Title
* Author
* Rating
* Awards (if available)

---

## Project Structure

```
BookFlix/
│── data/
│   ├── movie.csv
│   ├── books.csv
│   ├── best_movie_adaptations.csv
│
│── book_recommendation_model.pkl
│── main.ipynb / script.py
│── README.md
```

---

## Technology Stack

* **Programming Language:** Python
* **Libraries:**

  * Pandas, NumPy – Data manipulation
  * NLTK, Regex – Text preprocessing
  * Scikit-learn – TF-IDF and cosine similarity
  * Matplotlib, Seaborn – Data visualization

---

## Installation

```bash
git clone https://github.com/your-username/bookflix.git
cd bookflix
pip install -r requirements.txt
```

---

## Usage

```python
user_movie = "Inception"
recommended_books = recommend_books(user_movie)

print(f"Books recommended for fans of {user_movie}:")
print(recommended_books)
```

---

## Example Output

```python
[
  {
    'title': 'The Silent Patient',
    'author': 'Alex Michaelides',
    'ratings': 4.1
  },
  {
    'title': 'Dark Matter',
    'author': 'Blake Crouch',
    'ratings': 4.2
  }
]
```

---

## Model Persistence

The trained model is saved as:

```
book_recommendation_model.pkl
```

This file contains:

* Movie vectors
* Book vectors
* Processed datasets

---

## Key Features

* Content-based recommendation system
* Cross-domain mapping (Movies → Books)
* Efficient similarity computation using TF-IDF
* Clean and structured data preprocessing pipeline
* Model serialization using Pickle

---

## Future Enhancements

* Develop a web application using Streamlit or Flask
* Integrate external APIs (e.g., IMDb, Goodreads)
* Improve recommendations using deep learning embeddings (e.g., BERT)
* Add user preference-based filtering
* Enhance evaluation metrics for recommendation quality

---

## Author

**Jigisha Tomar**
[LinkedIn](http://www.linkedin.com/in/jigishatomar)
[GitHub](https://github.com/jigisha789)

---

## License

