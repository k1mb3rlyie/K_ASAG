# K\_ASAG: Automated Short Answer Grading System

K\_ASAG is a lightweight **Automated Short Answer Grading (ASAG)** system that leverages **semantic similarity (from SBERT embeddings)**, **keyword matching**, and **exact matching** to evaluate student answers against rubrics.
This project improves the ability for educators to leverage ASAG tools for pop quizzes and on-the-fly testing
It supports multiple grading modes (`semantic`, `exact`, `keyword`, and `hybrid`) and includes performance evaluation with standard metrics and confusion matrix visualization.

---

## Features

* Uses **Sentence-BERT (all-MiniLM-L6-v2)** embeddings for semantic similarity.
* Hybrid scoring strategy combining **semantic, keyword, and exact matches**.
* Configurable thresholds for pass/fail decisions. (But I wouldnt touch it if I were you, unless you really believe in yourself)
* Generates grading reports per answer set.
* Provides performance metrics:

  * Accuracy
  * F1-Score
  * Precision
  * Recall
  * Mean Absolute Error (MAE)
* Visualizes **confusion matrix** using matplotlib.

---

## Dataset

This project includes **custom JSON data** for evaluation (purrr):

* `english_answers.json` – Student answers.
* `english_rubrics.json` – Rubrics / reference answers.
* `english_questions.json` – Sample Questions (Biology) # Would perform similarly on questions from other subject domains

---

## Installation

Clone the repository:

```bash
git clone https://github.com/k1mb3rlyie/K_ASAG.git
cd K_ASAG
```

You can choose one of the following setup methods:

### `venv` (recommended for most users)

```bash
python3 -m venv venv

# Activate environment
# Linux/macOS: source venv/bin/activate
# Windows (PowerShell): .\venv\Scripts\Activate

pip install -r requirements.txt
```

### Conda

```bash
conda env create -f environment.yml
conda activate k_asag_env
```
---

## File Structure

```
project/
│── k_asag.py                # Core grading system
│── test_k_asag.py           # Unit tests / runnable test
│── demo_k_asag.py           # Demo script (will save to new file)
│── english_answers.json     # Sample student answers (130 students)
│── english_rubrics.json     # Rubrics (to 20 questions)
│── english_questions.json 
│── requirements.txt
│── README.md
```

---

## Future Improvements/reccomendations

* Add support for various **language support** across multiple answers.
* Web-based interface for teachers to upload answers and rubrics.
* Add anything you think could be helpful (❤️)

---

## Author

**Kimberly Tip’an Dawap**
