from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from typing import List, Dict, Union, Tuple
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import re

class K_Asag:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', cache_dir: str = None):
        self.model_name = model_name
        try:
            self.model = SentenceTransformer(self.model_name, cache_folder=cache_dir, device='cpu')
        except Exception as e:
            raise RuntimeError(f"Failed to load model {self.model_name}: {str(e)}")
        self.questions: List[Dict[str, Union[str, float]]] = []
        self.threshold_low = 0.6
        self.threshold_high = 0.7

    def kims_bedding(self, texts: Union[str, List[str]]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        if not all(isinstance(t, str) and t.strip() for t in texts):
            raise ValueError("All inputs must be non-empty strings")
        try:
            embeddings = self.model.encode(texts, convert_to_tensor=True, show_progress_bar=False)
            return embeddings.cpu().numpy()
        except Exception as e:
            raise RuntimeError(f"Failed to generate embeddings: {str(e)}")

    def kosine(self, text1: str, text2: str) -> float:
        if not (isinstance(text1, str) and isinstance(text2, str) and text1.strip() and text2.strip()):
            raise ValueError("Both inputs must be non-empty strings")
        try:
            embeddings = self.kims_bedding([text1, text2])
            return cosine_similarity(embeddings[0:1], embeddings[1:2])[0][0]
        except Exception as e:
            raise RuntimeError(f"Failed to compute similarity: {str(e)}")

    def exact_match(self, rubric: str, answer: str) -> bool:
        return rubric.strip().lower() == answer.strip().lower()

    def keyword_match(self, answer: str, rubric: str, min_keywords: float = 0.5) -> bool:
        rubric_words = set(re.findall(r'\w+', rubric.lower()))
        answer_words = set(re.findall(r'\w+', answer.lower()))
        common_words = rubric_words.intersection(answer_words)
        return len(common_words) / len(rubric_words) >= min_keywords if rubric_words else False

    def kwestions(self, question: str, rubric: str, mode: str = 'hybrid', weight: float = 1.0):
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question must be a non-empty string")
        if not isinstance(rubric, str) or not rubric.strip():
            raise ValueError("Rubric must be a non-empty string")
        if mode not in ['semantic', 'exact', 'keyword', 'hybrid']:
            raise ValueError("Mode must be 'semantic', 'exact', 'keyword', or 'hybrid'")
        if not isinstance(weight, (int, float)) or weight <= 0:
            raise ValueError("Weight must be a positive number")
        self.questions.append({
            'question': question,
            'rubric': rubric,
            'mode': mode,
            'weight': weight
        })

    def grade_answers(self, answers: List[str], pass_threshold: float = 0.5) -> Tuple[Dict, int]:
        if not self.questions:
            return {'error': 'No questions available'}, 400
        if not isinstance(answers, list):
            raise ValueError('Answers must be a list')
        if not answers or any(not isinstance(a, str) or not a.strip() for a in answers):
            raise ValueError('All answers must be non-empty strings')
        if len(answers) != len(self.questions):
            raise ValueError(f'Mismatch: {len(answers)} answers for {len(self.questions)} questions')

        results = []
        score = 0.0
        max_score = sum(q['weight'] for q in self.questions)

        for i, (question, answer) in enumerate(zip(self.questions, answers)):
            if not answer.strip():
                results.append({'question': question['question'], 'score': 0.0, 'similarity': None})
                continue
            rubric = question['rubric']
            mode = question['mode']
            weight = question['weight']

            try:
                if mode == 'semantic':
                    similarity = self.kosine(answer, rubric)
                    question_score = 0.0 if similarity < 0.5 else (0.5 if similarity < 0.65 else 1.0)
                    results.append({'question': question['question'], 'score': question_score, 'similarity': similarity})
                    score += question_score * weight
                elif mode == 'exact':
                    question_score = 1.0 if self.exact_match(answer, rubric) else 0.0
                    results.append({'question': question['question'], 'score': question_score, 'similarity': None})
                    score += question_score * weight
                elif mode == 'keyword':
                    question_score = 1.0 if self.keyword_match(answer, rubric) else 0.0
                    results.append({'question': question['question'], 'score': question_score, 'similarity': None})
                    score += question_score * weight
                elif mode == 'hybrid':
                    similarity = self.kosine(answer, rubric)
                    key_score = 1.0 if self.keyword_match(answer, rubric) else 0.0
                    exact_score = 1.0 if self.exact_match(answer, rubric) else 0.0
                    sem_score = 0.0 if similarity < 0.6 else (0.5 if similarity < 0.65 else 1.0)
                    hybrid_score = 0.30 * sem_score + 0.30 * key_score + 0.40 * exact_score
                    question_score = max(sem_score, hybrid_score)
                    results.append({'question': question['question'], 'score': question_score, 'similarity': similarity})
                    score += question_score * weight
            except RuntimeError as e:
                results.append({'question': question['question'], 'score': 0.0, 'similarity': None, 'error': str(e)})
                continue

        result = 'Pass' if score >= pass_threshold * max_score else 'Fail'
        return {
            'score': score,
            'max_score': max_score,
            'result': result,
            'details': results
        }, 200

    def evaluate_performance(self, test_data: List[Tuple[List[str], List[bool]]]) -> Dict[str, float]:
            y_true = []
            y_pred = []
            y_scores = []

            for answers, labels in test_data:
                if len(answers) != len(self.questions) or len(labels) != len(self.questions):
                    print(f"Skipping mismatched entry: {len(answers)} answers, {len(labels)} labels")
                    continue
                result, status = self.grade_answers(answers)
                if status != 200:
                    print(f"Skipping invalid result: {result['error']}")
                    continue
                for i, detail in enumerate(result['details']):
                    y_pred.append(detail['score'] > 0)
                    y_true.append(labels[i])
                    y_scores.append(detail['score'])

            if not y_true:
                return {'accuracy': 0.0, 'f1_score': 0.0, 'precision': 0.0, 'recall': 0.0, 'mae': 0.0, 'rmse': 0.0}

            y_true_numeric = [1 if l else 0 for l in y_true]
            mae = np.mean(np.abs(np.array(y_scores) - np.array(y_true_numeric)))
            rmse = np.sqrt(np.mean((np.array(y_scores) - np.array(y_true_numeric)) ** 2))

            cm = confusion_matrix(y_true_numeric, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Incorrect", "Correct"])
            disp.plot(cmap=plt.cm.Blues)
            plt.title("Confusion Matrix")
            plt.show()

            return {
                'accuracy': accuracy_score(y_true, y_pred),
                'f1_score': f1_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred),
                'recall': recall_score(y_true, y_pred),
                'mae': mae,
                'rmse': rmse
        }
