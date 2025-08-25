import unittest
import json
import numpy as np
from k_asag import K_Asag
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class TestKAsag(unittest.TestCase):
    def setUp(self):
        self.model = K_Asag(model_name='all-MiniLM-L6-v2', cache_dir='./model_cache')

    def load_test_data(self):
        with open('english_questions.json', 'r') as f_q:
            questions = json.load(f_q)
        with open('english_rubrics.json', 'r') as f_r:
            rubrics = json.load(f_r)
            if isinstance(rubrics, dict): rubrics = rubrics.get("rubrics", [])
        with open('english_answers.json', 'r') as f_a:
            answer_sets = json.load(f_a)

        if len(questions) != len(rubrics):
            self.fail("Mismatch in questions and rubrics length")

        for item in answer_sets:
            if len(item['answers']) != len(questions) or len(item['labels']) != len(questions):
                answer_sets.remove(item)

        return questions, rubrics, answer_sets

    def setup_model(self, questions, rubrics):
        for q, r in zip(questions, rubrics):
            self.model.kwestions(q, r, mode='hybrid', weight=1.0)

    def test_embedding_generation(self):
        text = "Sample text for embedding."
        embeddings = self.model.kims_bedding(text)
        self.assertIsInstance(embeddings, np.ndarray)
        self.assertEqual(embeddings.shape[0], 1)

    def test_cosine_similarity(self):
        similarity = self.model.kosine("Lagos is the capital", "The capital city is Lagos.")
        self.assertGreater(similarity, 0.7)

    def test_invalid_input(self):
        with self.assertRaises(ValueError): self.model.kims_bedding("")
        with self.assertRaises(ValueError): self.model.kosine("valid", "")
        with self.assertRaises(ValueError): self.model.kwestions("", "rubric")
        #with self.assertRaises(ValueError): self.model.grade_answers([None])
        #with self.assertRaises(ValueError): self.model.grade_answers(["Answer", 123])

    def test_grading(self):
        questions, rubrics, answer_sets = self.load_test_data()
        self.setup_model(questions, rubrics)
        for i, answer_set in enumerate(answer_sets):
            result, status = self.model.grade_answers(answer_set['answers'])
            self.assertEqual(status, 200)
            self.assertEqual(len(result['details']), len(questions))
            print(f"Answer Set {i+1}: Score {result['score']}/{result['max_score']}, Result: {result['result']}")

    def test_performance_evaluation(self):
        questions, rubrics, answer_sets = self.load_test_data()
        self.setup_model(questions, rubrics)
        test_data = [(a['answers'], a['labels']) for a in answer_sets]
        metrics = self.model.evaluate_performance(test_data)
        print("\n Metrics for K_ASAG:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
            self.assertGreaterEqual(value, 0.0)
        print(f"\nRMSE: {metrics.get('rmse', 0.0):.4f}")

        y_true = []
        y_pred = []
        for answers, labels in test_data:
            result, status = self.model.grade_answers(answers)
            if status != 200:
                continue
            for i, detail in enumerate(result['details']):
                y_pred.append(detail['score'] > 0)
                y_true.append(labels[i])

        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Incorrect", "Correct"])
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.show()
if __name__ == '__main__':
    unittest.main()
