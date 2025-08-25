# K_Asag Demo: Short Answer Grading System
# By Kimberly Tip’an Dawap

from k_asag import K_Asag
import json, os
from datetime import datetime

def ensure_dir(path):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)

def save_json(data, folder):
    """Save data to a JSON file named with today’s date."""
    date = datetime.today().strftime('%Y-%m-%d')
    ensure_dir(folder)
    filename = os.path.join(folder, f"{date}.json")
    
    existing = []
    if os.path.exists(filename):
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                existing = json.load(f)
        except json.JSONDecodeError:
            pass

    existing.append(data)
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(existing, f, indent=4, ensure_ascii=False)

def setup_teacher():
    """Collect teacher questions and rubrics (English only)."""
    print("\n TEACHER SETUP")
    model = K_Asag(language='english', cache_dir='./model_cache')

    questions = []
    count = int(input("How many questions? "))
    for i in range(count):
        print(f"\nQuestion {i+1}")
        q = input("Enter question: ").strip()
        r = input("Enter correct answer (rubric): ").strip()
        m = input("Mode (semantic/exact/keyword/hybrid)? [default: semantic]: ").strip().lower() or 'semantic'
        questions.append({'question': q, 'rubric': r, 'mode': m, 'weight': 1.0})
        model.kwestions(q, r, m, 1.0)

    save_json({'questions': questions}, 'new_questions')
    save_json({'rubrics': [q['rubric'] for q in questions]}, 'new_rubrics')
    return model, questions

def get_student_answers(questions):
    """Collect student answers for each question."""
    print("\n STUDENT SUBMISSION")
    answers = []
    for i, q in enumerate(questions):
        ans = input(f"Answer {i+1} ({q['question']}): ").strip()
        answers.append(ans)
    save_json({'answers': answers}, 'new_answers')
    return answers

def display_result(result, student_answers):
    """Print grading results with actual student answers."""
    print("\n RESULT")
    print(f"Overall: {result['result']} — Score: {result['score']} / {result['max_score']}")
    for i, d in enumerate(result['details']):
        sim = d.get('similarity')
        sim_str = f"{sim:.2f}" if sim is not None else "N/A"
        student_ans = student_answers[i] if i < len(student_answers) else "No answer provided"
        print(f"- Q: {d['question']}")
        print(f"   Student Answer: {student_ans}")
        print(f"   Score: {d['score']:.2f} | Similarity: {sim_str}\n")

def main():
    print("\n Welcome to the K_Asag Demo (by Kimberly Tip’an Dawap)")
    model, questions = setup_teacher()
    student_answers = get_student_answers(questions)

    result, status = model.grade_answers(student_answers)
    if status == 200:
        display_result(result, student_answers)
    else:
        print(f"Error: {result['error']}")

if __name__ == "__main__":
    main()