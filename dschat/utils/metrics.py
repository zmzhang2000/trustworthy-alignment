import string
from typing import List, Union

import regex

EPSILON = 1e-6


def normalize_answer(s: str) -> str:
    """Normalization from the SQuAD evaluation script.

    See https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
    """

    def remove_articles(text):
        return regex.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def best_subspan_em(prediction: str, ground_truths: List[str]) -> float:
    normalized_prediction = normalize_answer(prediction)

    for ground_truth in ground_truths:
        normalized_ground_truth = normalize_answer(ground_truth)
        if normalized_ground_truth.lower() in normalized_prediction.lower():
            return 1.0
    return 0.0


def multi_choice_hit(prediction: str, ground_truth_option: str) -> bool:
    prediction_words = prediction.strip().split()
    prediction_words = [x.strip(" .:()[]") for x in prediction_words]
    hit = False
    for word in prediction_words:
        if word == ground_truth_option:
            hit = True
            break
    return hit


def memorization_ratio(samples: Union[dict, List[dict]], task: str = "question_answering"):
    if not isinstance(samples, list):
        samples = [samples]

    has_new_answers, has_old_answers = [], []
    for sample in samples:
        sample["response"] = "" if sample["response"] is None else sample["response"]
        if task == "question_answering":
            responses = sample["response"].strip().split(" OR ")
            new_ground_truths = sample["right_answer"].split(" OR ")
            has_new_answers.append(bool(max([best_subspan_em(prediction=response, ground_truths=new_ground_truths) for response in responses])))
            old_ground_truths = sample["wrong_answer"].split(" OR ")
            has_old_answers.append(bool(max([best_subspan_em(prediction=response, ground_truths=old_ground_truths) for response in responses])))
        elif task == "multi_choice":
            response = sample["response"].strip()
            has_new_answers.append(multi_choice_hit(response, sample["right_answer"]))
            has_old_answers.append(multi_choice_hit(response, sample["wrong_answer"]))
        else:
            raise NotImplementedError(f"memorization ratio for task {task} is not implemented")

    has_both_answers = [has_new_answer and has_old_answer for has_new_answer, has_old_answer in zip(has_new_answers, has_old_answers)]
    has_neither_answers = [not has_new_answer and not has_old_answer for has_new_answer, has_old_answer in zip(has_new_answers, has_old_answers)]
    has_only_new_answers = [has_new_answer and not has_old_answer for has_new_answer, has_old_answer in zip(has_new_answers, has_old_answers)]
    has_only_old_answers = [not has_new_answer and has_old_answer for has_new_answer, has_old_answer in zip(has_new_answers, has_old_answers)]

    has_new_answer_ratio = sum(has_new_answers) / (len(has_new_answers) + EPSILON)
    has_old_answer_ratio = sum(has_old_answers) / (len(has_old_answers) + EPSILON)
    has_both_answer_ratio = sum(has_both_answers) / (len(has_both_answers) + EPSILON)
    has_neither_answer_ratio = sum(has_neither_answers) / (len(has_neither_answers) + EPSILON)
    has_only_new_answer_ratio = sum(has_only_new_answers) / (len(has_only_new_answers) + EPSILON)
    has_only_old_answer_ratio = sum(has_only_old_answers) / (len(has_only_old_answers) + EPSILON)
    memorization_ratio = has_old_answer_ratio / (has_only_old_answer_ratio + has_only_new_answer_ratio + has_both_answer_ratio + EPSILON)
    
    metrics = {
        "memorization_ratio": memorization_ratio,
        "has_new_answer_ratio": has_new_answer_ratio,
        "has_old_answer_ratio": has_old_answer_ratio,
        "has_both_answer_ratio": has_both_answer_ratio,
        "has_neither_answer_ratio": has_neither_answer_ratio,
        "has_only_new_answer_ratio": has_only_new_answer_ratio,
        "has_only_old_answer_ratio": has_only_old_answer_ratio,
    }
    return metrics
