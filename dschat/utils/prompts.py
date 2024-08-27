from typing import List


def generate_qa_base_prompt(question, document):
    prompt = f'{document}\nQ: {question}?\nA:'
    return prompt


def generate_qa_opinion_prompt(question, document):
    prompt = f'Bob said, \"{document}\"\nQ: {question} in Bob\'s opinion?\nA:'
    return prompt


def generate_qa_attributed_prompt(question, document):
    prompt = f'{document}\nQ: {question} based on the given text?\nA:'
    return prompt


def generate_qa_c_fisrt_instruction_prompt(question, document):
    prompt = f'Instruction: read the given information and answer the corresponding question. {document}\nQ: {question}?\nA:'
    return prompt


def generate_qa_instruction_prompt(question, document):
    prompt = f'Instruction: answer the question based on the given context.\nQ: {question}?\nContext: {document}\nA:'
    return prompt


def generate_qa_prompt(question: str, document: str, template: str):
    if template == "base":
        return generate_qa_base_prompt(question, document)
    elif template == "opinion-based":
        return generate_qa_opinion_prompt(question, document)
    elif template == "attributed":
        return generate_qa_attributed_prompt(question, document)
    elif template == "context-first-instruction":
        return generate_qa_c_fisrt_instruction_prompt(question, document)
    elif template == "instruction-based":
        return generate_qa_instruction_prompt(question, document)
    else:
        raise NotImplementedError(f"{template} prompt not implemented for question-answering task")


def generate_multi_choice_opinion_prompt(question: str, document: str, choices: List[str]):
    choice_str = "\n".join([f"{chr(ord('A') + i)}. {choice}" for i, choice in enumerate(choices)])
    prompt = f'According to what Bob said, choose the best choice that is in agreement with Bob from the following options.\n\nBob said: \"{document}\"\n\nQuestion: \n{question} in Bob\'s opinion?\n\nOptions:\n{choice_str}\n\nAnswer:'
    return prompt


def generate_multi_choice_instruction_prompt(question: str, document: str, choices: List[str]):
    choice_str = "\n".join([f"{chr(ord('A') + i)}. {choice}" for i, choice in enumerate(choices)])
    prompt = f'According to the given information, choose the best choice from the following options.\n\nInformation:\n{document}\n\nQuestion: \n{question}\n\nOptions:\n{choice_str}\n\nAnswer:'
    return prompt


def generate_multi_choice_prompt(question: str, document: str, choices: List[str], template: str):
    if template == "opinion-based":
        return generate_multi_choice_opinion_prompt(question, document, choices)
    elif template == "instruction-based":
        return generate_multi_choice_instruction_prompt(question, document, choices)
    else:
        raise NotImplementedError(f"{template} prompt not implemented for multiple-choice task")