import os
import random
# DeepSpeed Team
import datasets
from datasets import load_dataset, load_from_disk, Dataset, concatenate_datasets
from torch.utils.data import Subset
import re
from dschat.utils.prompts import generate_qa_prompt, generate_multi_choice_prompt


datasets.disable_caching()

# The template prompt dataset class that all new dataset porting needs to
# follow in order to have a unified API and unified data format.
class PromptRawDataset(object):

    def __init__(self, output_path, seed, local_rank, dataset_name):
        self.output_path = output_path
        self.seed = seed
        self.local_rank = local_rank
        if os.path.exists(dataset_name):
            self.raw_datasets = load_from_disk(dataset_name)
        elif not dataset_name == 'local/jsonfile':
            self.raw_datasets = load_dataset(dataset_name)

    def get_train_data(self):
        return

    def get_eval_data(self):
        return

    # The prompt should be in the format of: " Human: " + actual_prompt_sentence + " Assistant:"
    def get_prompt(self, sample):
        return

    # The chosen response should be in the format of: " " + actual_response_sentence
    def get_chosen(self, sample):
        return

    # The rejected response should be in the format of: " " + actual_response_sentence
    # If the dataset does not have rejected response, return None
    def get_rejected(self, sample):
        return

    def get_prompt_and_chosen(self, sample):
        return

    def get_prompt_and_rejected(self, sample):
        return


class ConflictQADataset(PromptRawDataset):

    def __init__(self, output_path, seed, local_rank, dataset_path=None, task="multi_choice",
                 prompt_template=None, right_choice_order="random", add_none_choice=True, counterfactual=True):
        self.output_path = output_path
        self.seed = seed
        self.local_rank = local_rank
        self.dataset_name = f"osunlp/ConflictQA/dataset_path"
        self.dataset_name_clean = self.dataset_name.replace("/", "_").replace("-", "_")
        self.task = task
        self.prompt_template = prompt_template
        self.right_choice_order = right_choice_order
        self.add_none_choice = add_none_choice
        self.counterfactual = counterfactual

        if task == "multi_choice":
            if self.prompt_template is None:
                self.prompt_template = "instruction-based"

            if seed is not None:
                random.seed(seed)

            def conflictqa_multi_choice_map_function(x):   
                question = x["question"]

                if counterfactual:
                    document, right_choice, wrong_choice = x["counter_memory_aligned_evidence"], x["counter_answer"], x["memory_answer"]
                else:
                    document, right_choice, wrong_choice = x["parametric_memory_aligned_evidence"], x["memory_answer"], x["counter_answer"]

                if self.right_choice_order == "random":
                    right_choice_order = random.randint(0, 1)
                elif self.right_choice_order == 1 or self.right_choice_order == 0:
                    right_choice_order = self.right_choice_order
                else:
                    raise ValueError(f"right_choice_order {right_choice_order} is not supported")

                if right_choice_order == 0:
                    choices = [right_choice, wrong_choice]
                    right_answer, wrong_answer = "A", "B"
                elif right_choice_order == 1:
                    choices = [wrong_choice, right_choice]
                    right_answer, wrong_answer = "B", "A"
                if add_none_choice:
                    choices.append("None of the above.")

                prompt = generate_multi_choice_prompt(question, document, choices, self.prompt_template)
                return {"text": prompt, "right_answer": right_answer, "wrong_answer": wrong_answer}

            dataset = load_dataset("osunlp/ConflictQA", name=dataset_path)["test"]
            dataset = dataset.map(conflictqa_multi_choice_map_function, remove_columns=dataset.column_names)
            self.eval_dataset = dataset

        else:
            raise NotImplementedError(f"MRQANaturalQuestionsDataset for task {task} is not implemented")

    def get_train_data(self):
        raise FileNotFoundError("ConflictQA does not have a train dataset")

    def get_eval_data(self):
        return self.eval_dataset

    def get_prompt(self, sample):
        return sample['text']

    def get_right_answer(self, sample):
        return sample['right_answer']

    def get_wrong_answer(self, sample):
        return sample['wrong_answer']

    def get_chosen(self, sample):
        return sample['right_answer']

    def get_rejected(self, sample):
        return sample['wrong_answer']

    def get_prompt_and_chosen(self, sample):
        return self.get_prompt(sample) + " " + self.get_chosen(sample)

    def get_prompt_and_rejected(self, sample):
        return self.get_prompt(sample) + " " + self.get_rejected(sample)


class MRQANaturalQuestionsDataset(PromptRawDataset):

    def __init__(self, output_path, seed, local_rank, dataset_path=None, task="question_answering",
                 prompt_template=None, right_choice_order="random", add_none_choice=True):
        self.output_path = output_path
        self.seed = seed
        self.local_rank = local_rank
        self.dataset_name = "MRQANaturalQuestions"
        self.dataset_name_clean = "MRQANaturalQuestions"
        self.task = task
        self.prompt_template = prompt_template
        self.right_choice_order = right_choice_order
        self.add_none_choice = add_none_choice

        if "SPLIT" not in dataset_path:
            raise ValueError("dataset_path should contain SPLIT as a placeholder")

        if task == "question_answering":
            if self.prompt_template is None:
                self.prompt_template = "instruction-based"

            def nq_question_answering_map_function1(x):
                sample = {
                    "text": generate_qa_prompt(x["question"], x["substituted_context"], self.prompt_template),
                    "right_answer": " OR ".join(x["substituted_answers"]),
                    "wrong_answer": " OR ".join(x["original_answers"])
                }
                return sample

            def nq_question_answering_map_function2(x):
                sample = {
                    "text": generate_qa_prompt(x["question"], x["original_context"], self.prompt_template),
                    "right_answer": " OR ".join(x["original_answers"]),
                    "wrong_answer": " OR ".join(x["substituted_answers"])
                }
                return sample

            dataset = load_dataset("json", data_files=dataset_path.replace("SPLIT", "Train"))['train']
            dataset1 = dataset.map(nq_question_answering_map_function1, remove_columns=dataset.column_names)
            dataset2 = dataset.map(nq_question_answering_map_function2, remove_columns=dataset.column_names)
            dataset_expanded = concatenate_datasets([dataset1, dataset2])
            self.train_dataset = dataset_expanded

            dataset = load_dataset("json", data_files=dataset_path.replace("SPLIT", "Dev"))['train']
            dataset = dataset.map(nq_question_answering_map_function1, remove_columns=dataset.column_names)
            self.eval_dataset = dataset

        elif task == "multi_choice":
            if self.prompt_template is None:
                self.prompt_template = "instruction-based"

            if seed is not None:
                random.seed(seed)

            def nq_multi_choice_map_function1(x):
                question, document = x["question"], x["substituted_context"]
                right_choice, wrong_choice = " OR ".join(x["substituted_answers"]), " OR ".join(x["original_answers"])

                if self.right_choice_order == "random":
                    right_choice_order = random.randint(0, 1)
                elif self.right_choice_order == 1 or self.right_choice_order == 0:
                    right_choice_order = self.right_choice_order
                else:
                    raise ValueError(f"right_choice_order {right_choice_order} is not supported")

                if right_choice_order == 0:
                    choices = [right_choice, wrong_choice]
                    right_answer, wrong_answer = "A", "B"
                elif right_choice_order == 1:
                    choices = [wrong_choice, right_choice]
                    right_answer, wrong_answer = "B", "A"
                if self.add_none_choice:
                    choices.append("None of the above.")

                prompt = generate_multi_choice_prompt(question, document, choices, self.prompt_template)
                return {"text": prompt, "right_answer": right_answer, "wrong_answer": wrong_answer}

            def nq_multi_choice_map_function2(x):
                question, document = x["question"], x["original_context"]
                right_choice, wrong_choice = " OR ".join(x["original_answers"]), " OR ".join(x["substituted_answers"])

                if self.right_choice_order == "random":
                    right_choice_order = random.randint(0, 1)
                elif self.right_choice_order == 1 or self.right_choice_order == 0:
                    right_choice_order = self.right_choice_order
                else:
                    raise ValueError(f"right_choice_order {right_choice_order} is not supported")

                if right_choice_order == 0:
                    choices = [right_choice, wrong_choice]
                    right_answer, wrong_answer = "A", "B"
                elif right_choice_order == 1:
                    choices = [wrong_choice, right_choice]
                    right_answer, wrong_answer = "B", "A"
                if self.add_none_choice:
                    choices.append("None of the above.")

                prompt = generate_multi_choice_prompt(question, document, choices, self.prompt_template)
                return {"text": prompt, "right_answer": right_answer, "wrong_answer": wrong_answer}

            dataset = load_dataset("json", data_files=dataset_path.replace("SPLIT", "Train"))['train']
            dataset1 = dataset.map(nq_multi_choice_map_function1, remove_columns=dataset.column_names)
            dataset2 = dataset.map(nq_multi_choice_map_function2, remove_columns=dataset.column_names)
            dataset_expanded = concatenate_datasets([dataset1, dataset2])
            self.train_dataset = dataset_expanded

            dataset = load_dataset("json", data_files=dataset_path.replace("SPLIT", "Dev"))['train']
            dataset = dataset.map(nq_multi_choice_map_function1, remove_columns=dataset.column_names)
            self.eval_dataset = dataset

        else:
            raise NotImplementedError(f"MRQANaturalQuestionsDataset for task {task} is not implemented")

    def get_train_data(self):
        return self.train_dataset

    def get_eval_data(self):
        return self.eval_dataset

    def get_prompt(self, sample):
        return sample['text']

    def get_right_answer(self, sample):
        return sample['right_answer']

    def get_wrong_answer(self, sample):
        return sample['wrong_answer']

    def get_chosen(self, sample):
        return sample['right_answer']

    def get_rejected(self, sample):
        return sample['wrong_answer']

    def get_prompt_and_chosen(self, sample):
        return self.get_prompt(sample) + " " + self.get_chosen(sample)

    def get_prompt_and_rejected(self, sample):
        return self.get_prompt(sample) + " " + self.get_rejected(sample)
