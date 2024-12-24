# Copyright Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
from datasets import load_dataset

from msmodelslim import logger as msmodelslim_logger


class Gsm8kDataset:
    logger = msmodelslim_logger

    def __init__(self, path, split, short_prompt_path, prompt_path, use_prompt='direct'):
        evaluate_data = load_dataset(path, data_files={"test": split})
        self.raw_data = evaluate_data['test']

        self.logger.info(f"Start loading dataset: GSM8K")
        if use_prompt == 'direct':
            self.cot_prompt = ''
        elif use_prompt == 'short':
            with open(short_prompt_path, 'r') as f:
                self.cot_prompt = f.read()
        else:
            with open(prompt_path, 'r') as f:
                self.cot_prompt = f.read()

    def get_cot_prompt(self, question, answer):
        if self.cot_prompt == '':
            question = 'Question: ' + question + '\n'
        else:
            question = self.cot_prompt + '\nQuestion: ' + question + '\n'
        answer = answer.split('####')[1].strip()

        return {"prompt": question, "label": answer}

    def process_data(self, sample_size):
        data_with_label = []

        for item in self.raw_data:
            question = item["question"]
            answer = item["answer"]
            data_with_label.append(self.get_cot_prompt(question, answer))

        return data_with_label
