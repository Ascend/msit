import os
import random

import pandas as pd
from tqdm import tqdm

from ascend_utils.common.security import get_valid_read_path
from msmodelslim import logger as msmodelslim_logger


class MMLUDataset:
    logger = msmodelslim_logger

    def __init__(self, config, get_token_len):
        self.config = config
        self.get_token_len = get_token_len
        self.choices = ('A', 'B', 'C', 'D')

        # load row data
        self.logger.info(f"Start loading dataset: MMLU 5-shot")

        self.test_data = self.load_csv_folder(self.config['test_split'], end_format='_test.csv')
        self.dev_data = self.load_csv_folder(self.config['dev_split'], end_format='_dev.csv')

        self.logger.info(f'loading test dataset (num_subjects: {str(len(self.test_data))}) '
                         f'from {str(self.config["test_split"])}')
        self.logger.info(f'loading dev dataset (num_subjects: {str(len(self.dev_data))}) '
                         f'from {str(self.config["dev_split"])}')

    @staticmethod
    def format_subject(subject):
        l = subject.split('_')
        s = ""
        for entry in l:
            s += " " + entry
        return s

    def load_csv_folder(self, path, end_format='_test.csv'):
        data_container = {}
        for f in os.listdir(path):
            if f.endswith(end_format):
                file_path = get_valid_read_path(os.path.join(path, f))
                if 'dev' in end_format:
                    df = pd.read_csv(file_path, header=None)[:int(self.config['few_shot_examples'])]
                else:
                    df = pd.read_csv(file_path, header=None)
                subject_name = f.split(end_format)[0]
                data_container[subject_name] = df
        return data_container

    def format_example(self, df, idx, include_answer=True):
        prompt = df.iloc[idx, 0]
        k = df.shape[1] - 2
        for j in range(k):
            prompt += '\n{}. {}'.format(self.choices[j], df.iloc[idx, j + 1])
        prompt += '\nAnswer:'
        if include_answer:
            prompt += ' {}\n\n'.format(df.iloc[idx, k + 1])
        return prompt

    def gen_prompt(self, train_df, subject, k=-1):
        prompt = ('The following are multiple choice questions (with answers) '
                  'about {}.\n\n').format(self.format_subject(subject))
        if k == -1:
            k = train_df.shape[0]
        for i in range(k):
            prompt += self.format_example(train_df, i)
        return prompt

    def process_data(self, sample_size):
        queries = []
        labels = []

        for subject, test_df in tqdm(random.choices(list(self.test_data.items()),
                                                    k=min(sample_size//100+1,
                                                          len(self.test_data)))):
            for i in tqdm(range(test_df.shape[0]), leave=False, position=1):
                # question format
                prompt_end = self.format_example(test_df, i, include_answer=False)
                # few shot format
                k = int(self.config['few_shot_examples'])
                train_prompt = self.gen_prompt(self.dev_data[subject], subject, k)
                prompt = train_prompt + prompt_end
                while k > 0 and self.get_token_len(prompt) > self.config['max_length']:
                    k -= 1
                    train_prompt = self.gen_prompt(self.dev_data[subject], subject, k)
                    prompt = train_prompt + prompt_end
                label = test_df.iloc[i, test_df.shape[1] - 1]
                queries.append(prompt)
                labels.append(label)
        return queries, labels
