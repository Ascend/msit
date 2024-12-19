# Copyright Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

import os
import json
import pandas as pd
from tqdm import tqdm
from security import json_safe_load, json_safe_dump, get_valid_path, get_valid_write_path, get_valid_read_path


class CEvalDataset():
    def __init__(self, dataset_path):
        self.choices = ["A", "B", "C", "D"]
        self.dataset_path = dataset_path
        self.dev_data = {}
        self.val_data = {}
        # load raw data
        subject_mapping = self.get_subject_mapping()
        for task_name in tqdm(subject_mapping):
            dev, val = self.load_csv_by_task_name(task_name, dataset_path)
            self.dev_data[task_name] = dev
            self.val_data[task_name] = val

    def get_subject_mapping(self):
        subject_mapping_path = os.path.join(self.dataset_path, "subject_mapping.json")
        subject_mapping_path = get_valid_read_path(subject_mapping_path)
        with open(subject_mapping_path) as f:
            subject_mapping = json.load(f)
        return subject_mapping

    def load_csv_by_task_name(self, task_name, dataset_path):
        shot = 5
        dev_file_path = get_valid_path(os.path.join(dataset_path, "dev", task_name + "_dev.csv"))
        dev_df = pd.read_csv(dev_file_path, header=None)[:shot + 1]
        val_file_path = get_valid_path(os.path.join(dataset_path, "val", task_name + "_val.csv"))
        val_df = pd.read_csv(val_file_path, header=None)

        dev_df = dev_df.iloc[1:, 1:]
        val_df = val_df.iloc[1:, 1:]
        return dev_df, val_df

    def format_example(self, df, idx, include_answer=True):
        # question,A,B,C,D,answer
        prompt = df.iloc[idx, 0]
        k = len(self.choices)
        for j in range(k):
            prompt += "\n{}. {}".format(self.choices[j], df.iloc[idx, j + 1])
        prompt += "\nAnswer:"
        if include_answer:
            prompt += " {}\n\n".format(df.iloc[idx, k + 1])
        return prompt

    def format_subject(self, subject):
        l = subject.split("_")
        s = ""
        for entry in l:
            s += " " + entry
        return s

    def gen_prompt(self, train_df, subject, k=-1):
        prompt = "以下是中国关于{}考试的单项选择题，请选出正确答案.\n\n".format(self.format_subject(subject))
        if k == -1:
            k = train_df.shape[0]
        for i in range(k):
            prompt += self.format_example(train_df, i)
        return prompt

    def process_data(self):
        queries = []
        labels = []

        for subject, val_df in self.val_data.items():
            for i in range(val_df.shape[0]):
                # question format
                prompt_end = self.format_example(val_df, i, include_answer=False)
                train_prompt = self.gen_prompt(self.dev_data[subject], subject, 5)
                prompt = train_prompt + prompt_end
                label = val_df.iloc[i, len(self.choices) + 1]
                gt_index = 'ABCD'.index(label)
                # generate prompt candidate
                queries.append(prompt)
                labels.append(label)

        return queries, labels
