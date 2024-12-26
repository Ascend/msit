# Copyright Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
import json
import re
import random
import torch
from tqdm import tqdm

from precision_tool.precision_tool import PrecisionTest
from msmodelslim.pytorch.llm_ptq.data_select.dataset.mmlu import MMLUDataset
from msmodelslim.pytorch.llm_ptq.data_select.dataset.gsm8k import Gsm8kDataset
from msmodelslim.pytorch.llm_ptq.data_select.dataset.ceval import CEvalDataset

batch_size = 8


class CalibDataSelect(object):
    def __init__(self, datasets, sample_size, shuffle_seed, tokenizer, split_gsm8k='', short_prompt_path_gsm8k='',
                 prompt_path_gsm8k='', model=None):
        self.datasets = datasets
        self.sample_size = sample_size
        self.shuffle_seed = shuffle_seed
        self.tokenizer = tokenizer
        self.model = model
        self.split_gsm8k = split_gsm8k
        self.short_prompt_path_gsm8k = short_prompt_path_gsm8k
        self.prompt_path_gsm8k = prompt_path_gsm8k

    def process(self):
        mixed_dataset = []
        for dataset in self.datasets:
            for dset, path in dataset.items():
                mixed_dataset.extend(self._get_mixed_dataset(dset, path, self.sample_size))

        if self.shuffle_seed:
            random.seed(self.shuffle_seed)
            random.shuffle(mixed_dataset)
            mixed_dataset = mixed_dataset[:self.sample_size]

        return mixed_dataset

    def _case_no_model(self, questions, labels):
        batch_dataset = []
        for idx, prpt in enumerate(questions):
            batch_dataset.append([{"prompt": prpt, "gt": labels[idx]}])
        return batch_dataset

    # gsm8k
    def _verify_answers_gsm8k(self, questions, answers, labels):
        correct_dataset = []

        def process_answers(questions, answers, split=False):
            outputs = []
            for i, answer in enumerate(answers):
                output = answer.strip()[len(questions[i]):]
                if split:
                    output = output.split('\n\n')[0]
                outputs.append(output)
            return outputs

        answers = process_answers(questions, answers, split=True)
        for answer, label in zip(answers, labels):
            response_number = re.findall(r'\d+', answer)
            if response_number is not None and len(response_number) > 0:
                last_number = response_number[-1]
            else:
                last_number = ''
            if label == last_number:
                correct_dataset.append([{"prompt": questions[i]}, {"gt": label}])

        return correct_dataset

    def _run_model_gsm8k(self, evaluate_data):
        calib_dataset = []

        dataloader = torch.utils.data.DataLoader(evaluate_data, batch_size)
        for _, batch in enumerate(tqdm(dataloader)):
            queries = [prompt.strip() for prompt in batch["prompt"]]
            labels = [label.strip() for label in batch["label"]]

            if self.model is None:
                calib_dataset.extend(self._case_no_model(queries, labels))
                continue

            inputs = self.tokenizer(queries, return_tensors="pt", padding=True).to("npu")
            outputs = self.model.generate(**inputs, max_new_tokens=256, eos_token_id=self.tokenizer.eos_token_id,
                                          do_sample=True)
            answers = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            calib_dataset.extend(self._verify_answers_gsm8k(queries, answers, labels))

        return calib_dataset

    def _get_mixed_dataset(self, dataset, path, sample_size):
        def build_prompt(title, text, passage):
            prompt = f"{title} -- {passage}\nQuestion:{text}?\nAnswer:"
            return prompt

        def get_queries_label(dataset, path):
            queries = []
            labels = []

            if dataset == 'boolq':
                datas = []
                with open(path, encoding="utf-8") as file:
                    for line in file:
                        data = json.loads(line)
                        datas.append(data)
                for ds in datas:
                    title = ds["title"]
                    text = ds["question"]
                    passage = ds["passage"]
                    label = ds["answer"]
                    queries.append(build_prompt(title, text, passage))
                    labels.append(label)
            elif dataset == 'ceval_5_shot':
                ceval_processor = CEvalDataset(path)
                qs, ls = ceval_processor.process_data(sample_size)
                queries.extend(qs)
                labels.extend(ls)
            elif dataset == 'mmlu':
                def _get_token_len(prompt):
                    inputs = self.tokenizer(prompt)
                    return len(inputs.input_ids)
                mmlu_processor = MMLUDataset(path, _get_token_len)
                qs, ls = mmlu_processor.process_data(sample_size)
                queries.extend(qs)
                labels.extend(ls)

            return queries, labels

        if dataset == "gsm8k":
            gsm8k_processor = Gsm8kDataset(path, self.split_gsm8k, self.short_prompt_path_gsm8k, self.prompt_path_gsm8k)
            evaluate_data = gsm8k_processor.process_data(sample_size)

            evaluate_data = self._run_model_gsm8k(evaluate_data)
            return evaluate_data

        else:
            if self.model is None:
                queries, labels = get_queries_label(dataset, path)
                return self._case_no_model(queries, labels)

            precision_test = PrecisionTest(self.model, self.tokenizer, dataset, batch_size if dataset == "ceval_5_shot" else 32, "npu")
            precision_test.mix_calib_dataset()
            precision_test.test()
            return precision_test.calib_dataset
