# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
import math
import random
import os
import json
from tqdm.auto import tqdm

import numpy as np
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, DynamicCache
from .omni_config import OmniAttentionConfig
from .omni_utils import patch_with_omni_attn_pattern


ORDINAL_NUMBERS = [
        "first",
        "second",
        "third",
        "fourth",
        "fifth",
        "sixth",
        "seventh",
        "eighth",
        "ninth",
        "tenth",
        "eleventh",
        "twelfth",
        "thirteenth",
        "fourteenth",
        "fifteenth",
        "sixteenth",
        "seventeenth",
        "eighteenth",
        "nineteenth",
        "twentieth"
    ]


class OmniAttentionGeneticSearcher:
    def __init__(self, config: OmniAttentionConfig):
        self.config = config
        self.stage = 1
        self.work_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(self.work_dir, 'data')
        self.out_dir = os.path.join(self.work_dir, 'output', self.config.model_name)
        os.makedirs(self.out_dir, exist_ok=True)

        # set random seed
        seed = config.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # determine shape of attention patterns
        model_config = AutoConfig.from_pretrained(config.model_path)
        self.num_layers = model_config.num_hidden_layers
        self.num_kv_heads = model_config.num_key_value_heads

        print(f"Loading model and tokenizer from path {config.model_path}.")
        self.load_model_and_tokenizer()
        self.tokenize_inputs()

    def load_model_and_tokenizer(self):
        """Loads the model and tokenizer from the specified model path.
        """
        if self.config.debug:
            self.model, self.tokenizer, self.device = None, None, None
            return
        path = self.config.model_path
        self.model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
            device_map="auto",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.device = self.model.device

    def tokenize_inputs(self):
        if self.config.debug:
            self.num_books, self.prompts, self.answers = None, None, None
            return
        with open(os.path.join(self.data_dir, 'data.json'), 'r', encoding='utf-8') as fi:
            data = json.load(fi)
        self.num_books = len(data)
        self.prompts: list[torch.LongTensor] = []
        self.answers: list[list[str]] = []
        for i in range(self.num_books):
            tokenized = self.tokenizer(data[i]['text'], add_special_tokens=False, return_tensors='pt').input_ids
            self.prompts.append(tokenized.to(self.device))
            self.answers.append(data[i]['answer'])

    def generate_gene_pool(self) -> list[np.ndarray]:
        """Generate a list of random binary matrices with 1s and 0s. All matrices are shaped
        `(num_layers, num_kv_heads)`. The ratio of 1s in each matrix is determined by `stage`.
        For example, if `stage`=1, the ratio of 1s would be 10%, i.e., the sparsity is 90%.
        Totally `pool_size` matrices will be generated.
        The random matrices are generated in such a way that the elements of each row are either
        all 0s or all 1s.
        """
        pool = []
        num_ones = int(self.num_layers * self.stage / 10)
        for _ in range(self.config.pool_size):
            mat = np.zeros((self.num_layers, 1), dtype=int)
            mat[np.random.permutation(self.num_layers)[:num_ones], 0] = 1
            mat = mat.repeat(self.num_kv_heads, axis=1)
            if all((mat != m).any() for m in pool):
                pool.append(mat)
        return pool

    @torch.no_grad()
    def score_one(self, pattern: np.ndarray) -> float:
        if self.config.debug:
            return random.randint(0, 100)

        score = 0
        patch_with_omni_attn_pattern(
            self.model,
            pattern,
            self.config.sink,
            self.config.recent,
        )

        pbar = tqdm(total=self.num_books * len(ORDINAL_NUMBERS), desc="Scoring current pattern...", leave=False, position=1)
        for book_id in range(self.num_books):
            prompt, answer = self.prompts[book_id], self.answers[book_id]

            for question_id, num in enumerate(ORDINAL_NUMBERS):
                question = f"\nBased on the content of the book, what is the {num} passkey to the vault?\nAnswer: The {num} passkey is:\n"
                tokenized = self.tokenizer(question, add_special_tokens=False, return_tensors='pt').input_ids.to(self.device)
                query = torch.cat([prompt, tokenized], dim=-1)

                cache = DynamicCache()
                out = self.model(query, use_cache=True, past_key_values=cache)
                generated_token = out.logits[:, -1, :].argmax(-1).unsqueeze(0)
                response = [generated_token.item()]
                cache = out.past_key_values

                for _ in range(3):
                    out = self.model(generated_token, use_cache=True, past_key_values=cache)
                    generated_token = out.logits[:, -1, :].argmax(-1).unsqueeze(0)
                    response.append(generated_token.item())
                    cache = out.past_key_values

                response = self.tokenizer.decode(response)
                if answer[question_id] in response:
                    score += 1
                pbar.update(1)
        pbar.close()
        return score

    def search(self):
        for stage in range(1, 10):
            self.stage = stage

            best_score, best = -100, None
            score_per_head = np.zeros((self.num_layers, self.num_kv_heads), dtype=float)
            occur_per_head = np.zeros((self.num_layers, self.num_kv_heads), dtype=float)

            pool = self.generate_gene_pool()
            for pattern in tqdm(pool, position=0, desc=f"Current genetic stage: {self.stage}. Scoring pool...", leave=False):
                score = self.score_one(pattern)
                score_per_head += pattern * score
                occur_per_head += pattern
                if score > best_score:
                    best_score = score
                    best = pattern

            score_per_head = score_per_head / occur_per_head
            mutations = self.mutation(score_per_head)
            for pattern in tqdm(mutations, position=0, desc=f"Current genetic stage: {self.stage}. Scoring mutation...", leave=False):
                score = self.score_one(pattern)
                if score > best_score:
                    best_score = score
                    best = pattern
            out_file = os.path.join(self.out_dir, f'genetic_rowwise_stage_{self.stage}.tsv')
            print(f"Saving best pattern to path {out_file}.")
            np.savetxt(out_file, 1 - best, delimiter='\t', fmt='%d')

    def mutation(self, score_per_head: np.ndarray) -> list[np.ndarray]:
        num_ones = int(self.num_layers * self.stage / 10)
        score_per_layer = score_per_head.sum(-1)
        rank_per_layer = score_per_layer.argsort().argsort()
        best_genes_mask = (rank_per_layer >= self.num_layers - num_ones).astype(int)
        ones_pos = np.nonzero(best_genes_mask)[0]
        zeros_pos = np.nonzero(1 - best_genes_mask)[0]

        mutations = []
        for _ in range(self.config.num_mutation):
            a = np.random.choice(ones_pos, size=1)
            b = np.random.choice(zeros_pos, size=1)
            mutated = best_genes_mask.copy()
            mutated[a] = 1 - mutated[a]
            mutated[b] = 1 - mutated[b]
            mutated = mutated[:, None].repeat(self.num_kv_heads, axis=1)
            mutations.append(mutated)

        return mutations