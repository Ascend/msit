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

EPSILON=5e-4


class OmniAttentionGeneticSearcher:
    def __init__(self, config: OmniAttentionConfig):
        """
        初始化 OmniAttentionGeneticSearcher 类。

        参数:
        config (OmniAttentionConfig): 包含模型配置和搜索参数的对象。
        """
        # 保存传入的配置对象
        self.config = config

        # 初始化搜索稀疏度为90%
        self.sparsity = 90

        # 获取当前脚本的工作目录
        self.work_dir = os.path.dirname(os.path.abspath(__file__))

        # 设置数据目录路径
        self.data_dir = os.path.join(self.work_dir, 'data')

        # 设置输出目录路径，基于模型名称
        self.out_dir = os.path.join(self.work_dir, 'output', self.config.model_name)

        # 创建输出目录（如果不存在）
        os.makedirs(self.out_dir, exist_ok=True)

        # 设置随机种子以确保实验的可重复性
        seed = config.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # 从预训练模型路径加载模型配置
        model_config = AutoConfig.from_pretrained(config.model_path)

        # 获取模型的隐藏层数量
        self.num_layers = model_config.num_hidden_layers

        # 获取模型的键值头数量
        self.num_kv_heads = model_config.num_key_value_heads

        # 打印加载模型和分词器的信息
        print(f"Loading model and tokenizer from path {config.model_path}.")

        # 加载模型和分词器
        self.load_model_and_tokenizer()

        # 对输入数据进行分词处理
        self.tokenize_inputs()

    @property
    def num_ones(self):
        return int(self.num_layers * (1 - self.sparsity/100))

    def load_model_and_tokenizer(self):
        """
        从指定的模型路径加载模型和分词器。

        如果处于调试模式（`self.config.debug`为True), 则将模型、分词器和设备设置为None并返回。
        否则，加载预训练模型和分词器，并设置设备为模型所在的设备。
        """
        if self.config.debug:
            # 调试模式下，不加载模型和分词器
            self.model, self.tokenizer, self.device = None, None, None
            return

        # 获取模型路径
        path = self.config.model_path

        # 加载预训练模型，设置参数：
        # - torch_dtype: 使用bfloat16数据类型
        # - low_cpu_mem_usage: 减少CPU内存占用
        # - attn_implementation: 使用eager模式实现注意力机制
        # - device_map: 自动分配设备
        self.model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
            device_map="auto",
        )

        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(path)

        # 获取模型所在的设备（如CPU或GPU）
        self.device = self.model.device

    def tokenize_inputs(self):
            """
            对输入数据进行分词处理。

            如果处于调试模式（`self.config.debug`为True），则将书籍数量、提示和答案设置为None并返回。
            否则，从数据文件中加载数据，对每本书的文本进行分词，并将结果存储到`self.prompts`和`self.answers`中。
            """
            if self.config.debug:
                # 调试模式下，不加载数据
                self.num_books, self.prompts, self.answers = None, None, None
                return

            # 打开数据文件并加载JSON数据
            with open(os.path.join(self.data_dir, 'data.json'), 'r', encoding='utf-8') as fi:
                data = json.load(fi)

            # 获取书籍数量
            self.num_books = len(data)

            # 初始化存储分词后提示和答案的列表
            self.prompts: list[torch.LongTensor] = []
            self.answers: list[list[str]] = []

            # 对每本书的文本进行分词处理
            for i in range(self.num_books):
                # 使用分词器对文本进行分词，返回PyTorch张量
                tokenized = self.tokenizer(data[i]['text'], add_special_tokens=False, return_tensors='pt').input_ids

                # 将分词结果移动到指定设备（如GPU）并存储
                self.prompts.append(tokenized.to(self.device))

                # 存储每本书的答案
                self.answers.append(data[i]['answer'])


    def generate_gene_pool(self) -> list[np.ndarray]:
        """
        生成一组随机的二进制矩阵（基因池）。

        每个矩阵的形状为`(num_layers, num_kv_heads)`，其中1的比例由当前阶段（`stage`）决定。
        例如，如果`stage`=1，则1的比例为10%（稀疏度为90%）。
        总共生成`pool_size`个矩阵。
        生成的随机矩阵确保每一行的元素要么全为0，要么全为1。

        返回:
        list[np.ndarray]: 生成的二进制矩阵列表。
        """
        pool = []

        # 计算每层中1的数量
        num_ones = self.num_ones

        # 生成指定数量的矩阵
        for _ in range(self.config.pool_size):
            # 初始化一个全0矩阵，形状为`(num_layers, 1)`
            mat = np.zeros((self.num_layers, 1), dtype=int)

            # 随机选择`num_ones`行，将其值设置为1
            mat[np.random.permutation(self.num_layers)[:num_ones], 0] = 1

            # 将矩阵沿列方向重复`num_kv_heads`次，扩展为`(num_layers, num_kv_heads)`形状
            mat = mat.repeat(self.num_kv_heads, axis=1)

            # 确保生成的矩阵与池中已有的矩阵不完全相同
            if all((mat != m).any() for m in pool):
                pool.append(mat)

        return pool

    @torch.no_grad()
    def score_one(self, pattern: np.ndarray) -> float:
        """
        对给定的注意力模式进行评分。

        参数:
        pattern (np.ndarray): 注意力模式矩阵，形状为 `(num_layers, num_kv_heads)`。

        返回:
        float: 当前模式的得分。
        """
        # 如果处于调试模式，则返回一个随机得分
        if self.config.debug:
            return random.randint(0, 100)

        # 初始化得分
        score = 0

        # 使用给定的注意力模式修补模型
        patch_with_omni_attn_pattern(
            self.model,
            pattern,
            self.config.sink,
            self.config.recent,
        )

        # 初始化进度条，显示评分进度
        pbar = tqdm(total=self.num_books * len(ORDINAL_NUMBERS), desc="Scoring current pattern...", leave=False, position=1)

        # 遍历每本书
        for book_id in range(self.num_books):
            # 获取当前书的提示和答案
            prompt, answer = self.prompts[book_id], self.answers[book_id]

            # 遍历每个问题（基于 ORDINAL_NUMBERS）
            for question_id, num in enumerate(ORDINAL_NUMBERS):
                # 构建问题文本
                question = f"\nBased on the content of the book, what is the {num} passkey to the vault?\nAnswer: The {num} passkey is:\n"

                # 对问题进行分词并转换为张量
                tokenized = self.tokenizer(question, add_special_tokens=False, return_tensors='pt').input_ids.to(self.device)

                # 将提示和问题拼接在一起作为输入
                query = torch.cat([prompt, tokenized], dim=-1)

                # 初始化动态缓存
                cache = DynamicCache()

                # 使用模型生成响应
                out = self.model(query, use_cache=True, past_key_values=cache)

                # 获取生成的第一个 token
                generated_token = out.logits[:, -1, :].argmax(-1).unsqueeze(0)
                response = [generated_token.item()]
                cache = out.past_key_values

                # 继续生成后续的 token（总共生成 4 个 token）
                for _ in range(3):
                    out = self.model(generated_token, use_cache=True, past_key_values=cache)
                    generated_token = out.logits[:, -1, :].argmax(-1).unsqueeze(0)
                    response.append(generated_token.item())
                    cache = out.past_key_values

                # 将生成的 token 解码为文本
                response = self.tokenizer.decode(response)

                # 如果生成的文本中包含正确答案，则增加得分
                if answer[question_id] in response:
                    score += 1

                # 更新进度条
                pbar.update(1)

        # 关闭进度条
        pbar.close()

        # 返回当前模式的总得分
        return score

    def search_incremental(self):
        """
        执行遗传搜索算法，寻找最佳的注意力模式。

        该方法通过多个阶段（stage）逐步优化注意力模式，每个阶段生成一组基因池并对其进行评分，
        最终选择最佳的模式并保存结果。
        """
        # 初始化每个注意力头的得分矩阵和出现次数矩阵
        score_per_head = np.zeros((self.num_layers, self.num_kv_heads), dtype=float) + EPSILON
        occur_per_head = np.zeros((self.num_layers, self.num_kv_heads), dtype=float) + EPSILON
        prev_best = None

        # 用于存储当前阶段的得分数组
        score_array_this_stage = None

        # 遍历每个阶段（从1到9）
        for stage in range(1, 10):
            # 设置当前稀疏度，从90到10
            self.sparsity = 100 - stage*10

            # 初始化最佳得分和最佳模式
            best_score, best = -100, None

            # 如果是第一阶段，生成初始基因池；否则，通过进化生成新的基因池
            if stage == 1:
                pool = self.generate_gene_pool()
            else:
                pool = self.evolution(score_array_this_stage, prev_best)

            # 对基因池中的每个模式进行评分
            for pattern in tqdm(pool, position=0, desc=f"Current search stage: {stage}", leave=False):
                # 对当前模式进行评分
                score = self.score_one(pattern)

                # 更新每个注意力头的总得分和出现次数
                score_per_head += pattern * score
                occur_per_head += pattern

                # 如果当前模式的得分更高，则更新最佳得分和最佳模式
                if score > best_score:
                    best_score = score
                    best = pattern

            # 计算当前阶段的平均得分数组
            score_array_this_stage = score_per_head / occur_per_head

            # 对当前阶段的得分数组进行变异操作，生成新的模式
            mutations = self.mutation(score_array_this_stage)

            # 对变异后的模式进行评分
            for pattern in tqdm(mutations, position=0, desc=f"Current search stage: {stage}. In mutation", leave=False):
                # 对当前模式进行评分
                score = self.score_one(pattern)

                # 更新每个注意力头的总得分和出现次数
                score_per_head += pattern * score
                occur_per_head += pattern

                # 如果当前模式的得分更高，则更新最佳得分和最佳模式
                if score > best_score:
                    best_score = score
                    best = pattern

            # 保存当前阶段的最佳模式到文件
            out_file = os.path.join(self.out_dir, f'genetic_rowwise_sparsity_{self.sparsity:d}_score_{best_score}.tsv')
            print(f"Saving best pattern to path {out_file}.")
            np.savetxt(out_file, 1 - best, delimiter='\t', fmt='%d')
            prev_best = best.copy()

    def search_on_this_sparsity(self, sparsity):
        """
        执行遗传搜索算法，寻找最佳的注意力模式。
        遗传算法通过进化生成模式池，对每个模式进行评分，并逐轮优化，直到得分不再提升为止。
        """
        # 固定稀疏度
        self.sparsity = sparsity

        # 初始化每个注意力头的得分矩阵和出现次数矩阵
        # score_per_head: 每层每个注意力头的累计得分
        # occur_per_head: 每层每个注意力头出现的次数，用于计算平均得分
        score_per_head = np.zeros((self.num_layers, self.num_kv_heads), dtype=float) + EPSILON
        occur_per_head = np.zeros((self.num_layers, self.num_kv_heads), dtype=float) + EPSILON

        # 初始化当前阶段的得分数组
        # score_array_this_stage: 保存当前阶段每层每个注意力头的平均得分
        score_array_this_stage = np.ones((self.num_layers, self.num_kv_heads), dtype=float)

        # 初始化当前轮次和上一轮次的最佳得分
        best_score_this_round = -10  # 当前轮次的最佳得分
        best_score_last_round = -100  # 上一轮次的最佳得分
        round = 0  # 记录遗传算法的轮数

        # 初始化历史搜索过的模式列表
        # historical_patterns_lis: 用于记录之前已经搜索过的模式，避免重复计算
        historical_patterns_lis = []

        # 初始化最佳模式
        # best_pattern: 保存当前的最佳模式
        best_pattern = np.zeros((self.num_layers, self.num_kv_heads), dtype=float)

        # 用于判断是否继续进化，如果本轮最佳得分不超过上一轮最佳得分，则停止搜索
        while best_score_this_round > best_score_last_round:
            # 更新上一轮的最佳得分
            best_score_last_round = best_score_this_round

            # 通过遗传算法生成新的模式池
            # evolution 函数会生成一组模式（基因池），并返回这些模式
            # 参数 A_with_random=True 表示生成的模式会包含一定的随机性
            # historical_patterns_lis 用于避免生成已存在的模式
            pool = self.evolution(
                score_array_this_stage,
                best_pattern,
                A_with_random=True,
                historical_patterns_lis=historical_patterns_lis
            )

            # 遍历基因池中的每个模式，对其进行评分
            for pattern in tqdm(pool, position=0, desc=f"Current round: {round}", leave=False):
                # 对当前模式进行评分，评分函数返回该模式的得分
                score = self.score_one(pattern)

                # 将当前模式添加到历史模式列表，避免重复计算
                historical_patterns_lis.append(pattern)

                # 更新每个注意力头的累计得分和出现次数
                # 当前模式得分会叠加到对应位置
                score_per_head += pattern * score
                occur_per_head += pattern

                # 如果当前模式的得分高于本轮最佳得分，则更新最佳得分和最佳模式
                if score > best_score_this_round:
                    best_score_this_round = score
                    best_pattern = pattern

            # 计算当前阶段的平均得分数组
            # 每个注意力头的平均得分 = 累计得分 / 出现次数
            score_array_this_stage = score_per_head / occur_per_head

            # 保存当前阶段的最佳模式到文件
            # 文件名包含稀疏度、轮次和最佳得分，用于记录当前阶段的结果
            out_file = os.path.join(
                self.out_dir,
                f'genetic_rowwise_sparsity_{self.sparsity:d}_round_{round}_score_{best_score_this_round}.tsv'
            )
            print(f"Saving best pattern to path {out_file}.")
            # 将最佳模式保存为文件，格式为 TSV，每个值为 0 或 1（1 表示稀疏位置）
            np.savetxt(out_file, 1 - best_pattern, delimiter='\t', fmt='%d')

            # 进入下一轮进化
            round += 1

    def mutation(self, score_per_head: np.ndarray) -> list[np.ndarray]:
        """
        对基因进行变异操作，生成一组新的基因, 用于基于得分的小型排序。

        参数:
        score_per_head (np.ndarray): 每个注意力头的得分矩阵，形状为 `(num_layers, num_kv_heads)`。

        返回:
        list[np.ndarray]: 变异后的基因矩阵列表，每个矩阵的形状为 `(num_layers, num_kv_heads)`。
        """
        # 计算每层中1的数量，基于当前阶段（`stage`）
        num_ones = self.num_ones

        # 对每层的得分求和，得到每层的总得分
        score_per_layer = score_per_head.sum(-1)

        # 对每层的得分进行排名，`argsort().argsort()` 用于获取排名
        rank_per_layer = score_per_layer.argsort().argsort()

        # 根据排名生成最佳基因掩码，选择排名最高的 `num_ones` 层
        best_genes_mask = (rank_per_layer >= self.num_layers - num_ones).astype(int)

        # 获取最佳基因掩码中1的位置（即被选中的层）
        ones_pos = np.nonzero(best_genes_mask)[0]

        # 获取最佳基因掩码中0的位置（即未被选中的层）
        zeros_pos = np.nonzero(1 - best_genes_mask)[0]

        # 初始化存储变异结果的列表, 并把根据排名生成最佳基因掩码加入基因池
        mutations = [best_genes_mask]

        # 生成指定数量的变异基因
        for _ in range(self.config.num_mutation):
            # 随机选择一个被选中的层和一个未被选中的层
            a = np.random.choice(ones_pos, size=1)
            b = np.random.choice(zeros_pos, size=1)

            # 复制最佳基因掩码并进行变异操作
            mutated = best_genes_mask.copy()
            mutated[a] = 1 - mutated[a]  # 将被选中的层从1变为0
            mutated[b] = 1 - mutated[b]  # 将未被选中的层从0变为1

            # 确保变异后的基因中1的数量与预期一致
            if mutated.sum() != num_ones:
                raise RuntimeError(f"The mutation {mutated} has wrong number of ones. Should be {num_ones}, but got {mutated.sum()} instead.")

            # 将变异后的基因添加到结果列表中
            mutations.append(mutated)

        # 将变异后的基因掩码扩展为 `(num_layers, num_kv_heads)` 形状
        for i in range(len(mutations)):
            mutations[i] = mutations[i][:, None].repeat(self.num_kv_heads, axis=1)
        return mutations

    def evolution(
            self,
            score_per_head: np.ndarray,
            best_pattern: np.ndarray,
            pool_size: int=-1,
            ab_rate: float=0.8,
            prob_bias_rate: float=144,
            A_with_random: bool=False,
            historical_patterns_lis: list[np.ndarray]=None,
        ) -> list[np.ndarray]:
        """
        生成新的模式池,支持检查是否与历史模式重复的版本。
        目前仅支持进化模式和同等stage里的继续进化, 以及退化(从更高级的stage退化)

        参数:
        - score_per_head: 每个头的得分,形状为 (dim1, dim2)。
        - best_pattern: 最佳模式,形状为 (dim1, dim2)。可以使用历史stage或当前stage的best_pattern。当使用当前stage的best_pattern来优化当前搜索成果时,建议将A_with_random设置为True以保留更多遗传信息。
        - pool_size: 需要搜索的pattern数量，不设置的话会使用 self.config.pool_size。
        - ab_rate: A部分的比例,默认为0.8。A部分更可能包含best_pattern,B部分更随机。当A_with_random为False时,A部分必包含best_pattern；当A_with_random为True时,A部分会赋予更高的稳定性。
        - prob_bias_rate: 概率偏置率,默认为6.0。
        - A_with_random: 是否在A部分添加随机性, 进化生成更多的1的时候默认为False, 其他情况建议设置为True。
        - historical_patterns_lis: 历史已搜索的模式列表,默认为None。

        返回:
        - 新的模式池,包含生成的模式。
        """
        # 计算每层的总1的数量
        total1s = self.num_ones

        # 定义维度
        dim1 = self.num_layers  # 层的数量
        dim2 = self.num_kv_heads  # 每个层的kv_heads数量
        dim22 = 1  # 用于reshape的辅助维度

        if historical_patterns_lis is None:
            historical_patterns_lis = []

        # 计算每个头的总得分
        ordermx = score_per_head.sum(-1).reshape(dim1, 1)

        # 计算最佳模式的每行总和
        best_pattern_fullrow = (np.sum(best_pattern.copy(), 1) / dim2 + 0.5).astype(int).reshape((dim1, dim22))

        # 计算最佳模式中需要保留的1的数量
        keep = int(np.sum(best_pattern.reshape(-1)) / dim2 + 0.5)

        # 计算A和B部分的数量,   pool_size is the expected amount of gene pools.
        if pool_size < 0 :
            pool_size = self.config.pool_size

        amountA = int(pool_size * ab_rate + 0.1)
        amountB = int(pool_size - amountA + 0.1)

        # 调整amountA和amountB,确保不超过组合数的限制
        amounthis = len(historical_patterns_lis)
        best_pattern_fullrow_1s = np.sum(best_pattern_fullrow)
        his_A = 0

        for i in range(amounthis):
            if np.sum(best_pattern_fullrow.reshape(-1) * (np.sum(historical_patterns_lis[i], 1) / dim2 + 0.5).astype(int).reshape(-1)) == best_pattern_fullrow_1s:
                his_A += 1

        if A_with_random:
            amountA = min(math.comb(dim1 * dim22, total1s) - amounthis, amountA)
        else:
            amountA = min(math.comb(dim1 * dim22 - keep, total1s - keep) - his_A, amountA)
        amountB = min(math.comb(dim1 * dim22, total1s) - amountA - amounthis, amountB)

        # 初始化原型矩阵,用于生成新的模式
        prototype_matrix = np.ones((dim1, dim2))

        # 计算A部分的采样概率
        minnA = np.min(ordermx[best_pattern_fullrow < 0.5])
        maxxA = np.max(ordermx[best_pattern_fullrow < 0.5])

        if minnA != maxxA:
            probA = ((1. - best_pattern_fullrow) * ordermx - minnA) / (maxxA - minnA) * (prob_bias_rate - 1) + 1.
        else:
            probA = np.ones((dim1, dim22))
        probA = probA * (1. - best_pattern_fullrow)
        probA = probA / np.sum(probA)

        # 初始化索引数组
        indxx = np.array(range(dim1 * dim22))
        newA = int(total1s - keep + 0.5)

        # 如果A_with_random为True,则调整probA并增加随机性
        if A_with_random:
            probA += best_pattern_fullrow * np.sum(probA) * 9.
            probA = probA / np.sum(probA)
            newA = total1s

        # 初始化输出列表
        output = []
        countA = 0

        # 生成A部分的模式
        while countA < amountA:
            if A_with_random:
                matrixA_new = np.zeros((dim1, dim22))
            else:
                matrixA_new = best_pattern_fullrow.copy()

            # 随机采样A部分的模式
            sampleA = np.random.choice(indxx, newA, replace=False, p=probA.reshape(-1))

            for index in sampleA:
                matrixA_new[index // dim22, index % dim22] = 1.

            # 检查生成的模式是否符合要求
            if int(np.sum(matrixA_new) + 0.5) != total1s:
                print("错误：总和不匹配！")
                continue

            matrixA_new = matrixA_new * prototype_matrix

            # 检查生成的模式是否已经存在于输出列表中
            if not any((matrixA_new == arr).all() for arr in output) and not any((matrixA_new == arr).all() for arr in historical_patterns_lis):
                output.append(matrixA_new)
                countA += 1

        # 计算B部分的采样概率
        minnB = np.min(ordermx.reshape(-1))
        maxxB = np.max(ordermx.reshape(-1))

        if minnB != maxxB:
            probB = (ordermx - minnB) / (maxxB - minnB) * (prob_bias_rate - 1) + 1.
        else:
            probB = np.ones((dim1, dim22))
        probB = probB / np.sum(probB)

        newB = total1s
        countB = 0

        # 生成B部分的模式
        while countB < amountB:
            matrixB_new = np.zeros((dim1, dim22))

            # 随机采样B部分的模式
            sampleB = np.random.choice(indxx, newB, replace=False, p=probB.reshape(-1))

            for index in sampleB:
                matrixB_new[index // dim22, index % dim22] = 1.

            # 检查生成的模式是否符合要求
            if int(np.sum(matrixB_new) + 0.5) != total1s:
                print("错误：总和不匹配！")
                continue

            matrixB_new = matrixB_new * prototype_matrix

            # 检查生成的模式是否已经存在于输出列表中
            if not any((matrixB_new == arr).all() for arr in output) and not any((matrixB_new == arr).all() for arr in historical_patterns_lis):
                output.append(matrixB_new)
                countB += 1

        # 检查输出数量是否匹配
        if len(output) != amountA + amountB:
            print("错误：输出数量不匹配！")

        return output