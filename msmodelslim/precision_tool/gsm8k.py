# Copyright Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
from datasets import load_dataset

class Gsm8kDataset():
    def __init__(self, path, split, short_prompt_path, prompt_path, use_prompt='direct'):
        evaluate_data = load_dataset(path, data_files={"test": split})
        self.raw_data = evaluate_data['test']

        if use_prompt == 'direct':
            self.cot_prompt = ''
        elif use_prompt == 'short':
            try:
                self.cot_prompt = open(short_prompt_path, 'r').read()
            except:
                self.cot_prompt = "Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today? \n \
                Let's think step by step \n \
                There are 15 trees originally. \n \
                Then there were 21 trees after some more were planted. \n \
                So there must have been 21 - 15 = 6. \n \
                The answer is 6."
        else:
            try:
                self.cot_prompt = open(prompt_path, 'r').read()
            except:
                self.cot_prompt = "Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today? \n \
                Let's think step by step \n \
                There are 15 trees originally. \n \
                Then there were 21 trees after some more were planted. \n \
                So there must have been 21 - 15 = 6. \n \
                The answer is 6. \n \
                \n \
                Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot? \n \
                Let's think step by step \n \
                There are originally 3 cars. \n \
                2 more cars arrive. \n \
                3 + 2 = 5. \n \
                The answer is 5. \n \
                \n \
                Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total? \n \
                Let's think step by step \n \
                Originally, Leah had 32 chocolates. \n \
                Her sister had 42. \n \
                So in total they had 32 + 42 = 74. \n \
                After eating 35, they had 74 - 35 = 39. \n \
                The answer is 39. \n \
                \n \
                Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny? \n \
                Let's think step by step \n \
                Jason started with 20 lollipops. \n \
                Then he had 12 after giving some to Denny. \n \
                So he gave Denny 20 - 12 = 8. \n \
                The answer is 8. \n \
                \n \
                Question: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now? \n \
                Let's think step by step \n \
                Shawn started with 5 toys. \n \
                If he got 2 toys each from his mom and dad, then that is 4 more toys. \n \
                5 + 4 = 9. \n \
                The answer is 9. \n \
                \n \
                Question: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room? \n \
                Let's think step by step \n \
                There were originally 9 computers. \n \
                For each of 4 days, 5 more computers were added. \n \
                So 5 * 4 = 20 computers were added. \n \
                9 + 20 is 29. \n \
                The answer is 29. \n \
                \n \
                Question: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday? \n \
                Let's think step by step \n \
                Michael started with 58 golf balls. \n \
                After losing 23 on tues- day, he had 58 - 23 = 35. \n \
                After losing 2 more, he had 35 - 2 = 33 golf balls. \n \
                The answer is 33. \n \
                \n \
                Question: Olivia has $23. She bought five bagels for $3 each. How much money does she have left? \n \
                Let's think step by step \n \
                Olivia had 23 dollars. \n \
                5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. \n \
                So she has 23 - 15 dollars left. \n \
                23 - 15 is 8. \n \
                The answer is 8."

    def get_cot_prompt(self, question, answer):
        if self.cot_prompt == '':
            question = 'Question: ' + question + '\n'
        else:
            question = self.cot_prompt + '\nQuestion: ' + question + '\n'
        answer = answer.split('####')[1].strip()

        return {"prompt": question, "label": answer}

    def process_data(self):
        data_with_label = []

        for item in self.raw_data:
            question = item["question"]
            answer = item["answer"]
            data_with_label.append(self.get_cot_prompt(question, answer))

        return data_with_label