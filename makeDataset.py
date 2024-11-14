from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import torch
import json


class QADataSet(Dataset):

    def __init__(self, json_path: str, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.question_data = []
        self.answer_explanation_data = []
        with open(json_path, "r", encoding='utf-8') as f:
            for line in f:
                if not line or line == "":
                    continue
                json_line = json.loads(line)
                question = json_line["Question"]
                correct_option = next(
                    (option["value"] for option in json_line["Options"] if option["key"] == json_line["Answer"]), "")
                explanation = json_line["Explanation"]
                answer_explanation = f"{correct_option}。{explanation}"
                self.question_data.append(question)
                self.answer_explanation_data.append(answer_explanation)
        print("Data load complete, size:", len(self.question_data))

    def __len__(self):
        return len(self.question_data)

    def __getitem__(self, index):
        source_text = str(self.question_data[index])
        target_text = str(self.answer_explanation_data[index])

        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.max_length,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.max_length,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze()
        target_mask = target["attention_mask"].squeeze()

        return {
            "source_ids": source_ids.to(dtype=torch.long),
            "source_mask": source_mask.to(dtype=torch.long),
            "target_ids": target_ids.to(dtype=torch.long),
            "target_mask": target_mask.to(dtype=torch.long)
        }


class BaichuanQADataset(Dataset):
    def __init__(self, json_path: str, tokenizer, max_length=256, char_limit=300):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        with open(json_path, "r", encoding='utf-8') as f:
            for line in f:
                if not line or line == "":
                    continue
                json_line = json.loads(line)

                question = json_line["Question"]

                # 生成选项内容
                options_text = "\n".join([f"{option['key']}: {option['value']}" for option in json_line["Options"]])

                # 获取正确答案的选项值
                correct_option = json_line["Answer"]
                correct_answer = next(
                    (option["value"] for option in json_line["Options"] if option["key"] == json_line["Answer"]), "")

                explanation = json_line["Explanation"]
                if explanation and len(explanation) > char_limit:
                    continue
                else :
                    # 按照要求格式化文本
                    full_text = f"对于后续问题，你需按照以下格式，向分析和正确选项字段填入答案。\n格式：\n问题：问题内容\n分析：{{填入分析，50字以内}}\n正确选项：{{填入正确的选项和选项文本}}\n\n问题：{question}\n{options_text}\n分析：{explanation}\n正确选项：{correct_option}.{correct_answer}{tokenizer.eos_token}"

                    self.data.append(full_text)

        print("Data load complete, size:", len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = str(self.data[index])
        inputs = self.tokenizer(
            text, max_length=self.max_length, truncation=True, padding="max_length", return_tensors="pt"
        )
        input_ids = inputs["input_ids"].squeeze()
        attention_mask = inputs["attention_mask"].squeeze()

        return {
            "input_ids": input_ids.to(dtype=torch.long),
            "attention_mask": attention_mask.to(dtype=torch.long)
        }


def custom_collate_fn(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    correct_answers = [item["correct_answer"] for item in batch]  # 保留为字符串列表

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "correct_answer": correct_answers
    }


class BaichuanQATestDataset(Dataset):
    def __init__(self, json_path: str, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        self.correct_answer_data = []  # 用于存储正确答案

        with open(json_path, "r", encoding='utf-8') as f:
            for line in f:
                if not line or line == "":
                    continue
                json_line = json.loads(line)

                question = json_line["Question"]

                # 生成选项内容
                options_text = "\n".join([f"{option['key']}: {option['value']}" for option in json_line["Options"]])

                # 获取正确答案的选项值
                correct_option = json_line["Answer"]
                correct_answer = next(
                    (option["value"] for option in json_line["Options"] if option["key"] == json_line["Answer"]), "")

                explanation = json_line["Explanation"]

                # 按照要求格式化文本
                full_text = f"问题：{question}\n{options_text}\n分析：{explanation}\n正确选项：{correct_option}.{correct_answer}"

                # 按照要求格式化文本，不包含答案和解释
                question_text = f"对于后续问题，你需按照以下格式，向分析和正确选项字段填入答案。\n格式：\n问题：问题内容\n分析：{{填入分析，50字以内}}\n正确选项：{{填入正确的选项和选项文本}}\n\n问题：{question}\n{options_text}\n分析："

                # 存储问题+选项文本
                self.data.append(question_text)

                # 存储正确答案
                self.correct_answer_data.append(full_text)

        print("Test data load complete, size:", len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = str(self.data[index])
        correct_answer = str(self.correct_answer_data[index])

        # Tokenize the text with just the question and options
        inputs = self.tokenizer(
            text, max_length=self.max_length, truncation=True, padding="max_length", return_tensors="pt"
        )

        input_ids = inputs["input_ids"].squeeze()
        attention_mask = inputs["attention_mask"].squeeze()

        return {
            "input_ids": input_ids.to(dtype=torch.long),
            "attention_mask": attention_mask.to(dtype=torch.long),
            "correct_answer": correct_answer  # 返回正确答案文本
        }

class HuatuoQADataset(Dataset):
    def __init__(self, json_path: str, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        with open(json_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():  # 跳过空行
                    continue
                json_line = json.loads(line)

                # 提取问题和答案
                question = json_line["questions"][0]  # 取出第一个问题
                answer = json_line["answers"][0]      # 取出第一个答案

                # 将问题和答案格式化成完整的文本，符合之前的数据格式
                full_text = f"问题：{question}\n回答：{answer}"

                self.data.append(full_text)

        print("Data load complete, size:", len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = str(self.data[index])
        inputs = self.tokenizer(
            text, max_length=self.max_length, truncation=True, padding="max_length", return_tensors="pt"
        )
        input_ids = inputs["input_ids"].squeeze()
        attention_mask = inputs["attention_mask"].squeeze()

        return {
            "input_ids": input_ids.to(dtype=torch.long),
            "attention_mask": attention_mask.to(dtype=torch.long)
        }

if __name__ == "__main__":
    #测试数据分类有效性
    tokenizer = AutoTokenizer.from_pretrained("Baichuan2-7B-Base", use_fast=False, trust_remote_code=True, add_eos_token=True)
    tokenizer.padding_side = "left"  # 设置填充在左侧
    # 加载测试集
    test_dataset = BaichuanQATestDataset("fzkuji/test.json", tokenizer, 256)

    # 随机选择 1% 的测试数据
    test_size = int(1 * len(test_dataset))
    test_indices = random.sample(range(len(test_dataset)), test_size)
    test_dataset = Subset(test_dataset, test_indices)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                                 collate_fn=custom_collate_fn)  # 使用自定义的 collate_fn
    with tqdm(test_dataloader, unit="batch") as tepoch:
        for batch in tepoch:
            print(test_dataloader)