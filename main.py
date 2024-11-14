# -*- coding: utf-8 -*-
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, DataCollatorForLanguageModeling
from dataclasses import dataclass, field
from peft import TaskType, get_peft_model
from peft import AdaLoraConfig
from peft import PeftModel
from makeDataset import BaichuanQADataset, BaichuanQATestDataset, custom_collate_fn, HuatuoQADataset
from accelerate import Accelerator
from tqdm import tqdm
import random
from peft import prepare_model_for_kbit_training


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="GLM4-9B", metadata={"help": "Path to the model."})


@dataclass
class DataArguments:
    train_data_path: str = field(default="./fzkuji/train.json", metadata={"help": "Path to the training data."})
    val_data_path: str = field(default="./fzkuji/valid.json", metadata={"help": "Path to the validation data."})
    test_data_path: str = field(default="./fzkuji/test.json", metadata={"help": "Path to the test data."})


@dataclass
class CustomTrainingArguments(TrainingArguments):
    output_dir: str = field(default="./model",
                            metadata={"help": "Output directory for model checkpoints and predictions."})
    model_max_length: int = field(default=256, metadata={"help": "Maximum sequence length."})
    use_lora: bool = field(default=False, metadata={"help": "Whether to use LoRA for fine-tuning."})
    char_limit: int = field(default=500, metadata={"help": "Maximum sequence length."})
    per_device_test_batch_size: int = field(default=8, metadata={"help": "test_batch_size."})


def pretrain(model_args, huatuo_data_path, training_args, accelerator):
    # 加载模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        device_map="auto"
    )

    model = prepare_model_for_kbit_training(model)

    # 如果使用 LoRA 进行微调
    if training_args.use_lora:
        peft_config = AdaLoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False
        )
    model = get_peft_model(model, peft_config)
    model.is_parallelizable = True
    model.model_parallel = True
    model.print_trainable_parameters()

    # 加载 Huatuo 数据集
    huatuo_dataset = HuatuoQADataset(huatuo_data_path, tokenizer, int(training_args.model_max_length / 2))
    # 随机选择 1% 的训练数据
    huatuo_size = int(0.1 * len(huatuo_dataset))
    huatuo_indices = random.sample(range(len(huatuo_dataset)), huatuo_size)
    huatuo_dataset = Subset(huatuo_dataset, huatuo_indices)
    huatuo_dataloader = DataLoader(huatuo_dataset, batch_size=int(training_args.per_device_train_batch_size * 2),
                                   shuffle=True,
                                   collate_fn=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False))

    # 准备模型和数据加载器以便加速
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_args.learning_rate)
    model, optimizer, huatuo_dataloader = accelerator.prepare(model, optimizer, huatuo_dataloader)

    # 预训练循环
    pretrain_epochs = 1  # 设定预训练的轮数
    for epoch in range(pretrain_epochs):
        model.train()
        epoch_loss = 0
        with tqdm(huatuo_dataloader, unit="batch") as tepoch:
            for step, batch in enumerate(tepoch):
                tepoch.set_description(f"Pretrain Epoch {epoch}")
                outputs = model(**batch)
                loss = outputs.loss
                epoch_loss += loss.item()
                accelerator.backward(loss)

                optimizer.step()
                optimizer.zero_grad()

                if step % 10 == 0:
                    tepoch.set_postfix(loss=loss.item())

        print(f"Pretrain Epoch {epoch} completed. Average Loss: {epoch_loss / len(huatuo_dataloader)}")

        # 保存模型
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(training_args.output_dir, save_function=accelerator.save)
    tokenizer.save_pretrained(training_args.output_dir)
    print("Preraining completed and model saved.")

    # 返回预训练的模型
    return model, tokenizer


def train(model_args, data_args, training_args, accelerator, pretrained_model=None, pretrained_tokenizer=None):
    # 加载模型和分词器，如果有预训练模型则使用该模型
    tokenizer = pretrained_tokenizer if pretrained_tokenizer else AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, use_fast=False, trust_remote_code=True)
    model = pretrained_model if pretrained_model else AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=True, device_map="auto")

    model = prepare_model_for_kbit_training(model)

    # 如果使用 LoRA 进行微调
    if training_args.use_lora:
        peft_config = AdaLoraConfig(
            init_r=128,
            lora_alpha=256,
            # target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "W_pack"],
            # target_modules=['down_proj', 'gate_proj', 'up_proj', 'W_pack', 'o_proj'],
            target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],  # 现存问题只微调部分演示即可
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False
        )
    model = get_peft_model(model, peft_config)
    model.is_parallelizable = True
    model.model_parallel = True
    model.print_trainable_parameters()

    # 创建数据集和加载器
    full_train_dataset = BaichuanQADataset(data_args.train_data_path, tokenizer, training_args.model_max_length,
                                           training_args.char_limit)

    # 随机选择 1% 的训练数据
    train_size = int(0.1 * len(full_train_dataset))
    train_indices = random.sample(range(len(full_train_dataset)), train_size)
    train_dataset = Subset(full_train_dataset, train_indices)

    # 加载验证集
    full_val_dataset = BaichuanQATestDataset(data_args.val_data_path, tokenizer, training_args.model_max_length)

    # 随机选择 1% 的验证数据
    val_size = int(1 * len(full_val_dataset))
    val_indices = random.sample(range(len(full_val_dataset)), val_size)
    val_dataset = Subset(full_val_dataset, val_indices)

    # 使用 DataCollatorForLanguageModeling 来自动创建 labels
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # 创建数据加载器
    train_dataloader = DataLoader(train_dataset, batch_size=training_args.per_device_train_batch_size, shuffle=True,
                                  collate_fn=data_collator)
    val_dataloader = DataLoader(val_dataset, batch_size=training_args.per_device_test_batch_size, shuffle=False,
                                collate_fn=custom_collate_fn)  # 使用自定义的 collate_fn

    # 准备模型、优化器和数据加载器以便加速
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_args.learning_rate)
    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(model, optimizer, train_dataloader,
                                                                             val_dataloader)

    # 开始训练循环
    for epoch in range(training_args.num_train_epochs):
        torch.cuda.empty_cache()  # 清理显存缓存
        model.train()
        # tokenizer.padding_side = "right" #设置回右填充
        epoch_loss = 0
        with tqdm(train_dataloader, unit="batch") as tepoch:
            for step, batch in enumerate(tepoch):
                tepoch.set_description(f"Epoch {epoch}")
                outputs = model(**batch)
                loss = outputs.loss
                epoch_loss += loss.item()
                accelerator.backward(loss)

                if step % 10 == 0:
                    tepoch.set_postfix(loss=loss.item(),
                                       step_time=f"{tepoch.format_dict['elapsed'] / (step + 1):.2f}s/it")

                optimizer.step()
                optimizer.zero_grad()

        print(f"Epoch {epoch} completed. Average Training Loss: {epoch_loss / len(train_dataloader)}")

        # 验证循环
        torch.cuda.empty_cache()  # 清理显存缓存
        # validate(model, tokenizer, val_dataloader, training_args, epoch, accelerator)

    # 保存模型
    # accelerator.wait_for_everyone()
    # unwrapped_model = accelerator.unwrap_model(model)
    # unwrapped_model.save_pretrained(training_args.output_dir, save_function=accelerator.save)
    # tokenizer.save_pretrained(training_args.output_dir)
    accelerator.wait_for_everyone()
    model.save_pretrained(training_args.output_dir, save_function=accelerator.save, safe_serialization=True)
    tokenizer.save_pretrained(training_args.output_dir)
    print("Training completed and model saved.")


def validate(model, tokenizer, val_dataloader, training_args, epoch, accelerator):
    model.eval()
    # tokenizer.padding_side = "left"  # 设置填充在左侧
    val_predictions = []
    val_labels = []
    with tqdm(val_dataloader, unit="batch") as vepoch:
        for batch in vepoch:
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=batch['input_ids'].to(accelerator.device),
                    attention_mask=batch['attention_mask'].to(accelerator.device),
                    max_new_tokens=256,
                    do_sample=False,
                    num_beams=1,
                )
            # 获取生成的预测文本
            val_predictions.extend([tokenizer.decode(pred, skip_special_tokens=True) for pred in outputs])

            # 获取实际的正确答案
            val_labels.extend(batch['correct_answer'])

    # 保存验证集预测结果
    output_file = os.path.join(training_args.output_dir, f"val_predictions_epoch_{epoch}.csv")
    pd.DataFrame({"Generated Text": val_predictions, "Actual Text": val_labels}).to_csv(output_file, index=False)
    print(f"Validation predictions for epoch {epoch} saved to '{output_file}'")


def test(model_args, data_args, training_args, accelerator):
    # 加载模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=False, trust_remote_code=True)
    # tokenizer.padding_side = "left"  # 设置填充在左侧
    # 使用 PeftModel.from_pretrained 来加载微调模型
    base_model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        device_map="auto"
    )
    # print(base_model)
    model = PeftModel.from_pretrained(base_model, training_args.output_dir)  # 使用 PEFT 加载微调后的模型
    # state_dict = load_file("model/adapter_model.safetensors")
    # model.load_state_dict(state_dict, strict=False)
    print("Model parameters after loading:")
    print(model)

    model = accelerator.prepare(base_model)

    # 加载测试集
    test_dataset = BaichuanQATestDataset(data_args.test_data_path, tokenizer, training_args.model_max_length)

    # 随机选择 1% 的测试数据
    # test_size = int(1 * len(test_dataset))
    # test_indices = random.sample(range(len(test_dataset)), test_size)
    # test_dataset = Subset(test_dataset, test_indices)

    test_dataloader = DataLoader(test_dataset, batch_size=training_args.per_device_test_batch_size, shuffle=False,
                                 collate_fn=custom_collate_fn)  # 使用自定义的 collate_fn

    # 使用模型进行测试集预测
    model.eval()
    test_predictions = []
    test_labels = []
    with tqdm(test_dataloader, unit="batch") as tepoch:
        for batch in tepoch:
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=batch['input_ids'].to(accelerator.device),
                    attention_mask=batch['attention_mask'].to(accelerator.device),
                    max_new_tokens=256,
                    do_sample=False,
                    num_beams=1,
                )

            # 获取生成的预测文本
            test_predictions.extend([tokenizer.decode(pred, skip_special_tokens=True) for pred in outputs])

            # 获取实际的正确答案
            test_labels.extend(batch['correct_answer'])

    # 保存测试集预测结果
    pd.DataFrame({"Generated Text": test_predictions, "Actual Text": test_labels}).to_csv(
        os.path.join(training_args.output_dir, "test_predictions.csv"), index=False)
    print("Test predictions saved to 'test_predictions.csv'")


if __name__ == "__main__":
    # 实例化参数对象
    model_args = ModelArguments()
    data_args = DataArguments()
    training_args = CustomTrainingArguments(
        output_dir="./model",
        per_device_train_batch_size=8,
        per_device_test_batch_size=64,
        num_train_epochs=4,
        logging_dir="./logs",
        model_max_length=256,
        char_limit=200,
        use_lora=True,
        learning_rate=1e-4
    )

    accelerator = Accelerator()

    # 预训练 Huatuo 数据集
    # huatuo_data_path = "huatuo_knowledge_graph_qa/train_datasets.jsonl"  # 设置 Huatuo 数据集路径
    # pretrained_model, pretrained_tokenizer = pretrain(model_args, huatuo_data_path, training_args, accelerator)
    torch.cuda.empty_cache()  # 清理显存缓存

    # 使用预训练模型进行正式训练
    # train(model_args, data_args, training_args, accelerator, pretrained_model=pretrained_model, pretrained_tokenizer=pretrained_tokenizer)
    # 不使用预训练模式
    train(model_args, data_args, training_args, accelerator)

    # 清理训练资源
    del accelerator
    torch.cuda.empty_cache()  # 清理显存缓存

    # 重新初始化 `Accelerator` 并运行测试
    accelerator = Accelerator()
    test(model_args, data_args, training_args, accelerator)