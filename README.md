# SFT_GLM4_on_CMExam
Fine tuning instructions on CMExam dataset using ChatGLM4 and AdaLoRA

使用ChatGLM4在CMEXAM数据集上通过AdaLoRA指令微调

数据集来源：https://huggingface.co/datasets/fzkuji/CMExam

----
安装必须文件requirement.txt
自行下载GLM4-9B-Base模型，并放置于GLM4-9B文件夹中

下载地址：https://huggingface.co/THUDM/glm-4-9b

目前模型存在一些bug，需要在模型文件 GLM4-9B/tokenization_chatglm.py 中，在 def _pad() 方法中增加参数：
padding_side: Optional[str] = None

否则会报错TypeError: ChatGLM4Tokenizer._pad() got an unexpected keyword argument 'padding_side'

参考教程：https://blog.csdn.net/m0_60801087/article/details/143160274


准备完成后，运行main文件开始训练：
python main.py

训练完成后，可通过check.py验证结果指标：
python check.py

----
至少需要使用16G以上显卡进行训练。使用A800 80G单卡进行训练，per_device_train_batch_size可以设置为8

目前最佳训练结果，对比chatgpt4：

| Models                 | ChatGPT4 (-) | ChatGLM4-Base+SFT (9B) |
|------------------------|--------------|-------------------------|
| Prediction_Acc         | 61.60        | 75.60                   |
| Reasoning_BLUE-1       | 0.17         | 40.24                   |
| Reasoning_BLUE-4       | 0.06         | 12.14                   |
| Reasoning_ROUGE-1      | 29.74        | 61.19                   |
| Reasoning_ROUGE-2      | 14.84        | 33.42                   |

目前的main.py函数中写入了pretrain训练函数，通过huatuo数据集进行再次预训练。但在代码以及最佳训练结果中均未使用。如有需要的话可以手动选择开启。

