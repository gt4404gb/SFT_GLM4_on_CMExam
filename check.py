import pandas as pd
import re
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

# 加载测试集和生成数据集的CSV文件
test_data = pd.read_csv('CMExam-main/data/test_with_annotations.csv')  # 替换为实际测试集文件路径
generated_data = pd.read_csv('最佳结果.csv')  # 替换为实际生成数据集文件路径


# 定义函数，将 Options 列转换为与 Generated Text 一致的格式，并去除每个选项文本后的多余空格
def format_options(options_text):
    options = options_text.split('\n')
    formatted_options = [re.sub(r"([A-E]) (.+)", r"\1: \2", opt).strip() for opt in options if opt]
    return '\n'.join(formatted_options)


# 对测试集的 Options 列进行格式化处理
test_data['Formatted Options'] = test_data['Options'].apply(format_options)


# 去除 Generated Text 列开头的提示文本
def remove_prompt(text):
    prompt_pattern = r"^对于后续问题，你需按照以下格式，向分析和正确选项字段填入答案。\s+格式：\s+问题：问题内容\s+分析：\{填入分析，50字以内\}\s+正确选项：\{填入正确的选项和选项文本\}"
    return re.sub(prompt_pattern, "", text).strip()


# 应用函数去除生成数据集中 Generated Text 列的开头提示文本
generated_data['Generated Text'] = generated_data['Generated Text'].apply(remove_prompt)

# 创建一个新列以存储 Generated Text
test_data['Generated Text'] = ''

# 使用包含关系来匹配，并同时检查 Question 和 Formatted Options，禁用正则表达式匹配
for index, row in test_data.iterrows():
    question = row['Question']
    formatted_options = row['Formatted Options']

    matched_text = generated_data[
        generated_data['Generated Text'].str.contains(question, na=False, regex=False) &
        generated_data['Generated Text'].str.contains(formatted_options, na=False, regex=False)
        ]

    if not matched_text.empty:
        test_data.at[index, 'Generated Text'] = matched_text.iloc[0]['Generated Text']


# 定义提取正确答案的函数
def extract_answer(text):
    match = re.search(r"正确选项：([A-E]+)", text)
    if match:
        return match.group(1)
    return None


# 创建新列 'Extracted Answer' 存储从 Generated Text 中提取的答案
test_data['Extracted Answer'] = test_data['Generated Text'].apply(
    lambda x: extract_answer(x) if pd.notnull(x) else None)

# 创建新列 'Correct'，判断 Answer 和 Extracted Answer 是否一致
test_data['Correct'] = test_data.apply(lambda x: x['Answer'] == x['Extracted Answer'], axis=1)
test_data['Correct'] = test_data['Correct'].apply(lambda x: '正确' if x else '错误')

# 计算正确率
accuracy = test_data['Correct'].value_counts(normalize=True).get('正确', 0) * 100
print(f"正确率: {accuracy:.2f}%")

# 初始化评分器
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=True)
bleu_1_scores = []
bleu_4_scores = []
rouge_1_scores = []
rouge_2_scores = []

# 定义平滑函数
smooth_fn = SmoothingFunction().method1

# 去除“问题”和“分析”部分，便于计算BLEU和ROUGE
generated_data['Generated Text'] = generated_data['Generated Text'].str.replace(r"问题：[\s\S]*?分析：", '', regex=True)
generated_data['Actual Text'] = generated_data['Actual Text'].str.replace(r"问题：[\s\S]*?分析：", '', regex=True)

# 遍历每行并计算分数
for _, row in generated_data.iterrows():
    generated_text = row['Generated Text'].split()
    actual_text = [row['Actual Text'].split()]

    # 计算BLEU-1和BLEU-4，应用平滑
    bleu_1 = sentence_bleu(actual_text, generated_text, weights=(1, 0, 0, 0), smoothing_function=smooth_fn)
    bleu_4 = sentence_bleu(actual_text, generated_text, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth_fn)
    bleu_1_scores.append(bleu_1)
    bleu_4_scores.append(bleu_4)

    # 计算ROUGE-1和ROUGE-2
    rouge_scores = scorer.score(row['Actual Text'], row['Generated Text'])
    rouge_1_scores.append(rouge_scores['rouge1'].fmeasure)
    rouge_2_scores.append(rouge_scores['rouge2'].fmeasure)

# 计算总体得分
overall_bleu_1 = sum(bleu_1_scores) / len(bleu_1_scores) * 100
overall_bleu_4 = sum(bleu_4_scores) / len(bleu_4_scores) * 100
overall_rouge_1 = sum(rouge_1_scores) / len(rouge_1_scores) * 100
overall_rouge_2 = sum(rouge_2_scores) / len(rouge_2_scores) * 100

# 输出总体得分
print(f"Overall BLEU-1: {overall_bleu_1:.2f}%")
print(f"Overall BLEU-4: {overall_bleu_4:.2f}%")
print(f"Overall ROUGE-1: {overall_rouge_1:.2f}%")
print(f"Overall ROUGE-2: {overall_rouge_2:.2f}%")

# 保存最终结果到新的CSV文件
output_path = 'final_result_with_accuracy_and_scores.csv'  # 替换为实际输出路径
test_data.to_csv(output_path, index=False, encoding='utf-8-sig')

print(
    "已成功保存包含 'Generated Text'、'Extracted Answer'、'Correct' 列以及计算的 BLEU 和 ROUGE 分数的数据集，并计算出正确率和总体得分。")
