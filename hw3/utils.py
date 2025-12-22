import os
import re
import json
from openai import OpenAI
from collections import Counter

# 配置 API Key 和 Base URL
# 请在环境变量中设置或直接修改此处
API_KEY = os.getenv("LLM_API_KEY", "sk-walldtjcnajmxkczdaubsgmcsnxgemcuxomfuhaqioymntat")
BASE_URL = os.getenv("LLM_BASE_URL", "https://api.siliconflow.cn/v1") # 假设使用 SiliconFlow 或类似的兼容 OpenAI 接口的服务，如果不同请修改
MODEL_NAME = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def get_llm_segmentation(text, model=MODEL_NAME, few_shot_examples=None):
    """
    调用 LLM 进行分词
    """
    prompt = "请对以下中文句子进行分词，词与词之间用空格分隔。直接输出分词结果，不要包含任何解释或其他内容。\n"
    
    if few_shot_examples:
        prompt += "示例：\n"
        for ex_raw, ex_seg in few_shot_examples:
            prompt += f"输入：{ex_raw}\n输出：{ex_seg}\n"
        prompt += "\n"
        
    prompt += f"输入：{text}\n输出："

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是一个专业的中文分词工具。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1, # 低温度以保证确定性
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"API调用失败: {e}")
        return text # 失败返回原文本

def calculate_metrics(ground_truth_list, prediction_list):
    """
    计算分词的 Precision, Recall, F1
    ground_truth_list: list of list of words (e.g. [['我', '爱', '北京'], ...])
    prediction_list: list of list of words
    """
    tp = 0
    gold_total = 0
    pred_total = 0

    for gold_words, pred_words in zip(ground_truth_list, prediction_list):
        # 转换为区间集合 (start, end)
        def get_intervals(words):
            intervals = set()
            start = 0
            for w in words:
                end = start + len(w)
                intervals.add((start, end))
                start = end
            return intervals

        gold_intervals = get_intervals(gold_words)
        pred_intervals = get_intervals(pred_words)

        tp += len(gold_intervals & pred_intervals)
        gold_total += len(gold_intervals)
        pred_total += len(pred_intervals)

    precision = tp / pred_total if pred_total > 0 else 0
    recall = tp / gold_total if gold_total > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1

def load_pku_corpus(filepath):
    """
    加载人民日报语料
    格式: 迈向/v  充满/v  希望/n  的/u  新/a  世纪/n
    返回: [(raw_text, [word1, word2, ...]), ...]
    """
    data = []
    # 尝试使用 GBK 编码读取，因为人民日报语料常见编码为 GBK
    try:
        with open(filepath, 'r', encoding='gbk') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                
                # 提取词 (忽略词性标记)
                # 1. 去掉复合词标记 [ ]nt
                line = re.sub(r'\[|\][a-z]*', '', line)
                
                tokens = line.split()
                words = []
                for token in tokens:
                    if '/' in token:
                        word = token.split('/')[0]
                        if word: # 避免空词
                            words.append(word)
                    else:
                        words.append(token)
                
                if words:
                    raw_text = "".join(words)
                    data.append((raw_text, words))
    except UnicodeDecodeError:
        # 如果 GBK 失败，尝试 UTF-8
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                
                line = re.sub(r'\[|\][a-z]*', '', line)
                tokens = line.split()
                words = []
                for token in tokens:
                    if '/' in token:
                        word = token.split('/')[0]
                        if word: 
                            words.append(word)
                    else:
                        words.append(token)
                
                if words:
                    raw_text = "".join(words)
                    data.append((raw_text, words))
    return data
