import jieba
from nltk.util import ngrams
from nltk.metrics import precision, recall, f_measure
from nltk.translate.bleu_score import sentence_bleu

def rouge_n(reference, hypothesis, n):
    
    '''
        rouge_n = 共享的n-grams 数目 / 参考的n-grams 数目
    '''
    hypothesis_tokens = list(jieba.cut(hypothesis))
    reference_tokens = list(jieba.cut(reference))
    
    reference_ngrams = list(ngrams(reference_tokens, n))
    hypothesis_ngrams = list(ngrams(hypothesis_tokens, n))
    
    intersection = set(reference_ngrams) & set(hypothesis_ngrams)
    recall_score = len(intersection) / len(reference_ngrams) if len(reference_ngrams) > 0 else 0
    precision_score = len(intersection) / len(hypothesis_ngrams) if len(hypothesis_ngrams) > 0 else 0

    return precision_score, recall_score

def rouge_l(reference, hypothesis):
    # reference_tokens = list(jieba.cut(reference))
    # hypothesis_tokens = list(jieba.cut(hypothesis))
    _,lcs_len = _longest_common_subsequence(list(reference), list(hypothesis))
    recall_score = lcs_len / len(reference) if len(reference) > 0 else 0
    precision_score = lcs_len / len(hypothesis) if len(hypothesis) > 0 else 0
    
    return precision_score, recall_score

def _longest_common_subsequence(reference_tokens, hypothesis_tokens):
    table = [[0] * (len(hypothesis_tokens) + 1) for _ in range(len(reference_tokens) + 1)]
    
    for i in range(1, len(reference_tokens) + 1):
        for j in range(1, len(hypothesis_tokens) + 1):
            if reference_tokens[i - 1] == hypothesis_tokens[j - 1]:
                table[i][j] = table[i - 1][j - 1] + 1
            else:
                table[i][j] = max(table[i - 1][j], table[i][j - 1])
    
    lcs_length = table[-1][-1]
    lcs = []
    
    i, j = len(reference_tokens), len(hypothesis_tokens)
    while i > 0 and j > 0:
        if reference_tokens[i - 1] == hypothesis_tokens[j - 1]:
            lcs.append(reference_tokens[i - 1])
            i -= 1
            j -= 1
        elif table[i - 1][j] >= table[i][j - 1]:
            i -= 1
        else:
            j -= 1
    
    lcs.reverse()
    return lcs, lcs_length

# def rouge_w(reference, hypothesis, window_size=1):
#     reference_tokens = reference.split()
#     hypothesis_tokens = hypothesis.split()
    
#     reference_word_pairs = _create_word_pairs(reference_tokens, window_size)
#     hypothesis_word_pairs = _create_word_pairs(hypothesis_tokens, window_size)
    
#     intersection = set(reference_word_pairs) & set(hypothesis_word_pairs)
#     recall_score = len(intersection) / len(reference_word_pairs) if len(reference_word_pairs) > 0 else 0
#     precision_score = len(intersection) / len(hypothesis_word_pairs) if len(hypothesis_word_pairs) > 0 else 0
    
#     return precision_score, recall_score

# def _create_word_pairs(tokens, window_size):
#     word_pairs = []
#     for i in range(len(tokens) - 1):
#         for j in range(1, window_size + 1):
#             if i + j < len(tokens):
#                 word_pairs.append((tokens[i], tokens[i + j]))
#     return word_pairs

# def rouge_s(reference, hypothesis):
#     return f_measure(set(reference), set(hypothesis))


if __name__ == "__main__":
    # 示例用法
    reference = "迈克生物定增结果：高毅资产获配近2.7亿元"
    hypothesis = "【迈克生物】定增完成：募资15.74亿，高毅资产获配近2.7亿元\n"

    # ROUGE-N (n=1)
    precision_score, recall_score = rouge_n(reference, hypothesis, n=1)
    print("ROUGE-1 Precision:", precision_score)
    print("ROUGE-1 Recall:", recall_score)

    # ROUGE-L
    precision_score, recall_score = rouge_l(reference, hypothesis)
    print("ROUGE-L Precision:", precision_score)
    print("ROUGE-L Recall:", recall_score)

    # # ROUGE-W (window_size=2)
    # precision_score, recall_score = rouge_w(reference, hypothesis, window_size=2)
    # print("ROUGE-W Precision:", precision_score)
    # print("ROUGE-W Recall:", recall_score)

    # # ROUGE-S
    # f1_score = rouge_s(reference.split(), hypothesis.split())
    # print("ROUGE-S F1 Score:", f1_score)
