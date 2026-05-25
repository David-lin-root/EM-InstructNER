'''
准确率：预测对的token / 生成的所有token
大致情况，不准确，可以作为一项参考指标
'''

def calculate_accuracy(generated_texts, reference_texts):
    correct_count = 0
    total_count = len(generated_texts)
    generated_tokens = set(generated_texts)
    reference_tokens = set(reference_texts)
    for g_token in generated_tokens:
        if g_token in reference_tokens:
            correct_count += 1
    accuracy = correct_count / total_count
    return accuracy


if __name__ == "__main__":
    # 假设有参考文本作为对照
    generated_texts = "你好"
    reference_texts = "你好啊"
    # 计算准确率
    accuracy = calculate_accuracy(generated_texts, reference_texts)
    print("Accuracy:", accuracy)
    assert accuracy == 1.0
