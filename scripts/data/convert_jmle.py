import json
import os

# 入力ファイルと出力ファイルのパス
input_file = "base_data/jmle_106-116_simpleQA.jsonl"
output_file = "data/sft_data.jsonl"

# 出力ディレクトリが存在しない場合は作成
os.makedirs(os.path.dirname(output_file), exist_ok=True)


def preprocess_fn(example):
    """
    JSONLのデータをSFTの形式に変換する
    """
    if "question" not in example or "answer" not in example:
        raise ValueError(
            f"Missing 'question' or 'answer' in example: {example}")

    # 問題文の整形
    question_text = example["question"]

    # 選択肢を抽出 ( "A. ..." の形式を探す)
    choices = []
    for part in question_text.split():
        if part.endswith(".") and len(part) == 2:  # "A.", "B.", "C." のようなもの
            choices.append(part)

    # `answer` をアルファベット → 選択肢の文章に変換
    answer_key = example["answer"]
    if answer_key in choices:
        correct_answer = question_text.split(
            answer_key + " ")[1].split(" ")[0]  # 選択肢の内容を抽出
    else:
        correct_answer = "該当なし"

    # `prompt` の作成
    prompt_text = f"User: {question_text}\nAssistant:"

    # `response` の作成 (正解選択肢を含む詳細回答)
    response_text = f"正解は {answer_key} の {correct_answer} です。"

    return {
        "prompt": prompt_text,
        "response": response_text
    }


# データを変換して保存
with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
    for line in infile:
        example = json.loads(line.strip())
        converted = preprocess_fn(example)
        json.dump(converted, outfile, ensure_ascii=False)
        outfile.write("\n")

print(f"Converted data saved to {output_file}")
