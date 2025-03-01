import json
import os

# 入力ファイル（JSON形式）
input_file = "base_data/nmle_110-117.json"

# 出力ファイル（JSONL形式）
output_file = "data/sft_data.jsonl"

# 出力ディレクトリの作成
os.makedirs(os.path.dirname(output_file), exist_ok=True)


def preprocess_fn(example):
    """
    JSONデータをSFTの形式に変換する
    """
    if "question" not in example or "choices" not in example or "answer" not in example:
        raise ValueError(f"Missing required fields in example: {example}")

    # 問題文と選択肢を作成
    question_text = example["question"]
    choices_text = "\n".join(
        [f"{chr(97 + i)}. {choice}" for i, choice in enumerate(example["choices"])])

    # 正解の選択肢を特定
    answer_keys = example["answer"]
    correct_answers = [choice for key, choice in zip(
        "abcdefghijklmnopqrstuvwxyz", example["choices"]) if key in answer_keys]

    # `prompt` の作成
    prompt_text = f"User: {question_text}\n{choices_text}\nAssistant:"

    # `response` の作成（詳細な解説付き）
    response_text = f"正解は {', '.join(answer_keys)} です。\n{example.get('explanation', '解説はありません。')}"

    return {
        "prompt": prompt_text,
        "response": response_text
    }


# JSONデータを読み込み
with open(input_file, "r", encoding="utf-8") as infile:
    data = json.load(infile)  # JSONファイル全体をロード

# JSONLとして追記モードで保存
with open(output_file, "a", encoding="utf-8") as outfile:
    for example in data:
        converted = preprocess_fn(example)
        json.dump(converted, outfile, ensure_ascii=False)
        outfile.write("\n")  # 各データを改行で区切る

print(f"Converted data appended to {output_file}")
