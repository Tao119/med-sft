# scripts/prepare_data.py

import json


def generate_sft_data():
    """
    SFT用: instruction, input, output を持つサンプルデータを生成。
    """
    data = [
        {
            "instruction": "感冒（かぜ）の患者への推奨対応",
            "input": "",
            "output": "十分な水分補給、安静、解熱鎮痛薬などを使用し、睡眠をしっかり取って休むよう指導します。"
        },
        {
            "instruction": "インフルエンザ予防接種の推奨時期は？",
            "input": "",
            "output": "流行が始まる2週間前までに接種するのが望ましいとされています。"
        }
    ]
    return data


def generate_reward_data():
    """
    Rewardモデル学習用: promptに対して複数回答 (response_1, response_2) を持ち、どちらが良いか(label) を付与。
    label: 1 => response_1が良い, 2 => response_2が良い
    """
    data = [
        {
            "prompt": "高血圧患者が気をつけるべき食事は？",
            "response_1": "塩分を控えめにする必要があり、野菜や果物を多く摂ることが推奨されます。",
            "response_2": "どのような食事でも好きなだけ食べて良い。",
            "label": 1
        },
        {
            "prompt": "糖尿病患者に運動は必要か？",
            "response_1": "軽い有酸素運動を週に数回行うとよいです。",
            "response_2": "特に運動はしなくても大丈夫です。",
            "label": 1
        }
    ]
    return data


def generate_ppo_prompts():
    """
    PPO用: プロンプトのみを用意し、学習時にモデルに回答させる。
    """
    data = [
        {"prompt": "風邪の症状が出始めたときにまずすべきことは？"},
        {"prompt": "インフルエンザと普通の風邪の見分け方を教えてください。"}
    ]
    return data


def generate_dpo_data():
    """
    DPO用: promptに対して (response_1, response_2) の良い回答・悪い回答ペアを持ち、どちらが良いかをlabelで指定。
    """
    data = [
        {
            "prompt": "妊娠中のインフルエンザ予防接種は安全か？",
            "response_1": "一般的に妊娠中でもインフルエンザ予防接種は推奨されるが、主治医と相談の上実施します。",
            "response_2": "妊娠中は絶対に予防接種をしてはいけません。",
            "label": 1
        },
        {
            "prompt": "発熱したらすぐ抗生物質を飲むべき？",
            "response_1": "発熱の原因が細菌性と診断された場合にのみ、医師の指示で抗生物質を用います。",
            "response_2": "どのような発熱でもとりあえず抗生物質を飲めばOKです。",
            "label": 1
        }
    ]
    return data


def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    # 出力先を指定
    sft_data_path = "../data/sft_data.jsonl"
    reward_data_path = "../data/reward_data.jsonl"
    ppo_prompts_path = "../data/ppo_prompts.jsonl"
    dpo_data_path = "../data/dpo_data.jsonl"

    sft_data = generate_sft_data()
    reward_data = generate_reward_data()
    ppo_data = generate_ppo_prompts()
    dpo_data = generate_dpo_data()

    save_jsonl(sft_data, sft_data_path)
    save_jsonl(reward_data, reward_data_path)
    save_jsonl(ppo_data, ppo_prompts_path)
    save_jsonl(dpo_data, dpo_data_path)

    print("Data generated successfully!")
