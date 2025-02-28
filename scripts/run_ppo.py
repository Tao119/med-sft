# scripts/run_ppo.py

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead


def main():
    # ベースポリシーモデル(すでにSFTしたLoRAアダプタを読み込むならそのパスを指定)
    # ここでは便宜上、SFT後の "rinna/qwen2.5-bakeneko-32" + LoRAアダプタを読み込むイメージ
    policy_model_path = "../outputs/sft_output"

    # Rewardモデル
    reward_model_path = "../outputs/reward_output"

    data_path = "../data/ppo_prompts.jsonl"
    output_dir = "../outputs/ppo_output"

    # PPOの設定
    ppo_config = PPOConfig(
        model_name=policy_model_path,  # 初期モデル(SFT済みモデル)を指定
        learning_rate=1e-6,
        log_with=None,                 # ログを取りたい場合は適宜設定
        batch_size=1,
        mini_batch_size=1,
        gradient_accumulation_steps=1,
        max_prompt_length=512,
        # ...
        output_dir=output_dir
    )

    # プロンプト群をロード
    dataset = load_dataset("json", data_files=data_path, split="train")

    tokenizer = AutoTokenizer.from_pretrained(policy_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # PPO用のValueHead付きモデルをロード
    # (SFTで学習済みの重み + ValueHeadを付与)
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        policy_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # Rewardモデルを読み込む(推論用)
    # RewardモデルはProximal Policy Optimization中に回答スコアを返す
    from transformers import AutoModelForSequenceClassification
    from trl import create_reference_model  # 参考モデル(旧Policy)を生成する場合に使用

    reward_model = AutoModelForSequenceClassification.from_pretrained(
        reward_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # 参考モデル (reference model) を作成 (PPOでのKL制御の際に使用)
    ref_model = create_reference_model(model)

    # PPOTrainer初期化
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        reward_model=reward_model,
        dataset=dataset,
    )

    # 単純な学習ループ (デモ用に数ステップのみ)
    for epoch in range(1):
        for sample in dataset:
            prompt = sample["prompt"]
            # モデルが回答を生成
            gen_tokens = ppo_trainer.generate(prompt, max_new_tokens=50)
            response = tokenizer.decode(
                gen_tokens[0], skip_special_tokens=True)

            # PPOの学習ステップ
            # ※ TRLの実装により、 "step" や "step_batched" などの使用が可能
            #   ここでは簡易化のため pseudo-code 的に書く
            ppo_trainer.step(prompt, response)

    # 学習後のモデルを保存
    ppo_trainer.save_model(output_dir)
    print("PPO training complete.")


if __name__ == "__main__":
    main()
