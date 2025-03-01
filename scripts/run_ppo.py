import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, create_reference_model

def preprocess_fn(example):
    """データの前処理関数"""
    if "prompt" not in example:
        raise ValueError(f"Missing prompt key in example: {example}")
    
    return {
        "prompt": example["prompt"]
    }

def main():
    policy_model_path = "/workspace/outputs/sft_output"
    reward_model_path = "/workspace/outputs/reward_output"
    data_path = "/workspace/data/ppo_prompts.jsonl"
    output_dir = "/workspace/outputs/ppo_output"

    # PPO の設定
    ppo_config = PPOConfig(
        learning_rate=1e-6,
        batch_size=1,
        mini_batch_size=1,
        gradient_accumulation_steps=1,
        output_dir=output_dir
    )

    # データセットの読み込みと前処理
    dataset = load_dataset("json", data_files=data_path, split="train")
    dataset = dataset.map(preprocess_fn, remove_columns=dataset.column_names)
    print("Dataset processed successfully. Features:", dataset.features)

    # トークナイザーの読み込み
    tokenizer = AutoTokenizer.from_pretrained(policy_model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # モデルの読み込み
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        policy_model_path,
        device_map="auto",
        torch_dtype=torch.float16
    )

    # 参照モデル（Reference Model）の作成
    ref_model = create_reference_model(model)

    # 報酬モデル（Reward Model）の読み込み
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        reward_model_path,
        device_map="auto",
        torch_dtype=torch.float16
    )

    # `stop_token_id` を手動で設定
    stop_token_id = tokenizer.eos_token_id

    # PPOTrainer のセットアップ
    ppo_trainer = PPOTrainer(
        args=ppo_config,
        processing_class=tokenizer,
        model=model,
        ref_model=ref_model,
        reward_model=reward_model,
        train_dataset=dataset,
        stop_token_id=stop_token_id  # 追加
    )

    # 1エポックのみデモ実行（データ数に応じて調整）
    for epoch in range(1):
        for sample in dataset:
            prompt = sample["prompt"]
            gen_tokens = ppo_trainer.generate(prompt, max_new_tokens=50)
            response = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
            ppo_trainer.step(prompt, response)

    # モデルの保存
    ppo_trainer.save_model(output_dir)
    print("PPO training complete.")

if __name__ == "__main__":
    main()
