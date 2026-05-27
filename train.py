import gc
import os
import pathlib
import sys
from typing import Any, Optional

import torch
from huggingface_hub import snapshot_download
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from config import (
    MODEL_NAME,
    CACHE_DIR,
    ADAPTER_OUTPUT_DIR,
    MAX_SEQ_LENGTH,
    NUM_TRAIN_EPOCHS,
    PER_DEVICE_TRAIN_BATCH_SIZE,
    PER_DEVICE_EVAL_BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS,
    LEARNING_RATE,
    WARMUP_STEPS,
    LR_SCHEDULER_TYPE,
    SAVE_STEPS,
    SAVE_TOTAL_LIMIT,
    LOGGING_STEPS,
    FP16,
    BF16,
    LORA_R,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_TARGET_MODULES,
)

from dataset import (
    load_and_join_data,
    build_records,
    split_records,
    records_to_dataset,
)


def patch_windows_utf8_for_trl() -> None:
    """
    Windows에서 TRL 내부 jinja 파일을 cp949로 읽어 발생하는 UnicodeDecodeError를 방지한다.
    """
    original_read_text = pathlib.Path.read_text

    def read_text_utf8(
        self: pathlib.Path,
        encoding: Optional[str] = None,
        errors: Optional[str] = None
    ) -> str:
        return original_read_text(self, encoding=encoding or "utf-8", errors=errors)

    pathlib.Path.read_text = read_text_utf8

    for name in list(sys.modules.keys()):
        if name == "trl" or name.startswith("trl."):
            del sys.modules[name]


def setup_environment() -> None:
    """
    Hugging Face 및 Torch 캐시를 v2 D드라이브 폴더로 고정한다.
    """
    os.environ["HF_HOME"] = str(CACHE_DIR)
    os.environ["TRANSFORMERS_CACHE"] = str(CACHE_DIR / "transformers")
    os.environ["HF_DATASETS_CACHE"] = str(CACHE_DIR / "datasets")
    os.environ["TORCH_HOME"] = str(CACHE_DIR / "torch")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ADAPTER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_tokenizer() -> Any:
    """
    Qwen3 tokenizer를 로드한다.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        use_fast=True,
        cache_dir=str(CACHE_DIR),
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"
    return tokenizer


def load_base_model() -> Any:
    """
    Qwen3-8B base model을 4bit quantization으로 GPU에 로드한다.

    주의:
    현재 환경의 transformers/Qwen3 조합에서는 dtype 인자가 오류를 일으킬 수 있으므로
    torch_dtype 인자를 사용한다.
    """
    model_cache_path = snapshot_download(
        repo_id=MODEL_NAME,
        cache_dir=str(CACHE_DIR),
        local_files_only=False,
    )

    compute_dtype = torch.float16

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )

    torch.cuda.empty_cache()
    gc.collect()

    model = AutoModelForCausalLM.from_pretrained(
        model_cache_path,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True,
        torch_dtype=compute_dtype,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )

    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    return model


def build_lora_config() -> LoraConfig:
    """
    8GB VRAM 기준의 경량 LoRA 설정을 반환한다.
    """
    return LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )


def find_latest_checkpoint() -> Optional[str]:
    """
    기존 checkpoint가 있으면 가장 최신 checkpoint 경로를 반환한다.
    """
    if not ADAPTER_OUTPUT_DIR.exists():
        return None

    checkpoints = []

    for path in ADAPTER_OUTPUT_DIR.glob("checkpoint-*"):
        if path.is_dir():
            try:
                step = int(path.name.split("-")[-1])
                checkpoints.append((step, path))
            except ValueError:
                continue

    if not checkpoints:
        return None

    checkpoints.sort(key=lambda x: x[0])
    return str(checkpoints[-1][1])


def main() -> None:
    """
    MediQ v2 CSV 데이터로 Qwen3-8B QLoRA fine-tuning을 수행한다.
    """
    patch_windows_utf8_for_trl()
    setup_environment()

    # TRL은 Windows UTF-8 패치 이후 import해야 한다.
    from trl import SFTConfig, SFTTrainer

    print("[MediQ v2] tokenizer 로드 중...")
    tokenizer = load_tokenizer()

    print("[MediQ v2] CSV 로드 및 데이터셋 생성 중...")
    joined_df = load_and_join_data()
    records = build_records(joined_df)
    train_records, valid_records, test_records = split_records(records)

    train_dataset = records_to_dataset(train_records, tokenizer)
    valid_dataset = records_to_dataset(valid_records, tokenizer)

    print(f"[MediQ v2] train: {len(train_dataset)}")
    print(f"[MediQ v2] valid: {len(valid_dataset)}")
    print(f"[MediQ v2] test: {len(test_records)}")

    print("[MediQ v2] Qwen3-8B 4bit model 로드 중...")
    model = load_base_model()

    print("[MediQ v2] LoRA 설정 중...")
    model = prepare_model_for_kbit_training(model)
    lora_config = build_lora_config()

    training_args = SFTConfig(
        output_dir=str(ADAPTER_OUTPUT_DIR),

        dataset_text_field="text",
        max_length=MAX_SEQ_LENGTH,
        packing=False,

        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,

        num_train_epochs=NUM_TRAIN_EPOCHS,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        lr_scheduler_type=LR_SCHEDULER_TYPE,

        optim="paged_adamw_8bit",
        fp16=FP16,
        bf16=BF16,

        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_strategy="steps",
        save_total_limit=SAVE_TOTAL_LIMIT,

        # RTX 4060 Laptop 8GB VRAM에서는 평가까지 켜면 OOM 가능성이 높아 비활성화
        eval_strategy="no",

        gradient_checkpointing=True,
        max_grad_norm=0.3,

        report_to="none",
        remove_unused_columns=True,
    )

    try:
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            peft_config=lora_config,
            processing_class=tokenizer,
        )
    except TypeError:
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            peft_config=lora_config,
            tokenizer=tokenizer,
        )

    latest_checkpoint = find_latest_checkpoint()

    if latest_checkpoint:
        print(f"[MediQ v2] checkpoint에서 이어서 학습합니다: {latest_checkpoint}")
        trainer.train(resume_from_checkpoint=latest_checkpoint)
    else:
        print("[MediQ v2] 새 학습을 시작합니다.")
        trainer.train()

    print("[MediQ v2] LoRA adapter 저장 중...")
    trainer.model.save_pretrained(str(ADAPTER_OUTPUT_DIR))
    tokenizer.save_pretrained(str(ADAPTER_OUTPUT_DIR))

    print(f"[MediQ v2] 학습 완료: {ADAPTER_OUTPUT_DIR}")


if __name__ == "__main__":
    main()