from pathlib import Path

# ============================================================
# Path Config
# ============================================================

BASE_DIR = Path(r"D:\학교\졸작 데이터\MediQ_AI_v3")

DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
CACHE_DIR = BASE_DIR / "hf_cache"

MAPPING_CSV_PATH = DATA_DIR / "mediq_mapping_reviewed.csv"
OTC_CSV_PATH = DATA_DIR / "mediq_otc_medicine.csv"

# v3 전용 adapter 저장 경로
ADAPTER_OUTPUT_DIR = OUTPUT_DIR / "qwen3_8b_mediq_qlora_adapter_v3"

MODEL_NAME = "Qwen/Qwen3-8B"

# ============================================================
# Dataset Config
# ============================================================

RANDOM_SEED = 42
DEFAULT_RECOMMENDED_MEDICINES = "해당 없음 (병원 방문 권고)"

# ============================================================
# Training Config
# RTX 4060 Laptop GPU 8GB VRAM 기준 안전 설정
# ============================================================

MAX_SEQ_LENGTH = 256
NUM_TRAIN_EPOCHS = 1

PER_DEVICE_TRAIN_BATCH_SIZE = 1
PER_DEVICE_EVAL_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4

LEARNING_RATE = 2e-4
WARMUP_STEPS = 100
LR_SCHEDULER_TYPE = "cosine"

SAVE_STEPS = 25
SAVE_TOTAL_LIMIT = 3
LOGGING_STEPS = 5

FP16 = True
BF16 = False

# ============================================================
# LoRA Config
# ============================================================

LORA_R = 4
LORA_ALPHA = 8
LORA_DROPOUT = 0.05

LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
]

# ============================================================
# Inference Config
# ============================================================

MAX_NEW_TOKENS = 256
RETRY_COUNT = 3