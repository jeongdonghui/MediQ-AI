# -*- coding: utf-8 -*-
"""
MediQ AI Inference Server v3.3 Free-Text-Aware
==============================================

[v3 → v3.3 개선 누적]
1) 할루시네이션 원천 차단: 후보 Top-K 안에서만 선택.
2) 무의미한 greedy 재시도 루프 제거.
3) Safety-First Red-flag 우회.
4) No-Save 정합 로깅: 증상 원문 마스킹.
5) 하이브리드 스코어링: 모델 단독 선택의 토큰 길이/빈도 편향 보정.
6) CSV-First 계층 게이트:
   ① Red-flag 응급 우회
   ② CSV exact match(부위3단계+증상 정확 일치) → CSV candidate_rank 1순위 즉시 선택
   ③ exact 없음/모호 → Qwen 로그우도 스코어링 + CSV prior 가중
   ④ 모델 결과가 CSV 1순위와 크게 다르지 않으면 CSV 1순위 우선
   ⑤ 실패 → CSV fallback
   ⑥ 최종 department/riskLevel/recommendedMedicines는 항상 CSV/룰 기반
7) 자유 문장 인지형 가중치:
   is_free_text=true일 때만 모델 로그우도 반영 비중을 높인다.
   버튼/정규화 입력은 CSV 우선 안정성을 유지한다.

실행 예시:
uvicorn inference_server_v3_3_free:app --host 0.0.0.0 --port 8000
"""

import gc
import json
import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from huggingface_hub import snapshot_download
from peft import PeftModel
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from config import (
    MODEL_NAME,
    CACHE_DIR,
    ADAPTER_OUTPUT_DIR,
    MAPPING_CSV_PATH,
    OTC_CSV_PATH,
    DEFAULT_RECOMMENDED_MEDICINES,
)

from dataset import SYSTEM_PROMPT, infer_department_and_risk, build_recommended_medicines


# ============================================================
# 0. 튜닝 가능한 상수
# ============================================================

TOP_K_CANDIDATES = 5
LENGTH_NORMALIZE = True

# 버튼/정규화 입력: CSV prior를 더 강하게 신뢰
ALPHA_CSV_PRIOR = 0.70

# 자유 문장 입력: 모델 로그우도 반영 비중을 조금 더 높임
ALPHA_CSV_PRIOR_FREETEXT = 0.55

CSV_SCORE_TEMP = 20.0
MODEL_SCORE_TEMP = 1.0

# 버튼 입력: 모델이 CSV 1순위를 쉽게 뒤집지 못하게 함
MODEL_OVERRIDE_MARGIN = 0.10

# 자유 문장 입력: 모델이 약하게 앞서도 뒤집을 수 있게 함
MODEL_OVERRIDE_MARGIN_FREETEXT = 0.03

TIER_EXACT = "exact"
TIER_REGION3 = "region3"
TIER_REGION2 = "region2"
TIER_SYMPTOM = "symptom"
TIER_GLOBAL = "global"


# ============================================================
# 1. 환경 변수 설정
# ============================================================

os.environ["HF_HOME"] = str(CACHE_DIR)
os.environ["TRANSFORMERS_CACHE"] = str(CACHE_DIR / "transformers")
os.environ["HF_DATASETS_CACHE"] = str(CACHE_DIR / "datasets")
os.environ["TORCH_HOME"] = str(CACHE_DIR / "torch")


# ============================================================
# 2. 요청 / 응답 DTO
# ============================================================

class AiAnalysisRequest(BaseModel):
    reportId: int
    step_1_main_area: str
    step_2_detailed_region: str
    step_3_sub_region: str
    step_4_symptom: str
    is_free_text: bool
    step_5_pain_range_code: str
    step_5_pain_range: str
    step_6_intensity_value: int = Field(ge=0, le=10)
    step_7_onset_time_code: str
    step_7_onset_time: str


class AiAnalysisResult(BaseModel):
    diseaseName: str
    department: str
    riskLevel: str
    recommendedMedicines: str


# ============================================================
# 3. 기본 유틸
# ============================================================

def safe_str(value: Any, default: str = "") -> str:
    """
    NaN, None 값을 안전한 문자열로 변환한다.
    """
    if pd.isna(value):
        return default

    value = str(value).strip()

    if value.lower() == "nan":
        return default

    return value if value else default


def safe_float(value: Any, default: float = 0.0) -> float:
    """
    NaN, None, 잘못된 숫자 값을 안전한 float으로 변환한다.
    """
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def safe_int(value: Any, default: int = 999) -> int:
    """
    NaN, None, 잘못된 숫자 값을 안전한 int로 변환한다.
    """
    try:
        if pd.isna(value):
            return default
        return int(float(value))
    except Exception:
        return default


def mask_pii(text: str, keep: int = 4) -> str:
    """
    로그에 증상 원문을 그대로 남기지 않고 앞 일부만 노출한다.
    """
    text = safe_str(text)

    if not text:
        return "(빈 입력)"

    if len(text) <= keep:
        return text + "***"

    return text[:keep] + "***"


# ============================================================
# 4. Safety-First Red-flag 룰베이스
# ============================================================

RED_FLAG_KEYWORDS: List[str] = [
    "극심한 흉통",
    "가슴이 찢어",
    "숨을 못",
    "호흡곤란",
    "숨이 안",
    "한쪽 마비",
    "편마비",
    "말이 어눌",
    "발음이 안",
    "의식",
    "쓰러",
    "경련",
    "발작",
    "피를 토",
    "각혈",
    "토혈",
    "검은 변",
    "혈변",
    "심한 두통과 구토",
    "벼락두통",
    "갑자기 안 보",
    "시야가 사라",
    "자살",
    "음독",
    "과다복용",
    "심정지",
]

EMERGENCY_RESULT = {
    "diseaseName": "응급 의심 증상 (Red-flag)",
    "department": "응급의학과",
    "riskLevel": "응급",
    "recommendedMedicines": "자가 약 복용 금지. 즉시 119 신고 또는 가까운 응급실 방문을 권고합니다.",
}


def detect_red_flag(input_json: Dict[str, Any]) -> bool:
    """
    증상 텍스트 및 강도/부위 조합으로 응급 패턴을 탐지한다.
    """
    haystack = " ".join(
        safe_str(input_json.get(k, ""))
        for k in ("step_3_sub_region", "step_4_symptom", "step_5_pain_range")
    )

    for keyword in RED_FLAG_KEYWORDS:
        if keyword in haystack:
            return True

    intensity = safe_int(input_json.get("step_6_intensity_value", 0), 0)
    main_area = safe_str(input_json.get("step_1_main_area", ""))

    if intensity >= 9 and ("가슴" in main_area or "머리" in main_area):
        return True

    return False


# ============================================================
# 5. CSV 로드
# ============================================================

def load_allowed_data() -> Tuple[set, Dict[str, str], pd.DataFrame]:
    """
    허용 질환명 목록, 약 성분 map, fallback용 mapping DataFrame을 로드한다.
    """
    if not MAPPING_CSV_PATH.exists():
        raise FileNotFoundError(f"mapping CSV를 찾을 수 없습니다: {MAPPING_CSV_PATH}")

    if not OTC_CSV_PATH.exists():
        raise FileNotFoundError(f"OTC CSV를 찾을 수 없습니다: {OTC_CSV_PATH}")

    mapping_df = pd.read_csv(
        MAPPING_CSV_PATH,
        encoding="utf-8-sig",
        comment="#",
    )

    otc_df = pd.read_csv(
        OTC_CSV_PATH,
        encoding="utf-8-sig",
    )

    if "candidate_label" not in mapping_df.columns:
        raise ValueError(
            f"mapping CSV에 candidate_label 컬럼이 없습니다. 현재: {mapping_df.columns.tolist()}"
        )

    required_otc = {
        "disease_name",
        "ingredient_1_name",
        "ingredient_2_name",
        "ingredient_3_name",
    }

    missing_otc = required_otc - set(otc_df.columns)

    if missing_otc:
        raise ValueError(f"OTC CSV에 필요한 컬럼이 없습니다: {missing_otc}")

    mapping_df["candidate_label"] = mapping_df["candidate_label"].apply(safe_str)

    if "user_input" in mapping_df.columns:
        mapping_df["user_input"] = mapping_df["user_input"].apply(safe_str)
    elif "step_4_symptom" in mapping_df.columns:
        mapping_df["user_input"] = mapping_df["step_4_symptom"].apply(safe_str)
    else:
        mapping_df["user_input"] = ""

    for col in ("candidate_department", "candidate_severity"):
        if col in mapping_df.columns:
            mapping_df[col] = mapping_df[col].apply(safe_str)
        else:
            mapping_df[col] = ""

    if "candidate_rank" in mapping_df.columns:
        mapping_df["candidate_rank"] = mapping_df["candidate_rank"].apply(
            lambda x: safe_int(x, 999)
        )
    else:
        mapping_df["candidate_rank"] = 999

    if "candidate_score" in mapping_df.columns:
        mapping_df["candidate_score"] = mapping_df["candidate_score"].apply(
            lambda x: safe_float(x, 0.0)
        )
    else:
        mapping_df["candidate_score"] = 0.0

    for col in [
        "step_1_main_area",
        "step_2_detailed_region",
        "step_3_sub_region",
        "step_4_symptom",
        "step_5_pain_range_code",
        "step_5_pain_range",
        "step_7_onset_time_code",
        "step_7_onset_time",
    ]:
        if col in mapping_df.columns:
            mapping_df[col] = mapping_df[col].apply(safe_str)

    otc_df["disease_name"] = otc_df["disease_name"].apply(safe_str)

    joined = mapping_df.merge(
        otc_df,
        how="left",
        left_on="candidate_label",
        right_on="disease_name",
    )

    allowed_labels = {
        label
        for label in mapping_df["candidate_label"]
        .dropna()
        .astype(str)
        .str.strip()
        .tolist()
        if label and label.lower() != "nan"
    }

    medicine_map: Dict[str, str] = {}

    for _, row in joined.iterrows():
        disease_name = safe_str(row.get("candidate_label", ""))

        if disease_name and disease_name not in medicine_map:
            medicine_map[disease_name] = build_recommended_medicines(row)

    fallback_df = mapping_df[
        (mapping_df["candidate_label"] != "") &
        (mapping_df["candidate_label"].str.lower() != "nan")
    ].reset_index(drop=True)

    return allowed_labels, medicine_map, fallback_df


ALLOWED_LABELS, MEDICINE_MAP, FALLBACK_DF = load_allowed_data()

print(f"[MediQ v3.3-free] 허용 질환 수: {len(ALLOWED_LABELS)}")
print(f"[MediQ v3.3-free] fallback 후보 수: {len(FALLBACK_DF)}")


# ============================================================
# 6. 후보 추출
# ============================================================

def sort_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """
    candidate_rank 오름차순, candidate_score 내림차순 정렬.
    """
    if df.empty:
        return df

    if "candidate_rank" not in df.columns:
        df["candidate_rank"] = 999

    if "candidate_score" not in df.columns:
        df["candidate_score"] = 0.0

    return df.sort_values(
        by=["candidate_rank", "candidate_score"],
        ascending=[True, False],
    )


def extract_candidates(
    input_json: Dict[str, Any],
    top_k: int = TOP_K_CANDIDATES,
) -> Tuple[pd.DataFrame, str]:
    """
    입력 JSON에 가까운 후보 행과 tier를 반환한다.

    tier:
    - exact: 부위 3단계 + 증상 정확 일치
    - region3: 부위 3단계 일치
    - region2: 부위 2단계 일치
    - symptom: 증상 문장 부분 일치
    - global: 전체 fallback
    """
    main_area = safe_str(input_json.get("step_1_main_area", ""))
    detailed_region = safe_str(input_json.get("step_2_detailed_region", ""))
    sub_region = safe_str(input_json.get("step_3_sub_region", ""))
    symptom_text = safe_str(input_json.get("step_4_symptom", ""))

    df = FALLBACK_DF.copy()
    matched = pd.DataFrame()
    tier = TIER_GLOBAL

    cols = [
        "step_1_main_area",
        "step_2_detailed_region",
        "step_3_sub_region",
        "step_4_symptom",
    ]

    if symptom_text and all(c in df.columns for c in cols):
        matched = df[
            (df["step_1_main_area"].astype(str).str.strip() == main_area) &
            (df["step_2_detailed_region"].astype(str).str.strip() == detailed_region) &
            (df["step_3_sub_region"].astype(str).str.strip() == sub_region) &
            (df["step_4_symptom"].astype(str).str.strip() == symptom_text)
        ].copy()

        if not matched.empty:
            tier = TIER_EXACT

    if matched.empty:
        cols = [
            "step_1_main_area",
            "step_2_detailed_region",
            "step_3_sub_region",
        ]

        if all(c in df.columns for c in cols):
            matched = df[
                (df["step_1_main_area"].astype(str).str.strip() == main_area) &
                (df["step_2_detailed_region"].astype(str).str.strip() == detailed_region) &
                (df["step_3_sub_region"].astype(str).str.strip() == sub_region)
            ].copy()

            if not matched.empty:
                tier = TIER_REGION3

    if matched.empty:
        cols = [
            "step_1_main_area",
            "step_2_detailed_region",
        ]

        if all(c in df.columns for c in cols):
            matched = df[
                (df["step_1_main_area"].astype(str).str.strip() == main_area) &
                (df["step_2_detailed_region"].astype(str).str.strip() == detailed_region)
            ].copy()

            if not matched.empty:
                tier = TIER_REGION2

    if matched.empty and symptom_text and "user_input" in df.columns:
        matched = df[
            df["user_input"].astype(str).str.contains(
                re.escape(symptom_text),
                na=False,
                regex=True,
            )
        ].copy()

        if not matched.empty:
            tier = TIER_SYMPTOM

    if matched.empty:
        matched = df.copy()
        tier = TIER_GLOBAL

    matched = sort_candidates(matched)
    matched = matched.drop_duplicates(subset=["candidate_label"], keep="first")

    return matched.head(top_k).reset_index(drop=True), tier


def rows_to_pairs(rows: pd.DataFrame) -> List[Tuple[str, float]]:
    """
    후보 DataFrame을 (질환명, CSV score) 튜플 리스트로 변환한다.
    """
    pairs: List[Tuple[str, float]] = []

    for _, row in rows.iterrows():
        label = safe_str(row.get("candidate_label", ""))

        if label and label in ALLOWED_LABELS:
            pairs.append(
                (
                    label,
                    safe_float(row.get("candidate_score", 0.0), 0.0),
                )
            )

    if not pairs:
        pairs = [(sorted(ALLOWED_LABELS)[0], 0.0)]

    return pairs


def get_candidate_disease_names(
    input_json: Dict[str, Any],
    top_k: int = TOP_K_CANDIDATES,
) -> List[str]:
    rows, _ = extract_candidates(input_json, top_k=top_k)
    return [candidate for candidate, _ in rows_to_pairs(rows)]


# ============================================================
# 7. 결과 정규화
# ============================================================

def make_result_from_disease_name(disease_name: str) -> Dict[str, str]:
    """
    질환명을 최종 AiAnalysisResult 형태로 변환한다.
    """
    disease_name = safe_str(disease_name)

    department, risk_level = infer_department_and_risk(disease_name)

    matched = FALLBACK_DF[FALLBACK_DF["candidate_label"] == disease_name]

    if not matched.empty:
        row = matched.iloc[0]

        csv_department = safe_str(row.get("candidate_department", ""))
        csv_risk = safe_str(row.get("candidate_severity", ""))

        if csv_department:
            department = csv_department

        if csv_risk:
            risk_level = csv_risk

    recommended = safe_str(
        MEDICINE_MAP.get(disease_name)
        or DEFAULT_RECOMMENDED_MEDICINES
    )

    return {
        "diseaseName": disease_name,
        "department": department,
        "riskLevel": risk_level,
        "recommendedMedicines": recommended,
    }


def fallback_by_csv(
    input_json: Dict[str, Any],
    candidates: Optional[List[str]] = None,
) -> Dict[str, str]:
    if not candidates:
        candidates = get_candidate_disease_names(
            input_json,
            top_k=TOP_K_CANDIDATES,
        )

    disease_name = safe_str(candidates[0])

    if disease_name not in ALLOWED_LABELS:
        disease_name = sorted(ALLOWED_LABELS)[0]

    print(f"[MediQ v3.3-free] CSV fallback diseaseName: {disease_name}")

    return make_result_from_disease_name(disease_name)


# ============================================================
# 8. 모델 로드
# ============================================================

def load_model() -> Tuple[Any, Any]:
    if not ADAPTER_OUTPUT_DIR.exists():
        raise FileNotFoundError(
            f"LoRA adapter 폴더를 찾을 수 없습니다: {ADAPTER_OUTPUT_DIR}\n"
            "먼저 v3 adapter를 학습/저장해야 합니다."
        )

    model_cache_path = snapshot_download(
        repo_id=MODEL_NAME,
        cache_dir=str(CACHE_DIR),
        local_files_only=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        str(ADAPTER_OUTPUT_DIR),
        trust_remote_code=True,
        use_fast=True,
        local_files_only=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    compute_dtype = torch.float16

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )

    torch.cuda.empty_cache()
    gc.collect()

    base_model = AutoModelForCausalLM.from_pretrained(
        model_cache_path,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True,
        torch_dtype=compute_dtype,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )

    model = PeftModel.from_pretrained(
        base_model,
        str(ADAPTER_OUTPUT_DIR),
        local_files_only=True,
    )

    model.eval()

    return tokenizer, model


tokenizer, model = load_model()


# ============================================================
# 9. 모호 케이스 전용: 하이브리드 스코어링
# ============================================================

def build_scoring_prompt(
    input_json: Dict[str, Any],
    candidates: List[str],
) -> str:
    payload = {
        "symptomReport": input_json,
        "allowedDiseaseCandidates": candidates,
        "instruction": (
            "아래 allowedDiseaseCandidates 중에서 환자 증상에 가장 부합하는 질환명 하나를 고른다. "
            "목록에 없는 질환명, 증상문장, 부위명은 절대 사용하지 않는다."
        ),
    }

    return json.dumps(payload, ensure_ascii=False)


def _softmax(values: List[float], temp: float = 1.0) -> List[float]:
    if not values:
        return []

    temperature = max(temp, 1e-6)
    scaled = [value / temperature for value in values]
    max_value = max(scaled)
    exps = [math.exp(value - max_value) for value in scaled]
    total = sum(exps) or 1.0

    return [value / total for value in exps]


@torch.no_grad()
def model_logprobs(
    input_json: Dict[str, Any],
    candidates: List[str],
) -> List[float]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_scoring_prompt(input_json, candidates)},
    ]

    try:
        prefix_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        prefix_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    prefix_text += '{"diseaseName": "'

    prefix_ids = tokenizer(
        prefix_text,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids.to(model.device)

    prefix_len = prefix_ids.shape[1]

    logps: List[float] = []

    for candidate in candidates:
        candidate_ids = tokenizer(
            candidate + '"',
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids.to(model.device)

        full_ids = torch.cat([prefix_ids, candidate_ids], dim=1)

        logits = model(full_ids).logits

        total_logprob = 0.0
        token_count = 0

        for position in range(prefix_len, full_ids.shape[1]):
            logprobs = F.log_softmax(logits[0, position - 1], dim=-1)
            total_logprob += logprobs[full_ids[0, position]].item()
            token_count += 1

        score = (
            total_logprob / max(token_count, 1)
            if LENGTH_NORMALIZE
            else total_logprob
        )

        logps.append(score)

    return logps


def select_best_candidate(
    input_json: Dict[str, Any],
    pairs: List[Tuple[str, float]],
    is_free_text: bool = False,
) -> Tuple[str, List[Tuple[str, float, float, float]], float]:
    """
    CSV prior와 모델 로그우도를 혼합하여 최종 후보를 선택한다.

    is_free_text=True일 때:
    - CSV prior 비중을 0.70 → 0.55로 낮춘다.
    - 모델이 CSV 1순위를 뒤집기 위한 margin을 0.10 → 0.03으로 낮춘다.
    """
    candidates = [candidate for candidate, _ in pairs]
    csv_scores = [score for _, score in pairs]

    p_csv = _softmax(csv_scores, temp=CSV_SCORE_TEMP)

    logps = model_logprobs(input_json, candidates)
    p_model = _softmax(logps, temp=MODEL_SCORE_TEMP)

    alpha = ALPHA_CSV_PRIOR_FREETEXT if is_free_text else ALPHA_CSV_PRIOR
    margin = MODEL_OVERRIDE_MARGIN_FREETEXT if is_free_text else MODEL_OVERRIDE_MARGIN

    finals: List[Tuple[str, float, float, float]] = []

    for index, candidate in enumerate(candidates):
        final_score = (
            alpha * p_csv[index]
            + (1.0 - alpha) * p_model[index]
        )

        finals.append(
            (
                candidate,
                p_csv[index],
                p_model[index],
                final_score,
            )
        )

    csv_top_label = candidates[0]
    csv_top_final = next(
        final_score
        for candidate, _, _, final_score in finals
        if candidate == csv_top_label
    )

    ranked = sorted(finals, key=lambda x: x[3], reverse=True)

    model_best_label, _, _, model_best_final = ranked[0]

    # 모델이 CSV 1순위를 마진 미만으로만 이기면 CSV 1순위 유지
    if (
        model_best_label != csv_top_label
        and (model_best_final - csv_top_final) < margin
    ):
        return csv_top_label, ranked, alpha

    return model_best_label, ranked, alpha


# ============================================================
# 10. 추론 오케스트레이션
# ============================================================

def analyze_constrained(input_json: Dict[str, Any]) -> Dict[str, str]:
    symptom_log = mask_pii(input_json.get("step_4_symptom", ""))

    if detect_red_flag(input_json):
        print(f"[MediQ v3.3-free] Red-flag 감지 → 응급 우회 (증상: {symptom_log})")
        return dict(EMERGENCY_RESULT)

    is_free_text = bool(input_json.get("is_free_text", False))

    rows, tier = extract_candidates(
        input_json,
        top_k=TOP_K_CANDIDATES,
    )

    pairs = rows_to_pairs(rows)
    candidates = [candidate for candidate, _ in pairs]

    print(
        f"[MediQ v3.3-free] tier={tier} free_text={is_free_text} "
        f"후보({len(candidates)})={candidates} / 증상={symptom_log}"
    )

    # exact match는 CSV 1순위 즉시 선택
    if tier == TIER_EXACT:
        chosen = candidates[0]
        print(f"[MediQ v3.3-free] exact match → CSV 1순위 즉시 선택: {chosen}")
        return make_result_from_disease_name(chosen)

    # 후보가 1개면 모델 호출 없이 선택
    if len(candidates) == 1:
        chosen = candidates[0]
        print(f"[MediQ v3.3-free] 단일 후보 선택: {chosen}")
        return make_result_from_disease_name(chosen)

    try:
        best, ranked, alpha = select_best_candidate(
            input_json,
            pairs,
            is_free_text=is_free_text,
        )

        detail = ", ".join(
            f"{candidate}(csv:{p_csv:.2f}/llm:{p_model:.2f}->{final_score:.2f})"
            for candidate, p_csv, p_model, final_score in ranked
        )

        print(
            f"[MediQ v3.3-free] 하이브리드[{tier}] "
            f"α={alpha:.2f}(free_text={is_free_text}) {detail} -> 선택: {best}"
        )

        # [FIXED] 기존 오타: make_result_from_disease_name(최고)
        return make_result_from_disease_name(best)

    except Exception as error:
        print(
            f"[MediQ v3.3-free] 스코어링 실패({type(error).__name__}: {error}) "
            "-> CSV fallback"
        )
        return fallback_by_csv(input_json, candidates)


# ============================================================
# 11. FastAPI 앱
# ============================================================

app = FastAPI(
    title="MediQ AI Inference Server",
    version="3.3-free-text-aware",
)


@app.get("/")
def health_check() -> Dict[str, str]:
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "adapter": str(ADAPTER_OUTPUT_DIR),
        "allowedDiseaseCount": str(len(ALLOWED_LABELS)),
        "fallbackCandidateCount": str(len(FALLBACK_DF)),
        "inferenceMode": "csv_first_tiered_then_hybrid_free_text_aware",
        "alphaCsvPrior": str(ALPHA_CSV_PRIOR),
        "alphaCsvPriorFreeText": str(ALPHA_CSV_PRIOR_FREETEXT),
        "safetyFirst": "enabled",
    }


@app.post("/analyze", response_model=AiAnalysisResult)
def analyze(request: AiAnalysisRequest) -> Dict[str, str]:
    try:
        input_json = request.model_dump()
        return analyze_constrained(input_json)

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"AI inference failed: {type(error).__name__}",
        )