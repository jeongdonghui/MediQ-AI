# -*- coding: utf-8 -*-
"""
MediQ AI Inference Server v3.2
CSV-Priority + Constrained Scoring Edition

핵심 구조:
1. Red-flag 응급 의심 입력은 모델 추론 없이 즉시 응급 안내 반환
2. CSV에서 입력과 가장 가까운 candidate_label 후보 Top-K 추출
3. exact/body+symptom 매칭이면 CSV 1순위 후보를 우선 선택
4. exact 매칭이 약하거나 애매한 경우에만 Qwen 로그우도 스코어링 사용
5. 최종 응답은 항상 CSV/룰 기반 metadata로 구성
"""

import gc
import json
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

# 아래 매칭 단계에서는 모델 스코어링보다 CSV 1순위를 우선한다.
CSV_PRIORITY_MATCH_LEVELS = {
    "exact_all",
    "body3_symptom",
    "symptom_exact",
}


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

    if len(text) <= keep:
        return text + "***" if text else "(빈 입력)"

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

    if "candidate_department" in mapping_df.columns:
        mapping_df["candidate_department"] = mapping_df["candidate_department"].apply(safe_str)
    else:
        mapping_df["candidate_department"] = ""

    if "candidate_severity" in mapping_df.columns:
        mapping_df["candidate_severity"] = mapping_df["candidate_severity"].apply(safe_str)
    else:
        mapping_df["candidate_severity"] = ""

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

print(f"[MediQ v3.2] 허용 질환 수: {len(ALLOWED_LABELS)}")
print(f"[MediQ v3.2] fallback 후보 수: {len(FALLBACK_DF)}")


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


def filter_exact_columns(
    df: pd.DataFrame,
    conditions: Dict[str, str],
) -> pd.DataFrame:
    """
    조건 dict에 들어온 컬럼이 모두 존재하고 값이 있을 때 exact match를 수행한다.
    """
    available = {
        col: val
        for col, val in conditions.items()
        if col in df.columns and safe_str(val)
    }

    if not available:
        return pd.DataFrame()

    condition = pd.Series([True] * len(df), index=df.index)

    for col, val in available.items():
        condition = condition & (df[col].astype(str).str.strip() == safe_str(val))

    return df[condition].copy()


def get_candidate_rows_with_level(
    input_json: Dict[str, Any],
    top_k: int = TOP_K_CANDIDATES,
) -> Tuple[pd.DataFrame, str]:
    """
    입력 JSON과 가장 가까운 후보 행 및 match_level을 반환한다.

    match_level:
    - exact_all: 주요 7단계 필드 exact
    - body3_symptom: 부위 3단계 + 증상 exact
    - body3: 부위 3단계 exact
    - body2: 부위 2단계 exact
    - symptom_exact: 증상 문장 exact
    - symptom_contains: 증상 문장 부분 포함
    - global: 전체 fallback
    """
    main_area = safe_str(input_json.get("step_1_main_area", ""))
    detailed_region = safe_str(input_json.get("step_2_detailed_region", ""))
    sub_region = safe_str(input_json.get("step_3_sub_region", ""))
    symptom_text = safe_str(input_json.get("step_4_symptom", ""))
    pain_range_code = safe_str(input_json.get("step_5_pain_range_code", ""))
    pain_range = safe_str(input_json.get("step_5_pain_range", ""))
    onset_code = safe_str(input_json.get("step_7_onset_time_code", ""))
    onset_time = safe_str(input_json.get("step_7_onset_time", ""))

    df = FALLBACK_DF.copy()

    # 1. 주요 필드 exact match
    exact_all = filter_exact_columns(
        df,
        {
            "step_1_main_area": main_area,
            "step_2_detailed_region": detailed_region,
            "step_3_sub_region": sub_region,
            "step_4_symptom": symptom_text,
            "step_5_pain_range_code": pain_range_code,
            "step_5_pain_range": pain_range,
            "step_7_onset_time_code": onset_code,
            "step_7_onset_time": onset_time,
        },
    )

    if not exact_all.empty:
        matched = sort_candidates(exact_all)
        matched = matched.drop_duplicates(subset=["candidate_label"], keep="first")
        return matched.head(top_k).reset_index(drop=True), "exact_all"

    # 2. 부위 3단계 + 증상 exact
    body3_symptom = filter_exact_columns(
        df,
        {
            "step_1_main_area": main_area,
            "step_2_detailed_region": detailed_region,
            "step_3_sub_region": sub_region,
            "step_4_symptom": symptom_text,
        },
    )

    if not body3_symptom.empty:
        matched = sort_candidates(body3_symptom)
        matched = matched.drop_duplicates(subset=["candidate_label"], keep="first")
        return matched.head(top_k).reset_index(drop=True), "body3_symptom"

    # 3. 부위 3단계 exact
    body3 = filter_exact_columns(
        df,
        {
            "step_1_main_area": main_area,
            "step_2_detailed_region": detailed_region,
            "step_3_sub_region": sub_region,
        },
    )

    if not body3.empty:
        matched = sort_candidates(body3)
        matched = matched.drop_duplicates(subset=["candidate_label"], keep="first")
        return matched.head(top_k).reset_index(drop=True), "body3"

    # 4. 부위 2단계 exact
    body2 = filter_exact_columns(
        df,
        {
            "step_1_main_area": main_area,
            "step_2_detailed_region": detailed_region,
        },
    )

    if not body2.empty:
        matched = sort_candidates(body2)
        matched = matched.drop_duplicates(subset=["candidate_label"], keep="first")
        return matched.head(top_k).reset_index(drop=True), "body2"

    # 5. 증상 문장 exact
    if symptom_text:
        if "user_input" in df.columns:
            symptom_exact = df[
                df["user_input"].astype(str).str.strip() == symptom_text
            ].copy()

            if not symptom_exact.empty:
                matched = sort_candidates(symptom_exact)
                matched = matched.drop_duplicates(subset=["candidate_label"], keep="first")
                return matched.head(top_k).reset_index(drop=True), "symptom_exact"

        if "step_4_symptom" in df.columns:
            symptom_exact = df[
                df["step_4_symptom"].astype(str).str.strip() == symptom_text
            ].copy()

            if not symptom_exact.empty:
                matched = sort_candidates(symptom_exact)
                matched = matched.drop_duplicates(subset=["candidate_label"], keep="first")
                return matched.head(top_k).reset_index(drop=True), "symptom_exact"

    # 6. 증상 부분 포함
    if symptom_text and "user_input" in df.columns:
        symptom_contains = df[
            df["user_input"].astype(str).str.contains(
                re.escape(symptom_text),
                na=False,
                regex=True,
            )
        ].copy()

        if not symptom_contains.empty:
            matched = sort_candidates(symptom_contains)
            matched = matched.drop_duplicates(subset=["candidate_label"], keep="first")
            return matched.head(top_k).reset_index(drop=True), "symptom_contains"

    # 7. 전체 fallback
    matched = sort_candidates(df)
    matched = matched.drop_duplicates(subset=["candidate_label"], keep="first")

    return matched.head(top_k).reset_index(drop=True), "global"


def get_candidate_disease_names_with_level(
    input_json: Dict[str, Any],
    top_k: int = TOP_K_CANDIDATES,
) -> Tuple[List[str], str]:
    rows, match_level = get_candidate_rows_with_level(input_json, top_k=top_k)

    candidates = [
        safe_str(x)
        for x in rows["candidate_label"].tolist()
        if safe_str(x) in ALLOWED_LABELS
    ]

    if not candidates:
        candidates = [sorted(ALLOWED_LABELS)[0]]
        match_level = "global"

    return candidates, match_level


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
        candidates, _ = get_candidate_disease_names_with_level(
            input_json,
            top_k=TOP_K_CANDIDATES,
        )

    disease_name = safe_str(candidates[0])

    if disease_name not in ALLOWED_LABELS:
        disease_name = sorted(ALLOWED_LABELS)[0]

    print(f"[MediQ v3.2] CSV fallback diseaseName: {disease_name}")

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
# 9. 후보 우도 스코어링
# ============================================================

def build_scoring_prompt(
    input_json: Dict[str, Any],
    candidates: List[str],
) -> str:
    payload = {
        "symptomReport": input_json,
        "allowedDiseaseCandidates": candidates,
        "instruction": (
            "allowedDiseaseCandidates 중에서 환자 증상에 가장 부합하는 질환명 하나를 고른다. "
            "목록에 없는 질환명, 증상문장, 부위명은 절대 사용하지 않는다."
        ),
    }

    return json.dumps(payload, ensure_ascii=False)


@torch.no_grad()
def score_candidates(
    input_json: Dict[str, Any],
    candidates: List[str],
) -> Tuple[str, List[Tuple[str, float]]]:
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

    scored: List[Tuple[str, float]] = []

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

        scored.append((candidate, score))

    scored.sort(key=lambda x: x[1], reverse=True)

    best = scored[0][0]

    return best, scored


# ============================================================
# 10. 추론 오케스트레이션
# ============================================================

def analyze_constrained(input_json: Dict[str, Any]) -> Dict[str, str]:
    """
    1. Red-flag 우회
    2. CSV 후보 추출
    3. 정확 매칭이면 CSV 우선 선택
    4. 애매한 매칭이면 Qwen 스코어링 보조
    5. 실패 시 CSV fallback
    """
    if detect_red_flag(input_json):
        print(
            "[MediQ v3.2] Red-flag 감지 → 응급 우회 "
            f"(증상: {mask_pii(input_json.get('step_4_symptom', ''))})"
        )
        return dict(EMERGENCY_RESULT)

    candidates, match_level = get_candidate_disease_names_with_level(
        input_json,
        top_k=TOP_K_CANDIDATES,
    )

    print(
        f"[MediQ v3.2] 후보({len(candidates)}), match_level={match_level}: {candidates} / "
        f"입력증상: {mask_pii(input_json.get('step_4_symptom', ''))}"
    )

    if not candidates:
        return fallback_by_csv(input_json, candidates)

    # 핵심 변경:
    # 정확 매칭이 강한 경우에는 모델 스코어링보다 CSV 1순위 후보를 우선한다.
    if match_level in CSV_PRIORITY_MATCH_LEVELS:
        selected = candidates[0]
        print(f"[MediQ v3.2] CSV 우선 선택: {selected} / match_level={match_level}")
        return make_result_from_disease_name(selected)

    # 후보가 1개뿐이면 모델 호출 없이 확정
    if len(candidates) == 1:
        selected = candidates[0]
        print(f"[MediQ v3.2] 단일 후보 선택: {selected}")
        return make_result_from_disease_name(selected)

    # 애매한 경우에만 모델 스코어링 사용
    try:
        best, scored = score_candidates(input_json, candidates)

        ranked = ", ".join(
            f"{candidate}:{score:.3f}"
            for candidate, score in scored
        )

        print(f"[MediQ v3.2] 스코어링 순위: {ranked} → 선택: {best}")

        if best not in candidates or best not in ALLOWED_LABELS:
            print("[MediQ v3.2] 스코어링 결과 검증 실패 → CSV fallback")
            return fallback_by_csv(input_json, candidates)

        return make_result_from_disease_name(best)

    except Exception as e:
        print(f"[MediQ v3.2] 스코어링 실패({e}) → CSV fallback")
        return fallback_by_csv(input_json, candidates)


# ============================================================
# 11. FastAPI 앱
# ============================================================

app = FastAPI(
    title="MediQ AI Inference Server",
    version="3.2.0",
)


@app.get("/")
def health_check() -> Dict[str, str]:
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "adapter": str(ADAPTER_OUTPUT_DIR),
        "allowedDiseaseCount": str(len(ALLOWED_LABELS)),
        "fallbackCandidateCount": str(len(FALLBACK_DF)),
        "inferenceMode": "csv_priority_then_candidate_scoring",
        "safetyFirst": "enabled",
    }


@app.post("/analyze", response_model=AiAnalysisResult)
def analyze(request: AiAnalysisRequest) -> Dict[str, str]:
    """
    Spring Boot AiInferenceService에서 호출하는 AI 분석 엔드포인트.
    """
    try:
        input_json = request.model_dump()
        return analyze_constrained(input_json)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"AI inference failed: {type(e).__name__}",
        )