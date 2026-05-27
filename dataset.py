import json
from typing import Any, Dict, List, Tuple

import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split

from config import (
    MAPPING_CSV_PATH,
    OTC_CSV_PATH,
    DEFAULT_RECOMMENDED_MEDICINES,
    RANDOM_SEED,
)


SYSTEM_PROMPT = """
너는 MediQ의 한국어 의료 보조 문진 AI이다.
사용자의 구조화된 증상 JSON을 분석하여 서비스 허용 질환명, 진료과, 위험도, 추천 일반의약품 성분을 JSON으로 반환한다.

반드시 지켜야 할 규칙:
1. 확정 진단을 하지 않는다.
2. 응답은 반드시 JSON 객체 하나만 출력한다.
3. Markdown, 설명문, 코드블록을 출력하지 않는다.
4. 새로운 질환명을 임의로 만들지 않는다.
5. diseaseName은 반드시 학습 데이터의 candidate_label에 존재하는 질환명만 사용한다.
6. step_4_symptom, user_input, 증상 문장, 통증 표현을 diseaseName에 복사하면 안 된다.
7. diseaseName은 증상 설명이 아니라 질환명이어야 한다.
8. 응답 JSON 키는 diseaseName, department, riskLevel, recommendedMedicines 네 개만 사용한다.
""".strip()


DISEASE_RULES: Dict[str, Dict[str, str]] = {
    "긴장성 두통": {"department": "신경과", "riskLevel": "일반"},
    "편두통": {"department": "신경과", "riskLevel": "일반"},
    "군발두통": {"department": "신경과", "riskLevel": "주의"},
    "부비동염": {"department": "이비인후과", "riskLevel": "일반"},
    "부비동염(축농증)": {"department": "이비인후과", "riskLevel": "일반"},
    "비염": {"department": "이비인후과", "riskLevel": "일반"},
    "감기": {"department": "내과", "riskLevel": "일반"},
    "상기도 감염": {"department": "이비인후과", "riskLevel": "일반"},
    "인후염": {"department": "이비인후과", "riskLevel": "일반"},
    "편도염": {"department": "이비인후과", "riskLevel": "일반"},
    "소화불량": {"department": "소화기내과", "riskLevel": "일반"},
    "기능성 위장 장애": {"department": "소화기내과", "riskLevel": "일반"},
    "위염": {"department": "소화기내과", "riskLevel": "일반"},
    "식도염": {"department": "소화기내과", "riskLevel": "일반"},
    "역류성 식도염": {"department": "소화기내과", "riskLevel": "일반"},
    "장염": {"department": "소화기내과", "riskLevel": "일반"},
    "과민성 대장 증후군": {"department": "소화기내과", "riskLevel": "일반"},
    "변비": {"department": "소화기내과", "riskLevel": "일반"},
    "방광염": {"department": "비뇨의학과", "riskLevel": "일반"},
    "근육통": {"department": "정형외과", "riskLevel": "일반"},
    "염좌": {"department": "정형외과", "riskLevel": "일반"},
    "요통": {"department": "정형외과", "riskLevel": "일반"},
    "결막염": {"department": "안과", "riskLevel": "일반"},
    "안구건조증": {"department": "안과", "riskLevel": "일반"},
    "피부염": {"department": "피부과", "riskLevel": "일반"},
    "두드러기": {"department": "피부과", "riskLevel": "일반"},
    "여드름": {"department": "피부과", "riskLevel": "일반"},
    "월경통": {"department": "산부인과", "riskLevel": "일반"},
    "질염": {"department": "산부인과", "riskLevel": "일반"},
    "치아 우식증[충치]": {"department": "치과", "riskLevel": "일반"},
    "심계항진": {"department": "순환기내과/심장내과", "riskLevel": "일반"},
}


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


def safe_int(value: Any, default: int = 5) -> int:
    """
    NaN, None, 잘못된 숫자 값을 안전한 정수로 변환한다.
    """
    try:
        if pd.isna(value):
            return default
        return int(float(value))
    except Exception:
        return default


def infer_department_and_risk(disease_name: str) -> Tuple[str, str]:
    """
    질환명 기반으로 진료과와 위험도를 결정적으로 추론한다.
    """
    disease_name = safe_str(disease_name)

    if disease_name in DISEASE_RULES:
        rule = DISEASE_RULES[disease_name]
        return rule["department"], rule["riskLevel"]

    if "두통" in disease_name or "편두통" in disease_name:
        return "신경과", "일반"
    if "위" in disease_name or "장" in disease_name or "소화" in disease_name:
        return "소화기내과", "일반"
    if "식도" in disease_name or "명치" in disease_name:
        return "소화기내과", "일반"
    if "비염" in disease_name or "인후" in disease_name or "편도" in disease_name:
        return "이비인후과", "일반"
    if "감염" in disease_name:
        return "이비인후과", "일반"
    if "피부" in disease_name or "두드러기" in disease_name:
        return "피부과", "일반"
    if "눈" in disease_name or "결막" in disease_name or "안구" in disease_name:
        return "안과", "일반"
    if "방광" in disease_name or "소변" in disease_name:
        return "비뇨의학과", "일반"
    if "월경" in disease_name or "질염" in disease_name:
        return "산부인과", "일반"
    if "근육" in disease_name or "염좌" in disease_name or "요통" in disease_name:
        return "정형외과", "일반"
    if "치아" in disease_name or "충치" in disease_name:
        return "치과", "일반"
    if "심계항진" in disease_name or "심장" in disease_name:
        return "순환기내과/심장내과", "일반"

    return "내과", "일반"


def build_recommended_medicines(row: pd.Series) -> str:
    """
    OTC CSV의 ingredient 컬럼을 조합해 recommendedMedicines 문자열을 만든다.
    """
    ingredients: List[str] = []

    for col in ["ingredient_1_name", "ingredient_2_name", "ingredient_3_name"]:
        value = safe_str(row.get(col, ""))
        if value:
            ingredients.append(value)

    if not ingredients:
        return DEFAULT_RECOMMENDED_MEDICINES

    return ", ".join(ingredients)


def looks_like_symptom_sentence(text: str) -> bool:
    """
    candidate_label이 질환명이 아니라 증상 문장처럼 보이는지 검사한다.
    너무 공격적으로 지우면 정상 질환명도 날아갈 수 있으므로 보수적으로 판단한다.
    """
    text = safe_str(text)

    if not text:
        return True

    symptom_endings = [
        "아파요",
        "아픕니다",
        "아프다",
        "쑤셔요",
        "쑤십니다",
        "답답해요",
        "답답합니다",
        "불편해요",
        "불편합니다",
        "같아요",
        "느낌",
        "통증",
        "시려요",
        "따가워요",
        "가려워요",
        "막혀요",
        "나와요",
        "생겨요",
    ]

    if any(ending in text for ending in symptom_endings):
        return True

    # 질환명은 보통 짧다. 공백이 많은 긴 문장은 증상 설명일 가능성이 높다.
    if text.count(" ") >= 3:
        return True

    # 너무 긴 label도 증상 설명일 가능성이 높다.
    if len(text) >= 25:
        return True

    return False


def clean_mapping_rows(mapping_df: pd.DataFrame) -> pd.DataFrame:
    """
    재학습 전 학습 데이터를 정제한다.

    제거 조건:
    1. candidate_label이 비어 있는 행
    2. user_input과 candidate_label이 동일한 행
    3. step_4_symptom과 candidate_label이 동일한 행
    4. candidate_label이 증상 문장처럼 보이는 행
    """
    before = len(mapping_df)

    mapping_df = mapping_df.copy()

    mapping_df["candidate_label"] = mapping_df["candidate_label"].apply(safe_str)

    if "user_input" in mapping_df.columns:
        mapping_df["user_input"] = mapping_df["user_input"].apply(safe_str)
    elif "step_4_symptom" in mapping_df.columns:
        mapping_df["user_input"] = mapping_df["step_4_symptom"].apply(safe_str)
    else:
        mapping_df["user_input"] = ""

    if "step_4_symptom" in mapping_df.columns:
        mapping_df["step_4_symptom"] = mapping_df["step_4_symptom"].apply(safe_str)
    else:
        mapping_df["step_4_symptom"] = ""

    # 1. 빈 candidate_label 제거
    mapping_df = mapping_df[
        (mapping_df["candidate_label"] != "") &
        (mapping_df["candidate_label"].str.lower() != "nan")
    ]

    # 2. user_input과 candidate_label이 동일한 행 제거
    mapping_df = mapping_df[
        mapping_df["user_input"].str.strip() != mapping_df["candidate_label"].str.strip()
    ]

    # 3. step_4_symptom과 candidate_label이 동일한 행 제거
    mapping_df = mapping_df[
        mapping_df["step_4_symptom"].str.strip() != mapping_df["candidate_label"].str.strip()
    ]

    # 4. candidate_label이 증상 문장처럼 보이는 행 제거
    mapping_df = mapping_df[
        ~mapping_df["candidate_label"].apply(looks_like_symptom_sentence)
    ]

    after = len(mapping_df)

    print(f"[MediQ v3] mapping 정제 전: {before}")
    print(f"[MediQ v3] mapping 정제 후: {after}")
    print(f"[MediQ v3] 제거된 행 수: {before - after}")

    return mapping_df.reset_index(drop=True)


def load_and_join_data() -> pd.DataFrame:
    """
    mapping CSV와 OTC CSV를 로드한 뒤
    candidate_label == disease_name 기준으로 left join한다.

    mediq_mapping_reviewed.csv 상단에 #로 시작하는 설명/주석 줄이 있을 수 있으므로
    comment="#" 옵션으로 주석 줄을 무시한다.

    CSV에 user_input 컬럼이 없고 step_4_symptom 컬럼이 있는 경우,
    step_4_symptom을 user_input처럼 사용한다.

    재학습 안정화를 위해 candidate_label이 증상 문장처럼 보이는 행은 제거한다.
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

    required_mapping_cols = {"candidate_label"}
    required_otc_cols = {
        "disease_name",
        "ingredient_1_name",
        "ingredient_2_name",
        "ingredient_3_name",
    }

    missing_mapping = required_mapping_cols - set(mapping_df.columns)
    missing_otc = required_otc_cols - set(otc_df.columns)

    if missing_mapping:
        raise ValueError(
            f"mapping CSV에 필요한 컬럼이 없습니다: {missing_mapping}\n"
            f"현재 컬럼: {mapping_df.columns.tolist()}"
        )

    if missing_otc:
        raise ValueError(
            f"OTC CSV에 필요한 컬럼이 없습니다: {missing_otc}\n"
            f"현재 컬럼: {otc_df.columns.tolist()}"
        )

    # user_input 자동 생성 및 candidate_label 정제
    mapping_df = clean_mapping_rows(mapping_df)

    otc_df["disease_name"] = otc_df["disease_name"].apply(safe_str)

    joined = mapping_df.merge(
        otc_df,
        how="left",
        left_on="candidate_label",
        right_on="disease_name",
    )

    joined = joined[
        (joined["user_input"] != "") &
        (joined["candidate_label"] != "")
    ].reset_index(drop=True)

    print(f"[MediQ v3] OTC join 후 데이터 수: {len(joined)}")

    return joined


def build_input_json(row: pd.Series, report_id: int) -> Dict[str, Any]:
    """
    Spring Boot AiAnalysisRequest DTO 형태의 입력 JSON을 생성한다.

    CSV에 실제 7단계 컬럼이 있으면 해당 값을 사용하고,
    없는 경우 기본값으로 채운다.
    """
    user_input = safe_str(row.get("user_input", ""))

    return {
        "reportId": report_id,
        "step_1_main_area": safe_str(row.get("step_1_main_area", ""), "증상"),
        "step_2_detailed_region": safe_str(row.get("step_2_detailed_region", ""), "미상"),
        "step_3_sub_region": safe_str(row.get("step_3_sub_region", ""), "미상"),
        "step_4_symptom": user_input,
        "is_free_text": True,
        "step_5_pain_range_code": safe_str(row.get("step_5_pain_range_code", ""), "UNKNOWN"),
        "step_5_pain_range": safe_str(row.get("step_5_pain_range", ""), "미상"),
        "step_6_intensity_value": safe_int(row.get("step_6_intensity_value", 5), 5),
        "step_7_onset_time_code": safe_str(row.get("step_7_onset_time_code", ""), "UNKNOWN"),
        "step_7_onset_time": safe_str(row.get("step_7_onset_time", ""), "미상"),
    }


def build_output_json(row: pd.Series) -> Dict[str, str]:
    """
    Spring Boot AiAnalysisResult entity 형태의 출력 JSON을 생성한다.
    """
    disease_name = safe_str(row.get("candidate_label", ""))
    department, risk_level = infer_department_and_risk(disease_name)
    recommended_medicines = build_recommended_medicines(row)

    # CSV에 candidate_department / candidate_severity가 있으면 우선 사용
    candidate_department = safe_str(row.get("candidate_department", ""))
    candidate_severity = safe_str(row.get("candidate_severity", ""))

    if candidate_department:
        department = candidate_department

    if candidate_severity:
        risk_level = candidate_severity

    return {
        "diseaseName": disease_name,
        "department": department,
        "riskLevel": risk_level,
        "recommendedMedicines": recommended_medicines,
    }


def build_messages(row: pd.Series, report_id: int) -> Dict[str, Any]:
    """
    한 행을 Qwen3 instruction tuning용 messages 구조로 변환한다.
    """
    input_json = build_input_json(row, report_id)
    output_json = build_output_json(row)

    # 최종 안전 검증: diseaseName이 입력 증상과 같으면 학습 샘플로 쓰지 않음
    if safe_str(output_json["diseaseName"]) == safe_str(input_json["step_4_symptom"]):
        raise ValueError(
            f"잘못된 학습 샘플: diseaseName이 step_4_symptom과 같습니다. {output_json['diseaseName']}"
        )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(input_json, ensure_ascii=False)},
        {"role": "assistant", "content": json.dumps(output_json, ensure_ascii=False)},
    ]

    return {
        "messages": messages,
        "diseaseName": output_json["diseaseName"],
        "input_json": input_json,
        "output_json": output_json,
    }


def build_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    전체 DataFrame을 instruction tuning record 리스트로 변환한다.
    """
    records: List[Dict[str, Any]] = []

    skipped = 0

    for idx, row in df.iterrows():
        try:
            records.append(build_messages(row, idx + 1))
        except ValueError:
            skipped += 1

    print(f"[MediQ v3] records 생성 수: {len(records)}")
    print(f"[MediQ v3] records 제외 수: {skipped}")

    return records


def split_records(
    records: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    records를 80/10/10 train/validation/test로 나눈다.
    diseaseName 기준 stratify를 시도하고, 불가능하면 일반 split으로 fallback한다.
    """
    labels = [r["diseaseName"] for r in records]

    try:
        train_records, temp_records = train_test_split(
            records,
            test_size=0.2,
            random_state=RANDOM_SEED,
            stratify=labels,
        )

        temp_labels = [r["diseaseName"] for r in temp_records]

        valid_records, test_records = train_test_split(
            temp_records,
            test_size=0.5,
            random_state=RANDOM_SEED,
            stratify=temp_labels,
        )

    except ValueError:
        train_records, temp_records = train_test_split(
            records,
            test_size=0.2,
            random_state=RANDOM_SEED,
        )

        valid_records, test_records = train_test_split(
            temp_records,
            test_size=0.5,
            random_state=RANDOM_SEED,
        )

    return train_records, valid_records, test_records


def records_to_dataset(records: List[Dict[str, Any]], tokenizer: Any) -> Dataset:
    """
    Qwen3 chat template을 적용해 Hugging Face Dataset으로 변환한다.
    """
    rows: List[Dict[str, str]] = []

    for record in records:
        text = tokenizer.apply_chat_template(
            record["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )

        rows.append(
            {
                "text": text,
                "diseaseName": record["diseaseName"],
            }
        )

    return Dataset.from_list(rows)