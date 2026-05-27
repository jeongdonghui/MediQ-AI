# -*- coding: utf-8 -*-

from pathlib import Path
from datetime import datetime

import pandas as pd


CSV_PATH = Path(r"D:\학교\졸작 데이터\MediQ_AI_v3\data\mediq_mapping_reviewed.csv")


TARGET_CONDITION = {
    "step_1_main_area": "목/가슴",
    "step_2_detailed_region": "가슴",
    "step_3_sub_region": "가슴 중앙",
    "step_4_symptom": "심장 박동이 비정상적인 것 같아요",
}

PREFERRED_LABEL = "심계항진"
LOWER_PRIORITY_LABEL = "부정맥"


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {CSV_PATH}")

    backup_path = CSV_PATH.with_name(
        f"{CSV_PATH.stem}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )

    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig", comment="#")

    print("전체 행 수:", len(df))
    print("컬럼 목록:", df.columns.tolist())

    required_cols = set(TARGET_CONDITION.keys()) | {"candidate_label"}

    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"필요한 컬럼이 없습니다: {missing_cols}")

    if "candidate_rank" not in df.columns:
        df["candidate_rank"] = 999

    if "candidate_score" not in df.columns:
        df["candidate_score"] = 0.0

    condition = pd.Series([True] * len(df), index=df.index)

    for col, value in TARGET_CONDITION.items():
        condition = condition & (df[col].astype(str).str.strip() == value)

    matched = df[condition].copy()

    print("\n[수정 전 후보]")
    if matched.empty:
        print("해당 조건과 일치하는 행이 없습니다.")
        return

    print(
        matched[
            [
                "step_1_main_area",
                "step_2_detailed_region",
                "step_3_sub_region",
                "step_4_symptom",
                "candidate_label",
                "candidate_rank",
                "candidate_score",
            ]
        ].to_string(index=True)
    )

    # 원본 백업
    df.to_csv(backup_path, index=False, encoding="utf-8-sig")
    print(f"\n백업 저장 완료: {backup_path}")

    # 심계항진을 1순위로 올림
    preferred_mask = condition & (
        df["candidate_label"].astype(str).str.strip() == PREFERRED_LABEL
    )

    # 부정맥은 그 아래 순위로 내림
    lower_mask = condition & (
        df["candidate_label"].astype(str).str.strip() == LOWER_PRIORITY_LABEL
    )

    if preferred_mask.sum() == 0:
        print(f"\n주의: {PREFERRED_LABEL} 행을 찾지 못했습니다.")
    else:
        df.loc[preferred_mask, "candidate_rank"] = 1
        df.loc[preferred_mask, "candidate_score"] = 1.0

    if lower_mask.sum() > 0:
        df.loc[lower_mask, "candidate_rank"] = 2
        df.loc[lower_mask, "candidate_score"] = 0.8

    df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")

    print("\n수정 저장 완료:", CSV_PATH)

    df_check = pd.read_csv(CSV_PATH, encoding="utf-8-sig", comment="#")

    check_condition = pd.Series([True] * len(df_check), index=df_check.index)

    for col, value in TARGET_CONDITION.items():
        check_condition = check_condition & (
            df_check[col].astype(str).str.strip() == value
        )

    checked = df_check[check_condition].copy()

    checked["candidate_rank"] = pd.to_numeric(
        checked["candidate_rank"], errors="coerce"
    ).fillna(999)

    checked["candidate_score"] = pd.to_numeric(
        checked["candidate_score"], errors="coerce"
    ).fillna(0.0)

    checked = checked.sort_values(
        by=["candidate_rank", "candidate_score"],
        ascending=[True, False],
    )

    print("\n[수정 후 후보]")
    print(
        checked[
            [
                "step_1_main_area",
                "step_2_detailed_region",
                "step_3_sub_region",
                "step_4_symptom",
                "candidate_label",
                "candidate_rank",
                "candidate_score",
            ]
        ].to_string(index=True)
    )


if __name__ == "__main__":
    main()