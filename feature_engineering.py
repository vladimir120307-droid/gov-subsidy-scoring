"""
Модуль извлечения признаков для скоринговой модели.
Формирует профиль каждого сельхозпроизводителя на основе истории заявок.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


def compute_producer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Вычисление агрегированных признаков для каждого производителя.
    Возвращает DataFrame с одной строкой на producer_id.
    """
    grouped = df.groupby("producer_id")

    features = pd.DataFrame()
    features["total_apps"] = grouped.size()
    features["region"] = grouped["region"].agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0])
    features["district"] = grouped["district"].agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0])
    features["main_direction"] = grouped["direction_short"].agg(
        lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0]
    )

    features["approved_count"] = grouped["is_approved"].sum()
    features["completed_count"] = grouped["is_completed"].sum()
    features["rejected_count"] = grouped["is_rejected"].sum()
    features["withdrawn_count"] = grouped["is_withdrawn"].sum()

    features["approval_rate"] = features["approved_count"] / features["total_apps"]
    features["completion_rate"] = features["completed_count"] / features["total_apps"]
    features["rejection_rate"] = features["rejected_count"] / features["total_apps"]

    features["total_amount"] = grouped["amount"].sum()
    features["avg_amount"] = grouped["amount"].mean()
    features["median_amount"] = grouped["amount"].median()
    features["max_amount"] = grouped["amount"].max()
    features["amount_std"] = grouped["amount"].std().fillna(0)
    features["amount_cv"] = (features["amount_std"] / features["avg_amount"]).replace([np.inf, -np.inf], 0).fillna(0)

    features["direction_count"] = grouped["direction_short"].nunique()
    features["subsidy_type_count"] = grouped["subsidy_name"].nunique()
    features["direction_diversity"] = features["direction_count"] / max(
        df["direction_short"].nunique(), 1
    )

    features["avg_rate"] = grouped["rate"].mean()

    dates = grouped["date"]
    features["first_app_date"] = dates.min()
    features["last_app_date"] = dates.max()
    features["activity_span_days"] = (features["last_app_date"] - features["first_app_date"]).dt.days.fillna(0)

    features["apps_per_month"] = features.apply(
        lambda row: (
            row["total_apps"] / max(row["activity_span_days"] / 30, 1)
            if row["activity_span_days"] > 0
            else row["total_apps"]
        ),
        axis=1,
    )

    features["unique_months"] = grouped["month"].nunique()
    features["month_regularity"] = features["unique_months"] / features["total_apps"]
    features["month_regularity"] = features["month_regularity"].clip(0, 1)

    features["preferred_hour"] = grouped["hour"].agg(
        lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 12
    )
    features["working_hours_ratio"] = grouped["hour"].agg(
        lambda x: ((x >= 9) & (x <= 18)).mean()
    )

    features["unique_districts"] = grouped["district"].nunique()

    completed_amounts = df[df["is_completed"] == 1].groupby("producer_id")["amount"].sum()
    features["completed_amount"] = completed_amounts.reindex(features.index).fillna(0)
    features["utilization_rate"] = (
        features["completed_amount"] / features["total_amount"]
    ).replace([np.inf, -np.inf], 0).fillna(0)
    features["utilization_rate"] = features["utilization_rate"].clip(0, 1)

    features["amount_log"] = np.log1p(features["total_amount"])
    features["avg_amount_log"] = np.log1p(features["avg_amount"])

    return features.reset_index()


def get_scoring_features() -> List[str]:
    """Список признаков, используемых в скоринговой модели."""
    return [
        "total_apps",
        "approval_rate",
        "completion_rate",
        "rejection_rate",
        "avg_amount_log",
        "amount_cv",
        "direction_diversity",
        "subsidy_type_count",
        "utilization_rate",
        "apps_per_month",
        "working_hours_ratio",
        "month_regularity",
        "unique_districts",
        "activity_span_days",
    ]


def get_feature_descriptions() -> Dict[str, str]:
    """Описание признаков на русском языке."""
    return {
        "total_apps": "Общее кол-во заявок",
        "approval_rate": "Доля одобренных заявок",
        "completion_rate": "Доля исполненных заявок",
        "rejection_rate": "Доля отклонённых заявок",
        "avg_amount_log": "Средняя сумма (log)",
        "amount_cv": "Вариативность сумм",
        "direction_diversity": "Диверсификация направлений",
        "subsidy_type_count": "Кол-во видов субсидий",
        "utilization_rate": "Коэф. освоения средств",
        "apps_per_month": "Заявок в месяц",
        "working_hours_ratio": "Подача в рабочее время",
        "month_regularity": "Регулярность подачи",
        "unique_districts": "Кол-во районов",
        "activity_span_days": "Период активности (дни)",
    }


def get_feature_weights_default() -> Dict[str, float]:
    """Веса признаков по умолчанию для правилового скоринга."""
    return {
        "approval_rate": 0.20,
        "completion_rate": 0.15,
        "utilization_rate": 0.15,
        "rejection_rate": -0.10,
        "direction_diversity": 0.10,
        "subsidy_type_count": 0.05,
        "total_apps": 0.05,
        "apps_per_month": 0.05,
        "working_hours_ratio": 0.03,
        "month_regularity": 0.05,
        "avg_amount_log": 0.02,
        "amount_cv": -0.02,
        "unique_districts": 0.02,
        "activity_span_days": 0.03,
    }


def prepare_model_data(
    producer_features: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Подготовка данных для модели ML.
    Создаёт целевую переменную на основе комбинации метрик.
    """
    feature_cols = get_scoring_features()
    X = producer_features[feature_cols].copy()
    X = X.fillna(0)
    X = X.replace([np.inf, -np.inf], 0)

    y = (
        producer_features["completion_rate"] * 0.35
        + producer_features["approval_rate"] * 0.25
        + producer_features["utilization_rate"] * 0.20
        + producer_features["direction_diversity"] * 0.10
        + (1 - producer_features["rejection_rate"]) * 0.10
    )
    y = y.clip(0, 1)

    return X, y
