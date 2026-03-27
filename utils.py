"""
Утилиты и вспомогательные функции.
"""

import pandas as pd
import numpy as np
from typing import Optional


def format_tenge(amount: float) -> str:
    """Форматирование суммы в тенге."""
    if amount >= 1_000_000_000:
        return f"{amount / 1_000_000_000:.1f} млрд ₸"
    elif amount >= 1_000_000:
        return f"{amount / 1_000_000:.1f} млн ₸"
    elif amount >= 1_000:
        return f"{amount / 1_000:.1f} тыс ₸"
    else:
        return f"{amount:.0f} ₸"


def format_percent(value: float) -> str:
    """Форматирование процента."""
    return f"{value * 100:.1f}%"


def format_score(score: float) -> str:
    """Форматирование скорингового балла."""
    return f"{score:.1f}"


def score_color(score: float) -> str:
    """Цвет для визуализации балла."""
    if score >= 75:
        return "#2ecc71"
    elif score >= 50:
        return "#f39c12"
    elif score >= 25:
        return "#e67e22"
    else:
        return "#e74c3c"


def score_label(score: float) -> str:
    """Текстовая метка балла."""
    if score >= 75:
        return "Высокий"
    elif score >= 50:
        return "Средний"
    elif score >= 25:
        return "Ниже среднего"
    else:
        return "Низкий"


def truncate_id(producer_id: str, length: int = 6) -> str:
    """Сокращённый ID для отображения."""
    return f"...{producer_id[-length:]}" if len(producer_id) > length else producer_id


def dataframe_to_display(
    df: pd.DataFrame,
    column_map: Optional[dict] = None,
    max_rows: int = 100,
) -> pd.DataFrame:
    """Подготовка DataFrame для отображения в Streamlit."""
    display = df.head(max_rows).copy()
    if column_map:
        display = display.rename(columns=column_map)
    return display


PRODUCER_DISPLAY_COLS = {
    "producer_id": "ID производителя",
    "region": "Регион",
    "district": "Район",
    "main_direction": "Направление",
    "total_apps": "Заявок",
    "approval_rate": "Одобрение",
    "completion_rate": "Исполнение",
    "rejection_rate": "Отклонение",
    "total_amount": "Общая сумма (₸)",
    "utilization_rate": "Освоение",
    "combined_score": "Балл",
    "rank": "Ранг",
    "rule_score": "Балл (правила)",
    "ml_score": "Балл (ML)",
    "direction_diversity": "Диверсификация",
}


SHORTLIST_COLS = [
    "rank",
    "producer_id",
    "region",
    "main_direction",
    "combined_score",
    "approval_rate",
    "utilization_rate",
    "total_amount",
    "total_apps",
]


def export_shortlist_csv(shortlist: pd.DataFrame) -> str:
    """Экспорт шортлиста в CSV-формат."""
    display = shortlist[
        [c for c in SHORTLIST_COLS if c in shortlist.columns]
    ].copy()
    display = display.rename(columns=PRODUCER_DISPLAY_COLS)
    return display.to_csv(index=False, encoding="utf-8-sig")
