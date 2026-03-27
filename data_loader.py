"""
Модуль загрузки и предобработки данных из реестра субсидий.
Обеспечивает чтение Excel-файла, очистку и нормализацию данных.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional

DATA_PATH = Path(__file__).parent / "data" / "subsidies_2025.xlsx"

COLUMN_MAP = {
    "№ п/п": "id",
    "Дата поступления": "date_raw",
    "Область": "region",
    "Акимат": "akimat",
    "Номер заявки": "app_num",
    "Направление водства": "direction",
    "Наименование субсидирования": "subsidy_name",
    "Статус заявки": "status",
    "Норматив": "rate",
    "Причитающая сумма": "amount",
    "Район хозяйства": "district",
}

STATUS_LABELS = {
    "Исполнена": "Исполнена",
    "Одобрена": "Одобрена",
    "Отклонена": "Отклонена",
    "Сформировано поручение": "Сформировано поручение",
    "Отозвано": "Отозвано",
    "Получена": "Получена",
}

POSITIVE_STATUSES = {"Исполнена", "Одобрена", "Сформировано поручение"}
COMPLETED_STATUS = "Исполнена"

DIRECTION_SHORT = {
    "Субсидирование в скотоводстве": "Скотоводство",
    "Субсидирование в овцеводстве": "Овцеводство",
    "Субсидирование в птицеводстве": "Птицеводство",
    "Субсидирование в свиноводстве": "Свиноводство",
    "Субсидирование в верблюдоводстве": "Верблюдоводство",
    "Субсидирование в коневодстве": "Коневодство",
    "Субсидирование затрат по искусственному осеменению": "Искусственное осеменение",
    "Субсидирование в пчеловодстве": "Пчеловодство",
    "Субсидирование в козоводстве": "Козоводство",
}


def load_raw_data(filepath: Optional[str] = None) -> pd.DataFrame:
    """Загрузка сырых данных из Excel-файла."""
    path = Path(filepath) if filepath else DATA_PATH
    if not path.exists():
        raise FileNotFoundError(f"Файл данных не найден: {path}")

    df = pd.read_excel(str(path), header=4, engine="openpyxl")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Очистка и нормализация данных."""
    drop_cols = [c for c in df.columns if c.startswith("Unnamed")]
    df = df.drop(columns=drop_cols, errors="ignore")

    rename_map = {}
    for orig, new in COLUMN_MAP.items():
        if orig in df.columns:
            rename_map[orig] = new
    df = df.rename(columns=rename_map)

    initial_len = len(df)
    df = df.dropna(subset=["app_num", "status", "region"])
    dropped = initial_len - len(df)
    if dropped > 0:
        print(f"Удалено {dropped} строк с пустыми ключевыми полями")

    df["app_num_str"] = (
        df["app_num"].astype(np.int64).astype(str).str.zfill(14)
    )

    df["date"] = pd.to_datetime(df["date_raw"], format="%d.%m.%Y %H:%M:%S", errors="coerce")
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek
    df["hour"] = df["date"].dt.hour
    df["week_num"] = df["date"].dt.isocalendar().week.astype(int)

    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0)
    df["rate"] = pd.to_numeric(df["rate"], errors="coerce").fillna(0)

    df["direction_short"] = df["direction"].map(DIRECTION_SHORT).fillna(df["direction"])

    df["is_approved"] = df["status"].isin(POSITIVE_STATUSES).astype(int)
    df["is_completed"] = (df["status"] == COMPLETED_STATUS).astype(int)
    df["is_rejected"] = (df["status"] == "Отклонена").astype(int)
    df["is_withdrawn"] = (df["status"] == "Отозвано").astype(int)

    df["producer_id"] = df["app_num_str"].str[:11]

    df["district"] = df["district"].fillna("Не указан")

    df["amount_mln"] = df["amount"] / 1_000_000

    return df


def load_and_process(filepath: Optional[str] = None) -> pd.DataFrame:
    """Полный пайплайн: загрузка + очистка."""
    raw = load_raw_data(filepath)
    clean = clean_data(raw)
    return clean


def get_summary_stats(df: pd.DataFrame) -> dict:
    """Основные статистики по датасету."""
    return {
        "total_applications": len(df),
        "unique_producers": df["producer_id"].nunique(),
        "unique_regions": df["region"].nunique(),
        "unique_districts": df["district"].nunique(),
        "total_amount": df["amount"].sum(),
        "avg_amount": df["amount"].mean(),
        "median_amount": df["amount"].median(),
        "approval_rate": df["is_approved"].mean(),
        "completion_rate": df["is_completed"].mean(),
        "rejection_rate": df["is_rejected"].mean(),
        "date_range": (
            df["date"].min().strftime("%d.%m.%Y") if df["date"].notna().any() else "N/A",
            df["date"].max().strftime("%d.%m.%Y") if df["date"].notna().any() else "N/A",
        ),
        "status_counts": df["status"].value_counts().to_dict(),
        "direction_counts": df["direction_short"].value_counts().to_dict(),
        "region_counts": df["region"].value_counts().to_dict(),
    }
