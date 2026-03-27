"""
Модуль анализа справедливости и выявления предвзятости.
Проверяет равномерность скоринга по регионам и направлениям.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple
from scipy import stats


COLOR_SCALE = px.colors.sequential.Teal


def compute_regional_fairness(scored: pd.DataFrame) -> pd.DataFrame:
    """Анализ справедливости распределения по регионам."""
    regional = scored.groupby("region").agg(
        producer_count=("producer_id", "size"),
        mean_score=("combined_score", "mean"),
        median_score=("combined_score", "median"),
        std_score=("combined_score", "std"),
        min_score=("combined_score", "min"),
        max_score=("combined_score", "max"),
        total_amount=("total_amount", "sum"),
        mean_amount=("total_amount", "mean"),
        mean_approval=("approval_rate", "mean"),
    ).reset_index()

    overall_mean = scored["combined_score"].mean()
    overall_std = scored["combined_score"].std()

    regional["score_deviation"] = (
        (regional["mean_score"] - overall_mean) / overall_std
    ).round(3)

    regional["bias_flag"] = regional["score_deviation"].abs() > 1.0
    regional["bias_direction"] = np.where(
        regional["score_deviation"] > 1.0,
        "Завышен",
        np.where(regional["score_deviation"] < -1.0, "Занижен", "Норма"),
    )

    return regional.sort_values("mean_score", ascending=False)


def compute_direction_fairness(scored: pd.DataFrame) -> pd.DataFrame:
    """Анализ справедливости по направлениям животноводства."""
    dir_stats = scored.groupby("main_direction").agg(
        producer_count=("producer_id", "size"),
        mean_score=("combined_score", "mean"),
        median_score=("combined_score", "median"),
        std_score=("combined_score", "std"),
        mean_approval=("approval_rate", "mean"),
    ).reset_index()

    overall_mean = scored["combined_score"].mean()
    dir_stats["deviation"] = (dir_stats["mean_score"] - overall_mean).round(2)

    return dir_stats.sort_values("mean_score", ascending=False)


def compute_gini_coefficient(values: np.ndarray) -> float:
    """Вычисление коэффициента Джини для оценки неравенства."""
    values = np.sort(values[values > 0])
    n = len(values)
    if n == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * values) - (n + 1) * np.sum(values)) / (n * np.sum(values)))


def compute_fairness_metrics(scored: pd.DataFrame) -> Dict[str, float]:
    """Сводные метрики справедливости."""
    regional = compute_regional_fairness(scored)

    gini_score = compute_gini_coefficient(scored["combined_score"].values)
    gini_amount = compute_gini_coefficient(scored["total_amount"].values)

    regional_means = scored.groupby("region")["combined_score"].mean()
    cv_regional = float(regional_means.std() / regional_means.mean()) if regional_means.mean() > 0 else 0

    biased_regions = int(regional["bias_flag"].sum())

    groups = [g["combined_score"].values for _, g in scored.groupby("region")]
    groups_clean = [g for g in groups if len(g) >= 2]
    if len(groups_clean) >= 2:
        h_stat, p_value = stats.kruskal(*groups_clean)
    else:
        h_stat, p_value = 0.0, 1.0

    return {
        "gini_score": round(gini_score, 4),
        "gini_amount": round(gini_amount, 4),
        "cv_regional": round(cv_regional, 4),
        "biased_regions": biased_regions,
        "total_regions": len(regional),
        "kruskal_h": round(float(h_stat), 2),
        "kruskal_p": round(float(p_value), 4),
        "score_range": round(float(regional_means.max() - regional_means.min()), 2),
        "max_deviation": round(float(regional["score_deviation"].abs().max()), 3),
    }


def plot_fairness_overview(scored: pd.DataFrame) -> go.Figure:
    """Обзор справедливости: средние баллы по регионам с доверительным интервалом."""
    regional = scored.groupby("region").agg(
        mean=("combined_score", "mean"),
        std=("combined_score", "std"),
        count=("producer_id", "size"),
    ).reset_index()

    regional["ci"] = 1.96 * regional["std"] / np.sqrt(regional["count"])
    regional = regional.sort_values("mean", ascending=True)

    overall_mean = scored["combined_score"].mean()

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=regional["region"],
            x=regional["mean"],
            orientation="h",
            marker_color=COLOR_SCALE[4],
            error_x=dict(type="data", array=regional["ci"].values, visible=True),
            text=regional["mean"].round(1),
            textposition="auto",
        )
    )
    fig.add_vline(
        x=overall_mean,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Среднее: {overall_mean:.1f}",
    )
    fig.update_layout(
        title="Средний балл по регионам (с 95% доверительным интервалом)",
        xaxis_title="Средний балл",
        template="plotly_white",
        height=600,
        margin=dict(l=250),
    )
    return fig


def plot_score_violin_by_region(scored: pd.DataFrame) -> go.Figure:
    """Violin-plot распределения баллов по регионам."""
    top_regions = scored["region"].value_counts().nlargest(10).index.tolist()
    filtered = scored[scored["region"].isin(top_regions)]

    fig = px.violin(
        filtered,
        x="region",
        y="combined_score",
        box=True,
        points="outliers",
        title="Распределение баллов по крупнейшим регионам",
    )
    fig.update_layout(
        template="plotly_white",
        height=500,
        xaxis_tickangle=-45,
        xaxis_title="",
        yaxis_title="Балл",
        margin=dict(b=150),
    )
    return fig


def plot_lorenz_curve(scored: pd.DataFrame) -> go.Figure:
    """Кривая Лоренца для оценки неравенства распределения субсидий."""
    amounts = np.sort(scored["total_amount"].values)
    cumulative = np.cumsum(amounts)
    cumulative = cumulative / cumulative[-1]
    x = np.linspace(0, 1, len(cumulative))

    scores = np.sort(scored["combined_score"].values)
    cum_scores = np.cumsum(scores)
    cum_scores = cum_scores / cum_scores[-1]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x, y=cumulative,
            mode="lines",
            name="Субсидии (суммы)",
            line=dict(color=COLOR_SCALE[5], width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=cum_scores,
            mode="lines",
            name="Скоринговые баллы",
            line=dict(color=COLOR_SCALE[3], width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1],
            mode="lines",
            name="Идеальное равенство",
            line=dict(color="gray", dash="dash"),
        )
    )
    fig.update_layout(
        title="Кривая Лоренца: неравенство распределения",
        xaxis_title="Доля производителей (кумулятивно)",
        yaxis_title="Доля (кумулятивно)",
        template="plotly_white",
        height=450,
    )
    return fig


def generate_fairness_report(scored: pd.DataFrame) -> List[str]:
    """Генерация текстовых рекомендаций по справедливости."""
    metrics = compute_fairness_metrics(scored)
    regional = compute_regional_fairness(scored)
    recommendations = []

    if metrics["gini_score"] < 0.3:
        recommendations.append(
            "[+] Коэффициент Джини для баллов ({:.3f}) указывает на умеренное неравенство -- "
            "распределение баллов достаточно равномерное.".format(metrics["gini_score"])
        )
    elif metrics["gini_score"] < 0.5:
        recommendations.append(
            "[!] Коэффициент Джини для баллов ({:.3f}) указывает на заметное неравенство. "
            "Рекомендуется проверить факторы, влияющие на разброс.".format(metrics["gini_score"])
        )
    else:
        recommendations.append(
            "[-] Коэффициент Джини для баллов ({:.3f}) указывает на значительное неравенство. "
            "Необходим пересмотр методологии.".format(metrics["gini_score"])
        )

    biased = regional[regional["bias_flag"]]
    if len(biased) > 0:
        for _, row in biased.iterrows():
            direction = "завышены" if row["bias_direction"] == "Завышен" else "занижены"
            recommendations.append(
                f"[!] Регион \"{row['region']}\": средние баллы {direction} "
                f"(отклонение {row['score_deviation']:.2f}s от среднего). "
                f"Кол-во производителей: {row['producer_count']}."
            )
    else:
        recommendations.append(
            "[+] Ни один регион не показывает статистически значимого отклонения (>1s)."
        )

    if metrics["kruskal_p"] < 0.05:
        recommendations.append(
            "[!] Тест Краскела-Уоллиса: различия между регионами статистически значимы "
            f"(H={metrics['kruskal_h']}, p={metrics['kruskal_p']}). "
            "Рекомендуется регионально-адаптивная нормализация."
        )
    else:
        recommendations.append(
            "[+] Тест Краскела-Уоллиса не выявил статистически значимых различий "
            f"между регионами (p={metrics['kruskal_p']})."
        )

    recommendations.append(
        "\nРекомендации по обеспечению справедливости:\n"
        "1. Ввести региональные квоты пропорционально числу хозяйств\n"
        "2. Применять нормализацию баллов внутри каждого региона\n"
        "3. Создать апелляционную комиссию для пограничных случаев\n"
        "4. Проводить ежеквартальный аудит распределения субсидий\n"
        "5. Учитывать социально-экономические различия регионов"
    )

    return recommendations
