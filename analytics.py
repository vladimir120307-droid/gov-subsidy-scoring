"""
Модуль аналитики и визуализации.
Генерация графиков и аналитических отчётов с использованием Plotly.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional


PLOTLY_TEMPLATE = "plotly_white"
COLOR_SCALE = px.colors.sequential.Teal
COLOR_DISCRETE = px.colors.qualitative.Set2

STATUS_COLORS = {
    "Исполнена": "#2ecc71",
    "Одобрена": "#3498db",
    "Отклонена": "#e74c3c",
    "Сформировано поручение": "#f39c12",
    "Отозвано": "#95a5a6",
    "Получена": "#9b59b6",
}


def _empty_figure(message: str = "Нет данных для отображения", height: int = 300) -> go.Figure:
    """Возвращает пустую фигуру с информационным сообщением."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=16, color="#888"),
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=height,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


def plot_status_distribution(df: pd.DataFrame) -> go.Figure:
    """Распределение заявок по статусам."""
    try:
        if df is None or df.empty or "status" not in df.columns:
            return _empty_figure("Нет данных о статусах заявок", 400)

        status_counts = df["status"].value_counts().reset_index()
        status_counts.columns = ["Статус", "Количество"]

        if status_counts.empty:
            return _empty_figure("Нет данных о статусах заявок", 400)

        colors = [STATUS_COLORS.get(s, "#bdc3c7") for s in status_counts["Статус"]]

        fig = go.Figure(
            go.Bar(
                x=status_counts["Статус"],
                y=status_counts["Количество"],
                marker_color=colors,
                text=status_counts["Количество"],
                textposition="auto",
            )
        )
        fig.update_layout(
            title="Распределение заявок по статусам",
            xaxis_title="Статус",
            yaxis_title="Количество заявок",
            template=PLOTLY_TEMPLATE,
            height=400,
        )
        return fig
    except Exception:
        return _empty_figure("Ошибка построения графика статусов", 400)


def plot_region_distribution(df: pd.DataFrame) -> go.Figure:
    """Распределение заявок по регионам."""
    try:
        if df is None or df.empty:
            return _empty_figure("Нет данных по регионам", 600)

        required = ["region", "id", "amount_mln", "is_approved"]
        if not all(c in df.columns for c in required):
            return _empty_figure("Отсутствуют необходимые столбцы", 600)

        region_data = df.groupby("region").agg(
            count=("id", "size"),
            total_amount=("amount_mln", "sum"),
            approval_rate=("is_approved", "mean"),
        ).reset_index().sort_values("count", ascending=True)

        if region_data.empty:
            return _empty_figure("Нет данных по регионам", 600)

        fig = go.Figure(
            go.Bar(
                y=region_data["region"],
                x=region_data["count"],
                orientation="h",
                marker_color=COLOR_SCALE[3],
                text=region_data["count"],
                textposition="auto",
            )
        )
        fig.update_layout(
            title="Количество заявок по регионам",
            xaxis_title="Количество заявок",
            yaxis_title="",
            template=PLOTLY_TEMPLATE,
            height=600,
            margin=dict(l=250),
        )
        return fig
    except Exception:
        return _empty_figure("Ошибка построения графика регионов", 600)


def plot_region_amounts(df: pd.DataFrame) -> go.Figure:
    """Суммы субсидий по регионам."""
    try:
        if df is None or df.empty:
            return _empty_figure("Нет данных по суммам регионов", 600)

        required = ["region", "amount_mln"]
        if not all(c in df.columns for c in required):
            return _empty_figure("Отсутствуют необходимые столбцы", 600)

        region_data = df.groupby("region")["amount_mln"].sum().reset_index()
        region_data.columns = ["Регион", "Сумма (млн тг)"]
        region_data = region_data.sort_values("Сумма (млн тг)", ascending=True)

        if region_data.empty:
            return _empty_figure("Нет данных по суммам регионов", 600)

        fig = go.Figure(
            go.Bar(
                y=region_data["Регион"],
                x=region_data["Сумма (млн тг)"],
                orientation="h",
                marker_color=COLOR_SCALE[5],
                text=region_data["Сумма (млн тг)"].round(1),
                textposition="auto",
            )
        )
        fig.update_layout(
            title="Объём субсидий по регионам (млн тг)",
            xaxis_title="Сумма (млн тг)",
            yaxis_title="",
            template=PLOTLY_TEMPLATE,
            height=600,
            margin=dict(l=250),
        )
        return fig
    except Exception:
        return _empty_figure("Ошибка построения графика сумм по регионам", 600)


def plot_direction_pie(df: pd.DataFrame) -> go.Figure:
    """Распределение по направлениям животноводства."""
    try:
        if df is None or df.empty or "direction_short" not in df.columns:
            return _empty_figure("Нет данных по направлениям", 450)

        dir_data = df["direction_short"].value_counts().reset_index()
        dir_data.columns = ["Направление", "Количество"]

        if dir_data.empty:
            return _empty_figure("Нет данных по направлениям", 450)

        fig = px.pie(
            dir_data,
            values="Количество",
            names="Направление",
            title="Распределение по направлениям",
            color_discrete_sequence=COLOR_DISCRETE,
            hole=0.3,
        )
        fig.update_layout(template=PLOTLY_TEMPLATE, height=450)
        return fig
    except Exception:
        return _empty_figure("Ошибка построения диаграммы направлений", 450)


def plot_monthly_trend(df: pd.DataFrame) -> go.Figure:
    """Динамика подачи заявок по месяцам."""
    try:
        if df is None or df.empty:
            return _empty_figure("Нет данных для отображения динамики", 400)

        required = ["month", "id", "is_approved", "amount_mln"]
        if not all(c in df.columns for c in required):
            return _empty_figure("Отсутствуют необходимые столбцы", 400)

        monthly = df.groupby("month").agg(
            count=("id", "size"),
            approved=("is_approved", "sum"),
            amount_mln=("amount_mln", "sum"),
        ).reset_index()

        if monthly.empty:
            return _empty_figure("Нет данных для отображения динамики", 400)

        month_names = {
            1: "Янв", 2: "Фев", 3: "Мар", 4: "Апр", 5: "Май", 6: "Июн",
            7: "Июл", 8: "Авг", 9: "Сен", 10: "Окт", 11: "Ноя", 12: "Дек",
        }
        monthly["month_name"] = monthly["month"].map(month_names)

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Bar(
                x=monthly["month_name"],
                y=monthly["count"],
                name="Всего заявок",
                marker_color=COLOR_SCALE[3],
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=monthly["month_name"],
                y=monthly["amount_mln"],
                name="Сумма (млн тг)",
                mode="lines+markers",
                line=dict(color=COLOR_SCALE[6], width=3),
            ),
            secondary_y=True,
        )
        fig.update_layout(
            title="Динамика подачи заявок по месяцам",
            template=PLOTLY_TEMPLATE,
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        fig.update_yaxes(title_text="Количество заявок", secondary_y=False)
        fig.update_yaxes(title_text="Сумма (млн тг)", secondary_y=True)
        return fig
    except Exception:
        return _empty_figure("Ошибка построения графика динамики", 400)


def plot_score_distribution(scored: pd.DataFrame, score_col: str = "combined_score") -> go.Figure:
    """Распределение скоринговых баллов."""
    try:
        if scored is None or scored.empty or score_col not in scored.columns:
            return _empty_figure("Нет данных о баллах", 400)

        fig = go.Figure(
            go.Histogram(
                x=scored[score_col],
                nbinsx=50,
                marker_color=COLOR_SCALE[4],
                opacity=0.8,
            )
        )
        fig.add_vline(
            x=scored[score_col].median(),
            line_dash="dash",
            line_color="red",
            annotation_text=f"Медиана: {scored[score_col].median():.1f}",
        )
        fig.update_layout(
            title="Распределение скоринговых баллов",
            xaxis_title="Балл",
            yaxis_title="Количество производителей",
            template=PLOTLY_TEMPLATE,
            height=400,
        )
        return fig
    except Exception:
        return _empty_figure("Ошибка построения распределения баллов", 400)


def plot_score_by_region(scored: pd.DataFrame) -> go.Figure:
    """Боксплот баллов по регионам."""
    try:
        if scored is None or scored.empty:
            return _empty_figure("Нет данных по регионам", 500)

        required = ["region", "combined_score"]
        if not all(c in scored.columns for c in required):
            return _empty_figure("Отсутствуют необходимые столбцы", 500)

        fig = px.box(
            scored.sort_values("combined_score", ascending=False),
            x="region",
            y="combined_score",
            title="Распределение баллов по регионам",
            color_discrete_sequence=[COLOR_SCALE[4]],
        )
        fig.update_layout(
            template=PLOTLY_TEMPLATE,
            height=500,
            xaxis_tickangle=-45,
            xaxis_title="",
            yaxis_title="Балл",
            margin=dict(b=150),
        )
        return fig
    except Exception:
        return _empty_figure("Ошибка построения боксплота", 500)


def plot_producer_breakdown(
    scored_row: pd.Series, feature_descriptions: Dict[str, str]
) -> go.Figure:
    """Разбивка скоринга для конкретного производителя."""
    try:
        contrib_cols = [c for c in scored_row.index if c.startswith("contrib_")]
        if not contrib_cols:
            return _empty_figure("Нет данных о вкладе факторов", 350)

        data = []
        for c in contrib_cols:
            feat = c.replace("contrib_", "")
            name = feature_descriptions.get(feat, feat)
            val = float(scored_row[c]) * 100
            data.append({"Фактор": name, "Вклад": val})

        if not data:
            return _empty_figure("Нет данных о вкладе факторов", 350)

        breakdown = pd.DataFrame(data).sort_values("Вклад", ascending=True)

        colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in breakdown["Вклад"]]

        fig = go.Figure(
            go.Bar(
                y=breakdown["Фактор"],
                x=breakdown["Вклад"],
                orientation="h",
                marker_color=colors,
                text=breakdown["Вклад"].round(2),
                textposition="auto",
            )
        )
        fig.update_layout(
            title="Вклад факторов в итоговый балл",
            xaxis_title="Вклад (баллы)",
            yaxis_title="",
            template=PLOTLY_TEMPLATE,
            height=max(300, len(data) * 35),
            margin=dict(l=200),
        )
        return fig
    except Exception:
        return _empty_figure("Ошибка построения разбивки факторов", 350)


def plot_feature_importance(importances: Dict[str, float]) -> go.Figure:
    """Визуализация важности признаков модели ML."""
    try:
        if not importances:
            return _empty_figure("Нет данных о важности признаков", 350)

        sorted_imp = sorted(importances.items(), key=lambda x: x[1])
        names = [x[0] for x in sorted_imp]
        values = [x[1] for x in sorted_imp]

        fig = go.Figure(
            go.Bar(
                y=names,
                x=values,
                orientation="h",
                marker_color=COLOR_SCALE[5],
                text=[f"{v:.3f}" for v in values],
                textposition="auto",
            )
        )
        fig.update_layout(
            title="Важность признаков (ML-модель)",
            xaxis_title="Важность",
            yaxis_title="",
            template=PLOTLY_TEMPLATE,
            height=max(300, len(names) * 30),
            margin=dict(l=200),
        )
        return fig
    except Exception:
        return _empty_figure("Ошибка построения графика важности", 350)


def plot_comparison_radar(
    producers: List[pd.Series],
    producer_ids: List[str],
    features: List[str],
    feature_descriptions: Dict[str, str],
) -> go.Figure:
    """Радарная диаграмма для сравнения производителей."""
    try:
        if not producers or not producer_ids or not features:
            return _empty_figure("Нет данных для сравнения", 500)

        fig = go.Figure()

        labels = [feature_descriptions.get(f, f) for f in features]
        labels.append(labels[0])

        for prod, pid in zip(producers, producer_ids):
            values = []
            for f in features:
                val = float(prod.get(f, 0))
                values.append(val)
            values.append(values[0])

            fig.add_trace(
                go.Scatterpolar(
                    r=values,
                    theta=labels,
                    fill="toself",
                    name=f"Производитель {pid[-4:]}",
                    opacity=0.6,
                )
            )

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title="Сравнительный профиль производителей",
            template=PLOTLY_TEMPLATE,
            height=500,
        )
        return fig
    except Exception:
        return _empty_figure("Ошибка построения радарной диаграммы", 500)


def plot_amount_vs_score(scored: pd.DataFrame) -> go.Figure:
    """Scatter: сумма субсидий vs скоринговый балл."""
    try:
        if scored is None or scored.empty:
            return _empty_figure("Нет данных для построения", 500)

        required = ["total_amount", "combined_score", "main_direction", "total_apps"]
        if not all(c in scored.columns for c in required):
            return _empty_figure("Отсутствуют необходимые столбцы", 500)

        fig = px.scatter(
            scored,
            x="total_amount",
            y="combined_score",
            color="main_direction",
            size="total_apps",
            hover_data=["producer_id", "region", "approval_rate"],
            title="Зависимость балла от объёма субсидий",
            labels={
                "total_amount": "Общая сумма субсидий (тг)",
                "combined_score": "Скоринговый балл",
                "main_direction": "Направление",
                "total_apps": "Кол-во заявок",
            },
            color_discrete_sequence=COLOR_DISCRETE,
        )
        fig.update_layout(template=PLOTLY_TEMPLATE, height=500)
        fig.update_xaxes(type="log")
        return fig
    except Exception:
        return _empty_figure("Ошибка построения scatter-графика", 500)


def plot_correlation_heatmap(scored: pd.DataFrame, features: List[str]) -> go.Figure:
    """Корреляционная матрица признаков."""
    try:
        if scored is None or scored.empty or not features:
            return _empty_figure("Нет данных для корреляционной матрицы", 600)

        from feature_engineering import get_feature_descriptions

        descs = get_feature_descriptions()
        display_features = [f for f in features if f in scored.columns]

        if len(display_features) < 2:
            return _empty_figure("Недостаточно признаков для корреляции", 600)

        corr = scored[display_features].corr()

        labels = [descs.get(f, f) for f in display_features]

        fig = go.Figure(
            go.Heatmap(
                z=corr.values,
                x=labels,
                y=labels,
                colorscale="RdBu_r",
                zmid=0,
                text=np.round(corr.values, 2),
                texttemplate="%{text}",
                textfont={"size": 9},
            )
        )
        fig.update_layout(
            title="Корреляционная матрица признаков",
            template=PLOTLY_TEMPLATE,
            height=600,
            width=700,
            xaxis_tickangle=-45,
            margin=dict(b=150, l=150),
        )
        return fig
    except Exception:
        return _empty_figure("Ошибка построения корреляционной матрицы", 600)


def plot_approval_rate_by_direction(df: pd.DataFrame) -> go.Figure:
    """Доля одобрения по направлениям."""
    try:
        if df is None or df.empty:
            return _empty_figure("Нет данных по направлениям", 400)

        required = ["direction_short", "id", "is_approved"]
        if not all(c in df.columns for c in required):
            return _empty_figure("Отсутствуют необходимые столбцы", 400)

        dir_stats = df.groupby("direction_short").agg(
            total=("id", "size"),
            approved=("is_approved", "sum"),
        ).reset_index()

        if dir_stats.empty:
            return _empty_figure("Нет данных по направлениям", 400)

        dir_stats["rate"] = (dir_stats["approved"] / dir_stats["total"] * 100).round(1)
        dir_stats = dir_stats.sort_values("rate", ascending=True)

        fig = go.Figure(
            go.Bar(
                y=dir_stats["direction_short"],
                x=dir_stats["rate"],
                orientation="h",
                marker_color=COLOR_SCALE[4],
                text=[f"{r}%" for r in dir_stats["rate"]],
                textposition="auto",
            )
        )
        fig.update_layout(
            title="Доля одобренных заявок по направлениям",
            xaxis_title="Доля одобрения (%)",
            yaxis_title="",
            template=PLOTLY_TEMPLATE,
            height=400,
            margin=dict(l=200),
        )
        return fig
    except Exception:
        return _empty_figure("Ошибка построения графика одобрений", 400)
