"""
Скоринговый движок: ML-модель и правиловый базовый скоринг.
Обеспечивает оценку каждого производителя с объяснимостью результатов.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score
from dataclasses import dataclass, field

from feature_engineering import (
    get_scoring_features,
    get_feature_descriptions,
    get_feature_weights_default,
    prepare_model_data,
)


@dataclass
class ScoreExplanation:
    """Структура объяснения скоринга для одного производителя."""
    producer_id: str
    total_score: float
    rank: int
    factor_contributions: Dict[str, float] = field(default_factory=dict)
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    recommendation: str = ""


class RuleBasedScorer:
    """Правиловый скоринг на основе экспертных весов."""

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or get_feature_weights_default()
        self.scaler = MinMaxScaler()
        self.feature_cols = get_scoring_features()
        self.descriptions = get_feature_descriptions()
        self._fitted = False

    def fit(self, producer_features: pd.DataFrame) -> "RuleBasedScorer":
        X = producer_features[self.feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
        self.scaler.fit(X)
        self._fitted = True
        return self

    def score(self, producer_features: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            self.fit(producer_features)

        X = producer_features[self.feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
        X_scaled = pd.DataFrame(
            self.scaler.transform(X),
            columns=self.feature_cols,
            index=X.index,
        )

        contributions = pd.DataFrame(index=X.index)
        total_positive_weight = sum(w for w in self.weights.values() if w > 0)
        total_negative_weight = abs(sum(w for w in self.weights.values() if w < 0))
        norm_factor = total_positive_weight + total_negative_weight

        for col in self.feature_cols:
            w = self.weights.get(col, 0)
            contributions[f"contrib_{col}"] = X_scaled[col] * w / norm_factor

        scores = contributions.sum(axis=1)
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
        scores = (scores * 100).round(2)

        result = producer_features[["producer_id"]].copy()
        result["rule_score"] = scores.values
        result["rule_rank"] = scores.rank(ascending=False, method="min").astype(int).values

        for col in self.feature_cols:
            result[f"contrib_{col}"] = contributions[f"contrib_{col}"].values

        return result

    def explain(self, producer_row: pd.Series, score_row: pd.Series) -> ScoreExplanation:
        contrib_cols = [c for c in score_row.index if c.startswith("contrib_")]
        contributions = {}
        for c in contrib_cols:
            feat = c.replace("contrib_", "")
            contributions[self.descriptions.get(feat, feat)] = round(float(score_row[c]) * 100, 2)

        sorted_contribs = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
        strengths = [f"{k}: +{v:.1f}" for k, v in sorted_contribs if v > 0][:5]
        weaknesses = [f"{k}: {v:.1f}" for k, v in sorted_contribs if v < 0][:3]

        score_val = float(score_row["rule_score"])
        if score_val >= 75:
            rec = "Рекомендован к приоритетному субсидированию"
        elif score_val >= 50:
            rec = "Рекомендован к субсидированию в стандартном порядке"
        elif score_val >= 25:
            rec = "Требуется дополнительная проверка"
        else:
            rec = "Не рекомендован без дополнительного обоснования"

        return ScoreExplanation(
            producer_id=str(producer_row.get("producer_id", "")),
            total_score=score_val,
            rank=int(score_row.get("rule_rank", 0)),
            factor_contributions=contributions,
            strengths=strengths,
            weaknesses=weaknesses,
            recommendation=rec,
        )


class MLScorer:
    """ML-скоринг на основе GradientBoosting с объяснимостью."""

    def __init__(self, n_estimators: int = 200, max_depth: int = 5, random_state: int = 42):
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=0.1,
            subsample=0.8,
            random_state=random_state,
        )
        self.scaler = MinMaxScaler()
        self.feature_cols = get_scoring_features()
        self.descriptions = get_feature_descriptions()
        self.feature_importances_ = {}
        self.cv_score_ = 0.0
        self._fitted = False

    def fit(self, producer_features: pd.DataFrame) -> "MLScorer":
        X, y = prepare_model_data(producer_features)
        X_clean = X.fillna(0).replace([np.inf, -np.inf], 0)

        self.scaler.fit(X_clean)
        X_scaled = self.scaler.transform(X_clean)

        self.model.fit(X_scaled, y)

        cv_scores = cross_val_score(self.model, X_scaled, y, cv=5, scoring="r2")
        self.cv_score_ = float(cv_scores.mean())

        importances = self.model.feature_importances_
        self.feature_importances_ = {
            self.descriptions.get(col, col): float(imp)
            for col, imp in zip(self.feature_cols, importances)
        }

        self._fitted = True
        return self

    def score(self, producer_features: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            self.fit(producer_features)

        X = producer_features[self.feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
        X_scaled = self.scaler.transform(X)

        raw_scores = self.model.predict(X_scaled)
        scores = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-10)
        scores = (scores * 100).round(2)

        result = producer_features[["producer_id"]].copy()
        result["ml_score"] = scores
        result["ml_rank"] = pd.Series(scores).rank(ascending=False, method="min").astype(int).values

        return result

    def get_feature_importance(self) -> Dict[str, float]:
        return dict(sorted(self.feature_importances_.items(), key=lambda x: x[1], reverse=True))

    def compute_permutation_importance(
        self, producer_features: pd.DataFrame, n_repeats: int = 10
    ) -> Dict[str, float]:
        X, y = prepare_model_data(producer_features)
        X_clean = X.fillna(0).replace([np.inf, -np.inf], 0)
        X_scaled = self.scaler.transform(X_clean)

        perm_imp = permutation_importance(
            self.model, X_scaled, y, n_repeats=n_repeats, random_state=42, scoring="r2"
        )

        return {
            self.descriptions.get(col, col): float(imp)
            for col, imp in zip(self.feature_cols, perm_imp.importances_mean)
        }


class ScoringEngine:
    """Комбинированный скоринговый движок."""

    def __init__(self, ml_weight: float = 0.6, rule_weight: float = 0.4):
        self.ml_scorer = MLScorer()
        self.rule_scorer = RuleBasedScorer()
        self.ml_weight = ml_weight
        self.rule_weight = rule_weight
        self._fitted = False

    def fit(self, producer_features: pd.DataFrame) -> "ScoringEngine":
        self.rule_scorer.fit(producer_features)
        self.ml_scorer.fit(producer_features)
        self._fitted = True
        return self

    def score(self, producer_features: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            self.fit(producer_features)

        rule_scores = self.rule_scorer.score(producer_features)
        ml_scores = self.ml_scorer.score(producer_features)

        result = producer_features.copy()
        result["rule_score"] = rule_scores["rule_score"].values
        result["ml_score"] = ml_scores["ml_score"].values
        result["combined_score"] = (
            result["rule_score"] * self.rule_weight + result["ml_score"] * self.ml_weight
        ).round(2)
        result["rank"] = result["combined_score"].rank(ascending=False, method="min").astype(int)

        for col in rule_scores.columns:
            if col.startswith("contrib_"):
                result[col] = rule_scores[col].values

        return result.sort_values("rank")

    def update_weights(
        self,
        feature_weights: Dict[str, float],
        ml_weight: float = 0.6,
        rule_weight: float = 0.4,
    ):
        self.rule_scorer.weights = feature_weights
        self.ml_weight = ml_weight
        self.rule_weight = rule_weight

    def explain_producer(
        self, producer_features: pd.DataFrame, scored: pd.DataFrame, producer_id: str
    ) -> Optional[ScoreExplanation]:
        mask = scored["producer_id"] == producer_id
        if not mask.any():
            return None

        prod_row = producer_features[producer_features["producer_id"] == producer_id].iloc[0]
        score_row = scored[mask].iloc[0]

        explanation = self.rule_scorer.explain(prod_row, score_row)
        explanation.total_score = float(score_row["combined_score"])
        explanation.rank = int(score_row["rank"])

        return explanation

    def generate_shortlist(
        self,
        scored: pd.DataFrame,
        top_n: int = 100,
        min_score: float = 0,
        region: Optional[str] = None,
        direction: Optional[str] = None,
    ) -> pd.DataFrame:
        filtered = scored.copy()
        if region and region != "Все":
            filtered = filtered[filtered["region"] == region]
        if direction and direction != "Все":
            filtered = filtered[filtered["main_direction"] == direction]
        if min_score > 0:
            filtered = filtered[filtered["combined_score"] >= min_score]

        return filtered.nsmallest(top_n, "rank")
