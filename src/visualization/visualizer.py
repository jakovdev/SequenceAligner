from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import logging

from ..core.config import Config
from ..core.results import AnalysisResults

logger = logging.getLogger(__name__)


class PlotFactory:
    @staticmethod
    def create_distribution(data: np.ndarray, title: str) -> go.Figure:
        fig = ff.create_distplot([data], [title])
        fig.update_layout(title=title)
        return fig

    @staticmethod
    def create_heatmap(matrix: np.ndarray, title: str) -> go.Figure:
        fig = px.imshow(matrix, title=title)
        return fig

    @staticmethod
    def create_scatter(
        x: np.ndarray, y: np.ndarray, title: str, labels: Optional[np.ndarray] = None
    ) -> go.Figure:
        fig = px.scatter(x=x, y=y, color=labels, title=title)
        return fig

    @staticmethod
    def create_bar(x: List, y: List, title: str) -> go.Figure:
        fig = px.bar(x=x, y=y, title=title)
        return fig


class Visualizer:
    def __init__(self, config: Config):
        self.config = config
        self.output_dir = Path(config.output_dir) / "visualizations"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.visualizations = []

    def _save_figure(self, fig: go.Figure, name: str) -> str:
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
        output_path = self.output_dir / f"{safe_name}.html"
        fig.write_html(str(output_path))
        self.visualizations.append(str(output_path))
        return str(output_path)

    def create_visualizations(self, results: AnalysisResults) -> List[str]:
        logger.info("Creating visualizations")

        with ThreadPoolExecutor() as executor:
            futures = []

            similarity_matrix = results.get_matrix("similarity_matrix")
            futures.append(
                executor.submit(self._visualize_similarity, similarity_matrix)
            )

            feature_matrix = results.get_matrix("feature_matrix")
            feature_names = results.get_feature_names()

            try:
                labels = results.get_matrix("labels")
            except (KeyError, ValueError):
                labels = None

            futures.append(
                executor.submit(
                    self._visualize_features,
                    feature_matrix,
                    feature_names,
                    labels,
                )
            )

            if "statistics" in results.summary:
                futures.append(
                    executor.submit(
                        self._visualize_statistics, results.summary["statistics"]
                    )
                )

            if "ml_analysis" in results.summary:
                futures.append(
                    executor.submit(
                        self._visualize_ml,
                        results.summary["ml_analysis"],
                        results.get_feature_names(),
                    )
                )

            for future in futures:
                future.result()

        return self.visualizations

    def _visualize_similarity(self, matrix: np.ndarray) -> None:
        fig = PlotFactory.create_heatmap(matrix, "Sequence Similarity Matrix")
        self._save_figure(fig, "similarity_heatmap")

        triu_indices = np.triu_indices_from(matrix, k=1)
        similarities = matrix[triu_indices]

        hist_values, hist_bins = np.histogram(
            similarities, bins=50, range=(similarities.min(), similarities.max())
        )

        fig = go.Figure()
        fig.add_trace(
            go.Bar(x=hist_bins[:-1], y=hist_values, name="Histogram", opacity=0.75)
        )

        fig.update_layout(
            title="Similarity Distribution",
            xaxis_title="Similarity Score",
            yaxis_title="Count",
            template="plotly_white",
            showlegend=False,
        )

        self._save_figure(fig, "similarity_distribution")

    def _visualize_features(
        self,
        matrix: np.ndarray,
        feature_names: List[str],
        labels: Optional[np.ndarray] = None,
    ) -> None:
        df = pd.DataFrame(matrix, columns=feature_names)

        if matrix.shape[1] <= 50:
            corr = df.corr()
            fig = PlotFactory.create_heatmap(corr, "Feature Correlations")
            self._save_figure(fig, "feature_correlations")

        from sklearn.decomposition import PCA

        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(matrix)
        # variance_explained = pca.explained_variance_ratio_.sum()
        # f"PCA of Features (Variance Explained: {variance_explained:.2%})",

        fig = PlotFactory.create_scatter(
            pca_result[:, 0], pca_result[:, 1], "PCA of Features", labels
        )
        self._save_figure(fig, "feature_pca")

        for col in df.columns:
            fig = PlotFactory.create_distribution(df[col].values, f"{col} Distribution")
            self._save_figure(fig, f"feature_dist_{col}")

    def _visualize_statistics(self, stats: Dict) -> None:
        if "aa_composition" in stats:
            aa_comp = stats["aa_composition"]["aa_frequencies"]
            fig = PlotFactory.create_bar(
                list(aa_comp.keys()), list(aa_comp.values()), "Amino Acid Composition"
            )
            self._save_figure(fig, "aa_composition")

        if "property_stats" in stats:
            self._visualize_property_stats(stats["property_stats"])

    def _visualize_property_stats(self, stats: List[Dict]) -> None:
        feature_names = [
            "Length",
            "Mol Weight",
            "Charge",
            "Hydrophobicity",
            "% Hydrophobic",
            "% Charged",
            "% Polar",
            "% Aromatic",
            "% Positive",
            "% Negative",
        ]

        feature_groups = {
            "Molecular Properties": [
                "Length",
                "Mol Weight",
                "Charge",
                "Hydrophobicity",
            ],
            "Composition (%)": [
                "% Hydrophobic",
                "% Charged",
                "% Polar",
                "% Aromatic",
                "% Positive",
                "% Negative",
            ],
        }

        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=list(feature_groups.keys()),
            vertical_spacing=0.15,
            row_heights=[0.5, 0.5],
        )

        for idx, (group_name, group_features) in enumerate(feature_groups.items(), 1):
            for name, stat in zip(feature_names, stats):
                if name not in group_features:
                    continue

                scale_factor = 1000 if name == "Mol Weight" else 1.0

                fig.add_trace(
                    go.Box(
                        name=name,
                        y=[
                            stat["min"] / scale_factor,
                            stat["median"] / scale_factor,
                            stat["max"] / scale_factor,
                        ],
                        q1=[stat["quartiles"][0] / scale_factor],
                        q3=[stat["quartiles"][2] / scale_factor],
                        boxmean=True,
                        orientation="v",
                    ),
                    row=idx,
                    col=1,
                )

        fig.update_layout(
            title={"text": "Property Statistics", "x": 0.5, "xanchor": "center"},
            showlegend=False,
            template="plotly_white",
            boxmode="group",
            margin=dict(t=100, b=50, l=50, r=50),
        )

        fig.update_yaxes(title_text="Value", row=1, col=1)
        fig.update_yaxes(title_text="Percentage", row=2, col=1)

        fig.add_annotation(
            text="Note: Molecular Weight shown in kDa",
            xref="paper",
            yref="paper",
            x=0.02,
            y=1.05,
            showarrow=False,
            font=dict(size=10, color="gray"),
        )

        self._save_figure(fig, "property_stats")

    def _visualize_ml(self, ml_results: Dict, feature_names: List[str]) -> None:
        if (
            "random_forest" in ml_results
            and "feature_importance" in ml_results["random_forest"]
        ):
            importances = ml_results["random_forest"]["feature_importance"]

            if len(feature_names) == len(importances):
                labels = feature_names
            else:
                labels = [f"Feature {i}" for i in range(len(importances))]

            sorted_indices = np.argsort(importances)
            sorted_importances = [importances[i] for i in sorted_indices]
            sorted_labels = [labels[i] for i in sorted_indices]

            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    y=sorted_labels,
                    x=sorted_importances,
                    orientation="h",
                    marker=dict(color="rgba(50, 171, 96, 0.7)"),
                )
            )

            fig.update_layout(
                title="Feature Importance",
                xaxis_title="Importance",
                yaxis_title="Feature",
                template="plotly_white",
                height=max(300, 50 * len(importances)),  # Dynamic height
            )

            self._save_figure(fig, "feature_importance")

        if "roc_auc" in ml_results.get("random_forest", {}):
            roc_data = ml_results["random_forest"]["roc_auc"]
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode="lines",
                    line=dict(color="gray", dash="dash"),
                    name="Random",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[0, 0, 1],
                    y=[0, 1, 1],
                    mode="lines",
                    line=dict(color="lightgray", dash="dot"),
                    name="Perfect",
                )
            )
            fig.update_layout(
                title=f"ROC Curve (AUC = {roc_data['mean']:.3f} ± {roc_data['std']:.3f})",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                template="plotly_white",
                legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99),
            )
            self._save_figure(fig, "roc_curve")
