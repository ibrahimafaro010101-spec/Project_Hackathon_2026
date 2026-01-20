# modules/insight_engine.py

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import io
from typing import Dict, List, Any, Optional, Union, Tuple
from scipy import stats


class CustomerAdapter:

    @staticmethod
    def adapt_for_powerbi(df: pd.DataFrame) -> pd.DataFrame:
        df_export = df.copy()

        if "niveau_risque" in df_export.columns:
            df_export["risk_color"] = df_export["niveau_risque"].map({
                "Faible": "#2ECC71",
                "Moyen": "#F39C12",
                "Élevé": "#E74C3C"
            })

        if "score_risque" in df_export.columns:
            df_export["priority"] = df_export["score_risque"].apply(
                lambda x: "Haute" if x > 70 else "Moyenne" if x > 30 else "Basse"
            )

        num_cols = df_export.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            df_export[col] = df_export[col].round(2)

        return df_export

    @staticmethod
    def to_csv_string(df: pd.DataFrame) -> str:
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False, sep=';', encoding='utf-8-sig')
        return csv_buffer.getvalue()

    @staticmethod
    def to_json_api(data: Union[pd.DataFrame, Dict, List]) -> str:
        if isinstance(data, pd.DataFrame):
            json_data = {
                "metadata": {
                    "row_count": len(data),
                    "column_count": len(data.columns),
                    "generated_at": pd.Timestamp.now().isoformat()
                },
                "data": data.to_dict(orient='records')
            }
        else:
            json_data = data
        return json.dumps(json_data, ensure_ascii=False, indent=2, default=str)


class InsightEngine:

    def __init__(self):
        self.insights = []
        self.adapter = CustomerAdapter()

    def get_variable_types(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Retourne les différents types de variables disponibles"""
        variable_types = {
            'numeric': [],
            'categorical': [],
            'boolean': [],
            'datetime': [],
            'text': []
        }

        for column in df.columns:
            # Variables numériques
            if pd.api.types.is_numeric_dtype(df[column]):
                if df[column].nunique() <= 10 and df[column].dropna().shape[0] > 0:
                    # Peu de valeurs uniques = peut être utilisé comme catégoriel
                    variable_types['categorical'].append(column)
                else:
                    variable_types['numeric'].append(column)

            # Variables catégorielles
            elif pd.api.types.is_string_dtype(df[column]) or pd.api.types.is_categorical_dtype(df[column]):
                if df[column].nunique() <= 50:  # Limite pour éviter trop de modalités
                    variable_types['categorical'].append(column)
                else:
                    variable_types['text'].append(column)

            # Variables booléennes
            elif pd.api.types.is_bool_dtype(df[column]):
                variable_types['boolean'].append(column)
                variable_types['categorical'].append(column)  # Peut être utilisé comme catégoriel

            # Variables datetime
            elif pd.api.types.is_datetime64_any_dtype(df[column]):
                variable_types['datetime'].append(column)

        return variable_types

    def build_client_risk_table(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()

        data["Prime"] = pd.to_numeric(data["Prime"], errors="coerce")
        data["nb_jour_couv"] = pd.to_numeric(data["nb_jour_couv"], errors="coerce")

        data["prime_par_jour"] = data["Prime"] / data["nb_jour_couv"].replace(0, np.nan)

        data["is_avenant"] = (
            data["libop"].astype(str).str.lower().str.contains("avenant")
        ).astype(int)

        client = (
            data.groupby(["ncli", "nomncli"], dropna=False)
            .agg(
                prime_totale=("Prime", "sum"),
                prime_moyenne=("Prime", "mean"),
                prime_par_jour_moy=("prime_par_jour", "mean"),
                duree_moyenne=("nb_jour_couv", "mean"),
                duree_min=("nb_jour_couv", "min"),
                duree_max=("nb_jour_couv", "max"),
                nb_avenants=("is_avenant", "sum"),
                nb_operations=("ncli", "count"),
                dernier_prime=("Prime", "last"),
                dernier_duree=("nb_jour_couv", "last")
            )
            .reset_index()
        )

        if 'sinistre' in data.columns:
            sinistres = data.groupby("ncli")["sinistre"].sum()
            client = client.merge(sinistres.rename('frequence_sinistre'),
                                  left_on='ncli', right_index=True, how='left')

        if 'montant_sinistre' in data.columns:
            cout_moyen = data.groupby("ncli")["montant_sinistre"].mean()
            cout_total = data.groupby("ncli")["montant_sinistre"].sum()
            client = client.merge(cout_moyen.rename('cout_moyen_sinistre'),
                                  left_on='ncli', right_index=True, how='left')
            client = client.merge(cout_total.rename('cout_total_sinistres'),
                                  left_on='ncli', right_index=True, how='left')

        if 'retard_jours' in data.columns:
            retard_moyen = data.groupby("ncli")["retard_jours"].mean()
            nb_retards = (data["retard_jours"] > 0).groupby(data["ncli"]).sum()
            client = client.merge(retard_moyen.rename('retard_paiement_moyen'),
                                  left_on='ncli', right_index=True, how='left')
            client = client.merge(nb_retards.rename('nb_retards'),
                                  left_on='ncli', right_index=True, how='left')

        if 'impaye' in data.columns:
            impayes = data.groupby("ncli")["impaye"].sum()
            client = client.merge(impayes.rename('nb_impayes'),
                                  left_on='ncli', right_index=True, how='left')

        if 'statut' in data.columns:
            taux_resil = (data["statut"] == "résilié").groupby(data["ncli"]).mean()
            statut_actuel = data.groupby("ncli")["statut"].last()
            client = client.merge(taux_resil.rename('taux_resiliation'),
                                  left_on='ncli', right_index=True, how='left')
            client = client.merge(statut_actuel.rename('statut_actuel'),
                                  left_on='ncli', right_index=True, how='left')

        if 'cout_sinistres' in data.columns:
            cout_sinistres_total = data.groupby("ncli")["cout_sinistres"].sum()
            client = client.merge(cout_sinistres_total.rename('cout_sinistres_total'),
                                  left_on='ncli', right_index=True, how='left')
            client['marge_technique'] = client['prime_totale'] - client['cout_sinistres_total']
            client['loss_ratio'] = client['cout_sinistres_total'] / client['prime_totale'].replace(0, np.nan)

        if 'date_souscription' in data.columns:
            try:
                data['date_souscription'] = pd.to_datetime(data['date_souscription'], errors='coerce')
                anciennete = (pd.Timestamp.now() - data.groupby('ncli')['date_souscription'].min()).dt.days
                client = client.merge(anciennete.rename('anciennete_jours'),
                                      left_on='ncli', right_index=True, how='left')
                client['anciennete_mois'] = client['anciennete_jours'] / 30.44
            except:
                pass

        if 'date_contrat' in data.columns:
            try:
                dates_contrat = pd.to_datetime(data['date_contrat'], errors='coerce')
                nb_renouvellements = data.groupby('ncli')['date_contrat'].nunique() - 1
                client = client.merge(nb_renouvellements.rename('nb_renouvellements'),
                                      left_on='ncli', right_index=True, how='left')
            except:
                pass

        client["prime_75_percentile"] = client["prime_totale"] > client["prime_totale"].quantile(0.75)
        client["variabilite_duree"] = client["duree_max"] - client["duree_min"]

        # AJOUTER DES VARIABLES CATÉGORIELLES POUR L'ACM
        # Catégoriser la prime
        client["categorie_prime"] = pd.cut(
            client["prime_totale"],
            bins=[-1, 10000, 50000, 200000, np.inf],
            labels=["Petite", "Moyenne", "Grande", "Très grande"]
        )

        # Catégoriser la durée
        client["categorie_duree"] = pd.cut(
            client["duree_moyenne"],
            bins=[-1, 90, 180, 365, np.inf],
            labels=["Très courte", "Courte", "Moyenne", "Longue"]
        )

        # Catégoriser le nombre d'avenants
        client["categorie_avenants"] = pd.cut(
            client["nb_avenants"],
            bins=[-1, 0, 1, 3, np.inf],
            labels=["Aucun", "1", "2-3", "4+"]
        )

        # Catégoriser la fréquence de sinistres
        if 'frequence_sinistre' in client.columns:
            client["categorie_sinistres"] = pd.cut(
                client["frequence_sinistre"],
                bins=[-1, 0, 1, 3, np.inf],
                labels=["Aucun", "1", "2-3", "4+"]
            )

        # Catégoriser l'ancienneté
        if 'anciennete_mois' in client.columns:
            client["categorie_anciennete"] = pd.cut(
                client["anciennete_mois"],
                bins=[-1, 6, 24, 60, np.inf],
                labels=["Nouveau", "Récent", "Ancien", "Très ancien"]
            )

        # Catégoriser le retard de paiement
        if 'retard_paiement_moyen' in client.columns:
            client["categorie_retard"] = pd.cut(
                client["retard_paiement_moyen"],
                bins=[-1, 0, 15, 30, np.inf],
                labels=["Ponctuel", "Retard léger", "Retard moyen", "Retard important"]
            )

        client = client.replace([np.inf, -np.inf], np.nan)

        fill_defaults = {
            'prime_par_jour_moy': client['prime_par_jour_moy'].median(),
            'duree_moyenne': client['duree_moyenne'].median(),
            'prime_totale': 0,
            'nb_avenants': 0,
            'frequence_sinistre': 0,
            'cout_moyen_sinistre': 0,
            'cout_total_sinistres': 0,
            'retard_paiement_moyen': 0,
            'nb_retards': 0,
            'nb_impayes': 0,
            'taux_resiliation': 0,
            'cout_sinistres_total': 0,
            'marge_technique': client['prime_totale'],
            'loss_ratio': 0,
            'anciennete_jours': client['duree_moyenne'].median() * 2,
            'nb_renouvellements': 0
        }

        for col, default_val in fill_defaults.items():
            if col in client.columns:
                client[col] = client[col].fillna(default_val)

        numeric_cols = client.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in fill_defaults and col != 'ncli':
                client[col] = client[col].fillna(0)

        return client

    def compute_risk_score(self, client_df: pd.DataFrame) -> pd.DataFrame:
        df = client_df.copy()

        def normalize(col):
            min_val = col.min()
            max_val = col.max()
            if max_val - min_val < 1e-9:
                return pd.Series([0.5] * len(col), index=col.index)
            return (col - min_val) / (max_val - min_val)

        df["ppj_norm"] = normalize(df["prime_par_jour_moy"])
        df["avenants_norm"] = normalize(df["nb_avenants"])
        df["duree_risque"] = 1 - normalize(df["duree_moyenne"])

        indicators = {
            "ppj_norm": 0.25,
            "avenants_norm": 0.15,
            "duree_risque": 0.10
        }

        if 'frequence_sinistre' in df.columns:
            df["freq_sinistre_norm"] = normalize(df["frequence_sinistre"])
            indicators["freq_sinistre_norm"] = 0.20

        if 'cout_moyen_sinistre' in df.columns:
            df["cout_sinistre_norm"] = normalize(df["cout_moyen_sinistre"])
            indicators["cout_sinistre_norm"] = 0.15

        if 'retard_paiement_moyen' in df.columns:
            df["retard_norm"] = normalize(df["retard_paiement_moyen"])
            indicators["retard_norm"] = 0.10

        if 'nb_impayes' in df.columns:
            df["impayes_norm"] = normalize(df["nb_impayes"])
            indicators["impayes_norm"] = 0.08

        if 'taux_resiliation' in df.columns:
            df["resiliation_norm"] = normalize(df["taux_resiliation"])
            indicators["resiliation_norm"] = 0.07

        if 'loss_ratio' in df.columns:
            df["loss_ratio_norm"] = normalize(df["loss_ratio"])
            indicators["loss_ratio_norm"] = 0.05

        df["score_risque"] = 0

        total_weight = sum(indicators.values())

        for indicator, weight in indicators.items():
            if indicator in df.columns:
                df["score_risque"] += (weight / total_weight) * df[indicator]

        df["score_risque"] = df["score_risque"] * 100
        df["score_risque"] = df["score_risque"].clip(0, 100).round(1)

        df["niveau_risque"] = pd.cut(
            df["score_risque"],
            bins=[-1, 30, 70, 100],
            labels=["Faible", "Moyen", "Élevé"]
        )

        df["decision_assurance"] = df["niveau_risque"].map({
            "Faible": "Maintien des conditions - Programme fidélité",
            "Moyen": "Surveillance renforcée - Revue trimestrielle",
            "Élevé": "Révision de la prime + Entretien conseil + Contrôle"
        })

        df["priorite_action"] = df["score_risque"].apply(
            lambda x: "Haute" if x > 70 else "Moyenne" if x > 30 else "Basse"
        )

        return df

    def generate_client_insight(self, row: pd.Series, ppj_median: float) -> str:
        reasons = []

        if pd.notna(row.get("prime_par_jour_moy")) and row["prime_par_jour_moy"] > ppj_median * 1.2:
            reasons.append("prime par jour élevée")

        if pd.notna(row.get("duree_moyenne")) and row["duree_moyenne"] < 180:
            reasons.append("durée de couverture courte")

        if pd.notna(row.get("nb_avenants")) and row["nb_avenants"] >= 2:
            reasons.append("instabilité contractuelle")

        if 'frequence_sinistre' in row and row.get("frequence_sinistre", 0) > 1:
            reasons.append("fréquence sinistres élevée")

        if 'retard_paiement_moyen' in row and row.get("retard_paiement_moyen", 0) > 15:
            reasons.append("retards de paiement fréquents")

        if 'loss_ratio' in row and row.get("loss_ratio", 0) > 0.7:
            reasons.append("loss ratio défavorable")

        if not reasons:
            return "Client stable, bon profil risque"

        return f"Client à risque {row.get('niveau_risque', 'inconnu')} : " + ", ".join(reasons) + "."

    def generate_insights(self, scored_df: pd.DataFrame) -> List[str]:
        insights = []

        insights.append(f"Clients analysés : {len(scored_df):,}")

        risk_dist = scored_df["niveau_risque"].value_counts(normalize=True)
        for niveau, pct in risk_dist.items():
            insights.append(f"{niveau} risque : {pct:.1%}")

        insights.append(f"Score de risque moyen : {scored_df['score_risque'].mean():.1f}/100")

        if 'frequence_sinistre' in scored_df.columns:
            high_claim = (scored_df['frequence_sinistre'] > 1).mean()
            if high_claim > 0:
                insights.append(f"Clients multi-sinistres : {high_claim:.1%}")

        if 'retard_paiement_moyen' in scored_df.columns:
            late_payers = (scored_df['retard_paiement_moyen'] > 15).mean()
            if late_payers > 0:
                insights.append(f"Retards paiement (>15j) : {late_payers:.1%}")

        if 'loss_ratio' in scored_df.columns:
            bad_loss_ratio = (scored_df['loss_ratio'] > 0.7).mean()
            if bad_loss_ratio > 0:
                insights.append(f"Loss ratio >70% : {bad_loss_ratio:.1%}")

        self.insights = insights
        return insights

    def portfolio_risk_summary(self, scored_df: pd.DataFrame) -> Dict[str, float]:
        summary = {
            "clients_total": len(scored_df),
            "pct_risque_eleve": round((scored_df["niveau_risque"] == "Élevé").mean() * 100, 1),
            "pct_risque_moyen": round((scored_df["niveau_risque"] == "Moyen").mean() * 100, 1),
            "pct_risque_faible": round((scored_df["niveau_risque"] == "Faible").mean() * 100, 1),
            "score_moyen": round(scored_df["score_risque"].mean(), 1),
            "prime_totale_portefeuille": round(scored_df["prime_totale"].sum(), 0)
        }

        if 'frequence_sinistre' in scored_df.columns:
            summary["sinistres_total"] = int(scored_df["frequence_sinistre"].sum())
            summary["freq_sinistre_moy"] = round(scored_df["frequence_sinistre"].mean(), 2)

        if 'cout_total_sinistres' in scored_df.columns:
            summary["cout_sinistres_total"] = round(scored_df["cout_total_sinistres"].sum(), 0)

        if 'marge_technique' in scored_df.columns:
            summary["marge_technique_totale"] = round(scored_df["marge_technique"].sum(), 0)
            summary["marge_moyenne"] = round(scored_df["marge_technique"].mean(), 0)

        return summary

    def create_dashboard_visualizations(self, scored_df: pd.DataFrame) -> Dict[str, go.Figure]:
        figs = {}

        if "score_risque" in scored_df.columns:
            figs["hist_score"] = px.histogram(
                scored_df,
                x="score_risque",
                nbins=20,
                title="Distribution du score de risque",
                labels={"score_risque": "Score de risque", "count": "Nombre de clients"}
            )

        if all(col in scored_df.columns for col in ["prime_par_jour_moy", "score_risque", "niveau_risque"]):
            figs["ppj_vs_score"] = px.scatter(
                scored_df,
                x="prime_par_jour_moy",
                y="score_risque",
                color="niveau_risque",
                title="Prime par jour vs Score de risque",
                hover_data=["nomncli", "nb_avenants",
                            "frequence_sinistre"] if 'frequence_sinistre' in scored_df.columns else ["nomncli",
                                                                                                     "nb_avenants"]
            )

        if 'frequence_sinistre' in scored_df.columns:
            figs["sinistres_vs_risque"] = px.scatter(
                scored_df,
                x="frequence_sinistre",
                y="score_risque",
                color="niveau_risque",
                title="Fréquence sinistres vs Score de risque",
                size="prime_totale" if 'prime_totale' in scored_df.columns else None
            )

        numeric_cols = scored_df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 2:
            important_cols = ['score_risque', 'prime_par_jour_moy', 'frequence_sinistre',
                              'retard_paiement_moyen', 'loss_ratio', 'nb_avenants',
                              'duree_moyenne', 'prime_totale']
            available_cols = [col for col in important_cols if col in numeric_cols]

            if len(available_cols) >= 3:
                corr_matrix = scored_df[available_cols].corr().round(2)
                figs["heatmap"] = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    colorscale='RdBu',
                    zmin=-1,
                    zmax=1,
                    text=corr_matrix.values,
                    texttemplate='%{text:.2f}'
                ))
                figs["heatmap"].update_layout(title="Corrélation entre indicateurs")

        return figs

    def dashboard_summary_table(self, scored_df: pd.DataFrame) -> pd.DataFrame:
        columns_to_show = ["nomncli", "score_risque", "niveau_risque", "decision_assurance", "priorite_action"]

        if 'frequence_sinistre' in scored_df.columns:
            columns_to_show.append("frequence_sinistre")

        if 'retard_paiement_moyen' in scored_df.columns:
            columns_to_show.append("retard_paiement_moyen")

        if 'marge_technique' in scored_df.columns:
            columns_to_show.append("marge_technique")

        return (
            scored_df[columns_to_show]
            .sort_values("score_risque", ascending=False)
            .head(15)
        )

    def generate_narrative_report(self, scored_df: pd.DataFrame) -> str:
        summary = self.portfolio_risk_summary(scored_df)

        report = []
        report.append("Rapport d'Analyse - Assurance LIK")
        report.append("\nSynthèse portefeuille")
        report.append(f"- Clients analysés : {summary['clients_total']:,}")
        report.append(f"- Score de risque moyen : {summary['score_moyen']}/100")
        report.append(f"- Clients à risque élevé : {summary['pct_risque_eleve']}%")
        report.append(f"- Prime totale portefeuille : {summary['prime_totale_portefeuille']:,} MAD")

        if 'sinistres_total' in summary:
            report.append(f"- Nombre total sinistres : {summary['sinistres_total']}")

        if 'cout_sinistres_total' in summary:
            report.append(f"- Coût total sinistres : {summary['cout_sinistres_total']:,} MAD")

        if 'marge_technique_totale' in summary:
            report.append(f"- Marge technique totale : {summary['marge_technique_totale']:,} MAD")

        report.append("\nFacteurs clés de risque")
        report.append("- Prime par jour élevée (>120% médiane)")
        report.append("- Fréquence sinistres élevée (>1 par an)")
        report.append("- Retards de paiement fréquents (>15 jours)")
        report.append("- Instabilité contractuelle (≥2 avenants)")
        report.append("- Durée couverture courte (<180 jours)")

        report.append("\nRecommandations stratégiques")
        report.append("1. Tarification dynamique : Ajuster primes selon score risque")
        report.append("2. Programme fidélisation : Clients stables à faible risque")
        report.append("3. Surveillance renforcée : Clients risque moyen (revue trimestrielle)")
        report.append("4. Actions correctives : Clients risque élevé (entretien conseil)")
        report.append("5. Prévention sinistres : Cibler clients multi-sinistrés")

        report.append("\nIndicateurs de performance")
        report.append("- Objectif : Réduction résiliation 15% dans 12 mois")
        report.append("- Suivi : Score risque moyen < 40/100")
        report.append("- Rentabilité : Marge technique > 25% prime totale")
        report.append("- Satisfaction : Taux renouvellement > 85%")

        return "\n".join(report)

    def create_univariate_histogram(self, df: pd.DataFrame, column: str, bins: int = 30) -> go.Figure:
        fig = px.histogram(
            df,
            x=column,
            nbins=bins,
            title=f"Distribution de {column}",
            labels={column: column, "count": "Fréquence"},
            marginal="box",
            opacity=0.7
        )
        return fig

    def create_univariate_boxplot(self, df: pd.DataFrame, column: str) -> go.Figure:
        fig = go.Figure()
        fig.add_trace(go.Box(y=df[column], name=column))
        fig.update_layout(title=f"Boîte à moustaches de {column}")
        return fig

    def create_bivariate_scatter(self, df: pd.DataFrame, x_col: str, y_col: str,
                                 color_col: Optional[str] = None) -> go.Figure:
        if color_col and color_col in df.columns:
            fig = px.scatter(df, x=x_col, y=y_col, color=color_col, title=f"{x_col} vs {y_col}")
        else:
            fig = px.scatter(df, x=x_col, y=y_col, title=f"{x_col} vs {y_col}")
        return fig

    def prepare_powerbi_data(self, scored_df: pd.DataFrame) -> Dict:
        data = scored_df.copy()

        risk_dist = data.groupby("niveau_risque").agg({
            "ncli": "count",
            "prime_totale": "sum",
            "score_risque": "mean",
            "prime_par_jour_moy": "mean"
        }).reset_index()
        risk_dist = risk_dist.rename(columns={"ncli": "count"})

        top_risky = data.nlargest(10, "score_risque")[[
            "ncli", "nomncli", "score_risque", "niveau_risque",
            "prime_totale", "nb_avenants", "decision_assurance"
        ]]

        overall_metrics = self.portfolio_risk_summary(data)

        return {
            "risk_distribution": risk_dist.to_dict("records"),
            "top_risky_clients": top_risky.to_dict("records"),
            "overall_metrics": overall_metrics,
            "raw_data_sample": data.head(100).to_dict("records")
        }

    def prepare_powerbi_dataframe(self, scored_df: pd.DataFrame) -> pd.DataFrame:
        return self.adapter.adapt_for_powerbi(scored_df)

    def export_to_csv(self, scored_df: pd.DataFrame) -> str:
        df_export = self.prepare_powerbi_dataframe(scored_df)

        essential_cols = [
            "ncli", "nomncli", "score_risque", "niveau_risque",
            "prime_totale", "prime_par_jour_moy", "duree_moyenne",
            "nb_avenants", "decision_assurance", "priority"
        ]

        if 'frequence_sinistre' in df_export.columns:
            essential_cols.append("frequence_sinistre")

        if 'retard_paiement_moyen' in df_export.columns:
            essential_cols.append("retard_paiement_moyen")

        if 'marge_technique' in df_export.columns:
            essential_cols.append("marge_technique")

        if 'loss_ratio' in df_export.columns:
            essential_cols.append("loss_ratio")

        available_cols = [col for col in essential_cols if col in df_export.columns]
        df_export = df_export[available_cols]

        return self.adapter.to_csv_string(df_export)

    def export_to_json(self, powerbi_data: Dict) -> str:
        return self.adapter.to_json_api(powerbi_data)

    # ==============================================
    # MÉTHODES POUR ANALYSE MULTIVARIÉE
    # ==============================================

    def get_available_variables_for_pca(self, df: pd.DataFrame,
                                        exclude_cols: list = None) -> List[str]:
        """Retourne les variables disponibles pour l'ACP"""
        if exclude_cols is None:
            exclude_cols = ['ncli', 'nomncli', 'score_risque',
                            'niveau_risque', 'decision_assurance',
                            'priorite_action', 'prime_75_percentile']

        # Variables numériques
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Filtrer les colonnes à exclure
        available_vars = [col for col in numeric_cols if col not in exclude_cols]

        # Retirer les colonnes avec trop de valeurs manquantes
        available_vars = [col for col in available_vars
                          if df[col].notna().sum() > len(df) * 0.5]  # Au moins 50% de données

        return available_vars

    def get_available_variables_for_acm(self, df: pd.DataFrame) -> List[str]:
        """Retourne les variables disponibles pour l'ACM"""
        categorical_vars = []

        for column in df.columns:
            # Variables catégorielles (object, category)
            if pd.api.types.is_string_dtype(df[column]) or pd.api.types.is_categorical_dtype(df[column]):
                unique_vals = df[column].nunique()
                if 2 <= unique_vals <= 20:  # Entre 2 et 20 modalités
                    categorical_vars.append(column)

            # Variables numériques avec peu de valeurs uniques (peuvent être catégorisées)
            elif pd.api.types.is_numeric_dtype(df[column]):
                unique_vals = df[column].nunique()
                if 2 <= unique_vals <= 10:  # Entre 2 et 10 modalités
                    categorical_vars.append(column)

        return categorical_vars

    def perform_pca_analysis(self, df: pd.DataFrame,
                             selected_variables: List[str] = None,
                             n_components: int = 3,
                             scale: bool = True) -> Dict:
        """Analyse en Composantes Principales (ACP) avec sélection de variables"""
        try:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler

            # Si aucune variable n'est sélectionnée, utiliser toutes les disponibles
            if selected_variables is None or len(selected_variables) == 0:
                selected_variables = self.get_available_variables_for_pca(df)

            # Vérifier qu'il y a assez de variables
            if len(selected_variables) < 2:
                return {"error": f"ACP nécessite au moins 2 variables. Variables disponibles: {selected_variables}"}

            # Préparation des données
            X = df[selected_variables].copy()

            # Gérer les valeurs manquantes
            X = X.fillna(X.median())

            # Standardisation si demandée
            if scale:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
            else:
                X_scaled = X.values
                scaler = None

            # Limiter le nombre de composantes
            n_components = min(n_components, len(selected_variables), X.shape[0])

            # Application de l'ACP
            pca = PCA(n_components=n_components)
            principal_components = pca.fit_transform(X_scaled)

            # Calculer les contributions des variables
            loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

            # Créer un DataFrame pour les loadings
            loadings_df = pd.DataFrame(
                loadings,
                columns=[f'PC{i + 1}' for i in range(n_components)],
                index=selected_variables
            )

            return {
                'success': True,
                'principal_components': principal_components,
                'explained_variance': pca.explained_variance_ratio_,
                'explained_variance_cumulative': np.cumsum(pca.explained_variance_ratio_),
                'loadings': loadings,
                'loadings_df': loadings_df,
                'feature_names': selected_variables,
                'scaler': scaler,
                'pca_model': pca,
                'n_components': n_components,
                'selected_variables': selected_variables,
                'total_variance_explained': sum(pca.explained_variance_ratio_) * 100
            }

        except ImportError:
            return {"error": "scikit-learn n'est pas installé. Exécutez: pip install scikit-learn"}
        except Exception as e:
            return {"error": f"Erreur lors de l'ACP: {str(e)}"}

    def perform_acm_analysis(self, df: pd.DataFrame,
                             selected_variables: List[str] = None) -> Dict:
        """Analyse des Correspondances Multiples (ACM) avec sélection de variables"""
        try:
            from prince import MCA

            # Si aucune variable n'est sélectionnée, utiliser toutes les disponibles
            if selected_variables is None or len(selected_variables) == 0:
                selected_variables = self.get_available_variables_for_acm(df)

            # Vérifier qu'il y a assez de variables
            if len(selected_variables) < 2:
                available_vars = self.get_available_variables_for_acm(df)
                return {
                    "error": f"ACM nécessite au moins 2 variables catégorielles. Variables disponibles: {available_vars}"}

            # Préparer les données pour MCA
            mca_data = df[selected_variables].copy()

            # Nettoyer les données
            for col in selected_variables:
                mca_data[col] = mca_data[col].astype(str)
                mca_data[col] = mca_data[col].fillna('Manquant')
                mca_data[col] = mca_data[col].str.strip()

                # Regrouper les modalités trop rares
                value_counts = mca_data[col].value_counts()
                rare_categories = value_counts[value_counts < len(mca_data) * 0.05].index
                if len(rare_categories) > 0:
                    mca_data[col] = mca_data[col].replace(list(rare_categories), 'Autres')

            # Vérifier la diversité après nettoyage
            diversity_check = {}
            for col in selected_variables:
                unique_vals = mca_data[col].nunique()
                diversity_check[col] = {
                    'unique': unique_vals,
                    'values': mca_data[col].unique()[:5].tolist()
                }

            # Appliquer MCA
            mca = MCA(
                n_components=2,
                random_state=42
            )
            mca.fit(mca_data)

            # Transformer les données
            transformed = mca.transform(mca_data)

            # Obtenir les coordonnées des colonnes
            try:
                column_coords = mca.column_coordinates(mca_data)
            except:
                column_coords = None

            # Calculer la variance expliquée
            try:
                explained_inertia = mca.explained_inertia_
            except AttributeError:
                try:
                    eigenvalues = mca.eigenvalues_
                    total_inertia = sum(eigenvalues)
                    explained_inertia = [eig / total_inertia for eig in eigenvalues[:2]]
                except:
                    explained_inertia = [0.5, 0.3]

            return {
                'success': True,
                'variables': selected_variables,
                'n_variables': len(selected_variables),
                'model': mca,
                'transformed_data': transformed,
                'eigenvalues': getattr(mca, 'eigenvalues_', [1, 0.5]),
                'total_inertia': getattr(mca, 'total_inertia_', 1.0),
                'explained_inertia': explained_inertia,
                'row_coordinates': transformed,
                'column_coordinates': column_coords,
                'mca_data': mca_data,
                'diversity_check': diversity_check,
                'selected_variables': selected_variables
            }

        except ImportError:
            return {"error": "La bibliothèque 'prince' n'est pas installée. Exécutez: pip install prince"}
        except Exception as e:
            import traceback
            return {"error": f"Erreur lors de l'ACM: {str(e)}"}

    def create_mca_visualization(self, mca_result, df_with_original_data):
        """Crée une visualisation pour l'ACM."""
        try:
            if 'error' in mca_result:
                return None

            transformed = mca_result.get('row_coordinates')

            if transformed is None:
                return None

            # Créer un DataFrame pour la visualisation
            vis_df = pd.DataFrame({
                'Axe 1': transformed.iloc[:, 0],
                'Axe 2': transformed.iloc[:, 1]
            })

            # Ajouter des informations supplémentaires si disponibles
            if 'niveau_risque' in df_with_original_data.columns:
                vis_df['niveau_risque'] = df_with_original_data['niveau_risque'].values
            if 'score_risque' in df_with_original_data.columns:
                vis_df['score_risque'] = df_with_original_data['score_risque'].values
            if 'prime_totale' in df_with_original_data.columns:
                vis_df['prime_totale'] = df_with_original_data['prime_totale'].values

            # Créer le graphique
            fig = go.Figure()

            # Si nous avons des niveaux de risque, colorer par niveau
            if 'niveau_risque' in vis_df.columns:
                niveaux = vis_df['niveau_risque'].unique()
                colors = {'Faible': '#2ECC71', 'Moyen': '#F39C12', 'Élevé': '#E74C3C'}

                for niveau in niveaux:
                    subset = vis_df[vis_df['niveau_risque'] == niveau]
                    fig.add_trace(go.Scatter(
                        x=subset['Axe 1'],
                        y=subset['Axe 2'],
                        mode='markers',
                        name=niveau,
                        marker=dict(
                            size=12,
                            color=colors.get(niveau, '#3498DB'),
                            line=dict(width=1, color='white')
                        ),
                        hovertemplate=(
                                '<b>Axe 1</b>: %{x:.3f}<br>' +
                                '<b>Axe 2</b>: %{y:.3f}<br>' +
                                '<b>Niveau risque</b>: ' + str(niveau) + '<br>' +
                                '<b>Score</b>: %{customdata[0]:.1f}<br>' +
                                '<b>Prime totale</b>: %{customdata[1]:,.0f} MAD' +
                                '<extra></extra>'
                        ),
                        customdata=subset[['score_risque', 'prime_totale']].values
                        if 'score_risque' in subset.columns and 'prime_totale' in subset.columns else None
                    ))
            else:
                # Sinon, juste des points
                fig.add_trace(go.Scatter(
                    x=vis_df['Axe 1'],
                    y=vis_df['Axe 2'],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color='#3498DB'
                    )
                ))

            # Ajouter les cercles d'inertie
            explained_inertia = mca_result.get('explained_inertia', [0, 0])
            if len(explained_inertia) >= 2:
                title_text = f'Analyse des Correspondances Multiples (ACM)<br>Variance expliquée: Axe 1: {explained_inertia[0] * 100:.1f}%, Axe 2: {explained_inertia[1] * 100:.1f}%<br>Variables: {", ".join(mca_result.get("selected_variables", [])[:3])}...'
            else:
                title_text = 'Analyse des Correspondances Multiples (ACM)'

            fig.update_layout(
                title=title_text,
                xaxis_title='Axe 1',
                yaxis_title='Axe 2',
                hovermode='closest',
                showlegend=True,
                plot_bgcolor='white',
                width=800,
                height=600
            )

            # Ajouter une grille
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

            return fig

        except Exception as e:
            print(f"Erreur lors de la création de la visualisation MCA: {e}")
            return None

    def create_mca_variable_plot(self, mca_result):
        """Crée un graphique montrant les variables dans l'espace ACM - Version améliorée."""
        try:
            if 'error' in mca_result or 'column_coordinates' not in mca_result:
                return None

            column_coords = mca_result['column_coordinates']

            # VÉRIFICATION CRITIQUE : S'assurer qu'il y a suffisamment de variation
            if len(column_coords) < 5:
                # Créer un message d'information plutôt qu'un graphique vide
                fig = go.Figure()
                fig.add_annotation(
                    text="⚠️ Trop peu de modalités distinctes pour l'ACM<br>Les données catégorielles doivent avoir plusieurs modalités différentes",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5,
                    showarrow=False,
                    font=dict(size=14, color="red"),
                    align="center"
                )
                fig.update_layout(
                    title="Information sur l'ACM",
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                    plot_bgcolor='white'
                )
                return fig

            # Vérifier la variance des coordonnées
            if column_coords.iloc[:, 0].std() < 0.1 and column_coords.iloc[:, 1].std() < 0.1:
                # Créer un message d'avertissement
                fig = go.Figure()
                fig.add_annotation(
                    text="⚠️ Variance insuffisante dans les données<br>Les modalités sont trop similaires pour une ACM significative",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5,
                    showarrow=False,
                    font=dict(size=14, color="orange"),
                    align="center"
                )
                fig.update_layout(
                    title="Avertissement - Données peu variées",
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                    plot_bgcolor='white'
                )
                return fig

            # NORMALISATION : Ajouter un petit bruit pour éviter la superposition exacte
            noise_level = 0.01  # Niveau de bruit faible
            noisy_coords = column_coords.copy()

            # Ajouter un bruit unique pour chaque modalité
            np.random.seed(42)  # Pour la reproductibilité
            for i in range(len(noisy_coords)):
                noisy_coords.iloc[i, 0] += np.random.uniform(-noise_level, noise_level)
                noisy_coords.iloc[i, 1] += np.random.uniform(-noise_level, noise_level)

            # Créer le graphique avec les positions uniques
            fig = go.Figure()

            # Tracer chaque modalité avec une position légèrement différente
            for i, modalite in enumerate(noisy_coords.index):
                # Tronquer les noms trop longs pour la lisibilité
                modalite_display = modalite[:30] + "..." if len(modalite) > 30 else modalite

                fig.add_trace(go.Scatter(
                    x=[noisy_coords.iloc[i, 0]],
                    y=[noisy_coords.iloc[i, 1]],
                    mode='markers+text',
                    marker=dict(
                        size=12,
                        color='#E74C3C',
                        opacity=0.8,
                        line=dict(width=1, color='black')
                    ),
                    text=[modalite_display],
                    textposition="top center",
                    name=modalite_display,
                    hovertemplate=f'<b>{modalite}</b><br>' +
                                  f'Axe 1: {column_coords.iloc[i, 0]:.3f}<br>' +
                                  f'Axe 2: {column_coords.iloc[i, 1]:.3f}<br>' +
                                  '<extra></extra>',
                    customdata=[[modalite, column_coords.iloc[i, 0], column_coords.iloc[i, 1]]]
                ))

            # Calculer les limites adaptées
            x_min, x_max = noisy_coords.iloc[:, 0].min(), noisy_coords.iloc[:, 0].max()
            y_min, y_max = noisy_coords.iloc[:, 1].min(), noisy_coords.iloc[:, 1].max()

            # Ajouter une marge pour la lisibilité
            x_margin = max(0.1, (x_max - x_min) * 0.2)
            y_margin = max(0.1, (y_max - y_min) * 0.2)

            fig.update_layout(
                title='Position des modalités dans l\'espace ACM',
                xaxis_title='Axe 1',
                yaxis_title='Axe 2',
                hovermode='closest',
                showlegend=False,
                plot_bgcolor='white',
                width=900,
                height=600,
                xaxis=dict(
                    range=[x_min - x_margin, x_max + x_margin],
                    gridcolor='lightgray',
                    gridwidth=1
                ),
                yaxis=dict(
                    range=[y_min - y_margin, y_max + y_margin],
                    gridcolor='lightgray',
                    gridwidth=1
                )
            )

            return fig

        except Exception as e:
            print(f"Erreur création graphique variables ACM: {e}")
            return None

    def generate_mca_summary(self, mca_result):
        """Génère un résumé texte de l'ACM."""
        if 'error' in mca_result:
            return f"Erreur: {mca_result['error']}"

        summary = []
        summary.append("# Résumé de l'Analyse des Correspondances Multiples")
        summary.append(f"**Nombre de variables analysées:** {mca_result['n_variables']}")
        summary.append(f"**Inertie totale:** {mca_result.get('total_inertia', 0):.3f}")

        if 'explained_inertia' in mca_result and len(mca_result['explained_inertia']) >= 2:
            explained = mca_result['explained_inertia']
            summary.append(f"**Variance expliquée par l'axe 1:** {explained[0] * 100:.1f}%")
            summary.append(f"**Variance expliquée par l'axe 2:** {explained[1] * 100:.1f}%")
            summary.append(f"**Variance totale expliquée:** {(explained[0] + explained[1]) * 100:.1f}%")

        summary.append("\n**Variables analysées:**")
        for var in mca_result.get('variables', []):
            summary.append(f"- {var}")

        # Ajouter les informations de diversité si disponibles
        if 'diversity_check' in mca_result:
            summary.append("\n**Diversité des variables:**")
            for var, info in mca_result['diversity_check'].items():
                summary.append(f"- {var}: {info['unique']} modalités uniques")

        return "\n".join(summary)

    def create_pca_visualization(self, pca_result, df_with_original_data=None):
        """Crée une visualisation pour l'ACP."""
        try:
            if 'error' in pca_result:
                return None

            # Graphique 1: Variance expliquée
            fig_var = go.Figure()

            explained_variance = pca_result['explained_variance']
            explained_variance_cumulative = pca_result['explained_variance_cumulative']

            fig_var.add_trace(go.Bar(
                x=[f'PC{i + 1}' for i in range(len(explained_variance))],
                y=explained_variance * 100,
                name='Variance expliquée (%)',
                marker_color='#3498DB'
            ))

            fig_var.add_trace(go.Scatter(
                x=[f'PC{i + 1}' for i in range(len(explained_variance_cumulative))],
                y=explained_variance_cumulative * 100,
                name='Variance cumulée (%)',
                mode='lines+markers',
                line=dict(color='#E74C3C', width=2),
                marker=dict(size=8)
            ))

            fig_var.update_layout(
                title=f'Variance expliquée par les composantes principales<br>Total: {pca_result["total_variance_explained"]:.1f}%',
                xaxis_title='Composantes principales',
                yaxis_title='Pourcentage de variance (%)',
                hovermode='x unified',
                plot_bgcolor='white',
                width=800,
                height=500
            )

            # Graphique 2: Cercle des corrélations (pour PC1 et PC2)
            if pca_result['loadings_df'] is not None and len(pca_result['loadings_df']) > 0:
                loadings = pca_result['loadings_df']

                fig_circle = go.Figure()

                # Tracer les variables
                for i, var in enumerate(loadings.index):
                    fig_circle.add_trace(go.Scatter(
                        x=[0, loadings.iloc[i, 0]],
                        y=[0, loadings.iloc[i, 1]],
                        mode='lines',
                        line=dict(color='lightgray', width=1),
                        showlegend=False,
                        hoverinfo='skip'
                    ))

                    fig_circle.add_trace(go.Scatter(
                        x=[loadings.iloc[i, 0]],
                        y=[loadings.iloc[i, 1]],
                        mode='markers+text',
                        marker=dict(size=10, color='#2ECC71'),
                        text=[var],
                        textposition="top center",
                        name=var,
                        hovertemplate=f'<b>{var}</b><br>' +
                                      f'PC1: {loadings.iloc[i, 0]:.3f}<br>' +
                                      f'PC2: {loadings.iloc[i, 1]:.3f}<br>' +
                                      '<extra></extra>'
                    ))

                # Ajouter le cercle unitaire
                theta = np.linspace(0, 2 * np.pi, 100)
                fig_circle.add_trace(go.Scatter(
                    x=np.cos(theta),
                    y=np.sin(theta),
                    mode='lines',
                    line=dict(color='gray', dash='dash'),
                    name='Cercle unitaire',
                    showlegend=False
                ))

                fig_circle.update_layout(
                    title='Cercle des corrélations (PC1 vs PC2)',
                    xaxis_title='PC1',
                    yaxis_title='PC2',
                    hovermode='closest',
                    plot_bgcolor='white',
                    width=700,
                    height=600,
                    xaxis=dict(range=[-1.1, 1.1]),
                    yaxis=dict(range=[-1.1, 1.1])
                )

                return {'variance_plot': fig_var, 'correlation_circle': fig_circle}

            return {'variance_plot': fig_var}

        except Exception as e:
            print(f"Erreur lors de la création de la visualisation ACP: {e}")
            return None

    def perform_clustering(self, df: pd.DataFrame, n_clusters: int = 3,
                           features: list = None, scale: bool = True):
        """Clustering K-means"""
        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler

            # Sélection des features
            if features is None:
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                exclude_cols = ['ncli', 'score_risque', 'prime_75_percentile']
                features = [col for col in numeric_cols if col not in exclude_cols][:5]

            if len(features) < 2:
                return None

            # Préparation des données
            X = df[features].fillna(0)

            if scale:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
            else:
                X_scaled = X.values

            # Application du clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X_scaled)

            return {
                'clusters': clusters,
                'cluster_centers': kmeans.cluster_centers_,
                'inertia': kmeans.inertia_,
                'features': features,
                'scaler': scaler if scale else None
            }
        except ImportError:
            raise ImportError("scikit-learn n'est pas installé. Exécutez: pip install scikit-learn")
        except Exception as e:
            raise Exception(f"Erreur lors du clustering: {str(e)}")

    def perform_afc_analysis(self, df: pd.DataFrame, row_var: str, col_var: str):
        """Analyse Factorielle des Correspondances (AFC)"""
        try:
            # Création du tableau de contingence
            contingency_table = pd.crosstab(
                df[row_var].fillna('Manquant').astype(str),
                df[col_var].fillna('Manquant').astype(str)
            )

            # Test du Chi²
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

            return {
                'contingency_table': contingency_table,
                'chi2_test': {
                    'chi2': chi2,
                    'p_value': p_value,
                    'dof': dof,
                    'expected': expected
                },
                'row_var': row_var,
                'col_var': col_var
            }
        except Exception as e:
            raise Exception(f"Erreur lors de l'AFC: {str(e)}")

    def perform_acf_analysis(self, series: pd.Series, lags: int = 20):
        """Analyse de la Fonction d'Autocorrélation (ACF)"""
        try:
            from statsmodels.graphics.tsaplots import plot_acf
            import matplotlib.pyplot as plt
            import io

            # Création du graphique ACF
            fig, ax = plt.subplots(figsize=(10, 4))
            plot_acf(series.dropna(), ax=ax, lags=min(lags, len(series) // 2))
            plt.title("Fonction d'Autocorrélation (ACF)")
            plt.close(fig)

            # Conversion en image pour Streamlit
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100)
            buf.seek(0)

            return {
                'figure': fig,
                'image_buffer': buf,
                'series_name': series.name if hasattr(series, 'name') else 'Series'
            }
        except ImportError:
            raise ImportError("statsmodels n'est pas installé. Exécutez: pip install statsmodels")
        except Exception as e:
            raise Exception(f"Erreur lors de l'ACF: {str(e)}")

    def get_customer_adapter(self) -> CustomerAdapter:
        return self.adapter