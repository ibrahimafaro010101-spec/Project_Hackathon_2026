import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class InsightEngine:
    """
    Moteur de génération d'insights et visualisations.
    """
    def __init__(self):
        self.insights = []
    
    def generate_insights(self, df: pd.DataFrame, predictions: pd.Series = None):
        """Génère des insights automatiques à partir des données."""
        insights_list = []
        
        # Insight 1: Taux de renouvellement global
        if 'renewed' in df.columns:
            renewal_rate = df['renewed'].mean()
            insights_list.append(f"📊 **Taux de renouvellement global** : {renewal_rate:.1%}")
        
        # Insight 2: Prime moyenne
        if 'current_premium' in df.columns:
            avg_premium = df['current_premium'].mean()
            insights_list.append(f"💰 **Prime moyenne** : {avg_premium:,.0f} €")
        
        # Insight 3: Corrélation prime/renouvellement
        if 'current_premium' in df.columns and 'renewed' in df.columns:
            correlation = df['current_premium'].corr(df['renewed'])
            insights_list.append(f"🔗 **Corrélation prime/renouvellement** : {correlation:.2f}")
        
        # Insight 4: Top risques
        if predictions is not None and 'client_id' in df.columns:
            high_risk = df[predictions < 0.3].head(3)
            if len(high_risk) > 0:
                insights_list.append("⚠️ **Clients à haut risque** : " + 
                                   ", ".join(high_risk['client_id'].astype(str).tolist()))
        
        self.insights = insights_list
        return insights_list
    
    def create_dashboard_visualizations(self, df: pd.DataFrame, predictions: pd.Series = None):
        """Crée des visualisations interactives pour le dashboard."""
        figs = {}
        
        # 1. Distribution des probabilités de renouvellement
        if predictions is not None:
            fig1 = px.histogram(
                x=predictions,
                nbins=20,
                title="Distribution des probabilités de renouvellement",
                labels={'x': 'Probabilité', 'y': 'Nombre de contrats'}
            )
            fig1.update_layout(bargap=0.1)
            figs['distribution'] = fig1
        
        # 2. Prime vs Renouvellement
        if 'current_premium' in df.columns and predictions is not None:
            fig2 = px.scatter(
                df,
                x='current_premium',
                y=predictions,
                color=df['renewed'] if 'renewed' in df.columns else None,
                title="Prime vs Probabilité de renouvellement",
                labels={'current_premium': 'Prime (€)', 'y': 'Probabilité'}
            )
            figs['premium_vs_renewal'] = fig2
        
        # 3. Feature importance (si disponible)
        # Cette partie nécessite d'avoir l'importance des features du modèle
        
        return figs
    
    def generate_narrative_report(self, df: pd.DataFrame):
        """Génère un rapport narratif automatisé."""
        report = []
        report.append("# 📈 Rapport d'Analyse des Polices d'Assurance")
        report.append(f"**Période analysée** : {len(df)} contrats")
        
        if 'start_date' in df.columns:
            report.append(f"**Plage temporelle** : {df['start_date'].min().date()} au {df['start_date'].max().date()}")
        
        if 'renewed' in df.columns:
            renewal_pct = df['renewed'].mean() * 100
            report.append(f"**Taux de renouvellement** : {renewal_pct:.1f}%")
            report.append(f"**Contrats à risque** : {len(df[df['renewed'] == 0])} ({100-renewal_pct:.1f}%)")
        
        report.append("\n## Recommandations stratégiques :")
        report.append("1. **Cibler les clients avec primes > 150% de la moyenne**")
        report.append("2. **Renforcer la fidélisation 60 jours avant expiration**")
        report.append("3. **Offrir des rabais ciblés aux clients à risque modéré**")
        
        return "\n\n".join(report)