import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Ajouter le dossier modules au path
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

from nlp_engine import NLOEngine
from data_prep_engine import DataPrepEngine
from predictive_engine import PredictiveEngine
from insight_engine import InsightEngine

# Configuration de la page
st.set_page_config(
    page_title="RenewAI Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Titre principal
st.title("🤖 RenewAI - Tableau de Bord Intelligent d'Analyse des Renouvellements")
st.markdown("---")

# Initialisation des moteurs
@st.cache_resource
def init_engines():
    return {
        'nlp': NLOEngine(),
        'data_prep': DataPrepEngine(),
        'predictive': PredictiveEngine(),
        'insight': InsightEngine()
    }

engines = init_engines()

# Sidebar pour la navigation
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/insurance.png", width=80)
    st.title("Navigation")
    
    section = st.radio(
        "Sélectionnez une section :",
        ["📊 Tableau de Bord", "🗣️ Assistant NLQ", "🤖 Modèle Prédictif", "🔍 Insights IA", "📈 Data Quality"]
    )
    
    st.markdown("---")
    st.info("**Hackathon Insurance Analytics**\n\nPrédiction intelligente du renouvellement des polices")

# Section 1: Tableau de Bord Principal
if section == "📊 Tableau de Bord":
    st.header("📊 Vue d'ensemble")
    
    # Upload de données
    uploaded_file = st.file_uploader("📁 Téléversez votre fichier de données (CSV)", type=['csv'])


    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Contrats analysés", "1,247", "+12%")
    with col2:
        st.metric("Taux de renouvellement", "87.3%", "+2.1%")
    with col3:
        st.metric("Prime moyenne", "€1,245", "-3.2%")
    with col4:
        st.metric("Risque élevé", "34", "+5")

    # Graphiques principaux
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Distribution des risques")
        # Graphique exemple
        fig1 = go.Figure(data=[go.Bar(
            x=['Faible', 'Modéré', 'Élevé', 'Critique'],
            y=[650, 320, 210, 67],
            marker_color=['green', 'yellow', 'orange', 'red']
        )])
        st.plotly_chart(fig1, use_container_width=True)

    with col_right:
        st.subheader("Renouvellement par segment")
        fig2 = go.Figure(data=[go.Pie(
            labels=['Renouvelés', 'Résiliés'],
            values=[1087, 160],
            hole=.3
        )])
        st.plotly_chart(fig2, use_container_width=True)

    # Dernières prédictions
    st.subheader("🔔 Alertes récentes")
    alert_data = pd.DataFrame({
        'Client': ['CLT-78901', 'CLT-78902', 'CLT-78903'],
        'Probabilité': [0.12, 0.23, 0.31],
        'Risque': ['Critique', 'Élevé', 'Modéré'],
        'Recommandation': ['Contact immédiat', 'Offre de fidélité', 'Surveillance']
    })
    st.dataframe(alert_data, use_container_width=True)

# Section 2: Assistant NLQ
elif section == "🗣️ Assistant NLQ":
    st.header("🗣️ Assistant en Langage Naturel")
    st.markdown("Posez vos questions en français sur vos données.")
    
    query = st.text_area(
        "💬 Tapez votre question :",
        placeholder="Ex: Quelle est la probabilité de renouvellement pour la police ABC123 ?",
        height=100
    )
    
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("🔍 Analyser la requête", type="primary"):
            if query:
                with st.spinner("Analyse en cours..."):
                    result = engines['nlp'].parse_query(query)
                    
                    st.success("Requête analysée avec succès !")
                    
                    col_res1, col_res2 = st.columns(2)
                    with col_res1:
                        st.metric("Intention détectée", result['intent'])
                    with col_res2:
                        st.metric("Confiance", f"{result['confidence']:.0%}")
                    
                    if result['entities']:
                        st.subheader("Entités identifiées :")
                        for key, value in result['entities'].items():
                            st.write(f"**{key}** : {value}")
                    
                    # Simulation de réponse
                    if result['intent'] == 'renewal_probability':
                        proba = np.random.uniform(0.6, 0.95)
                        st.info(f"**Réponse** : {engines['nlp'].generate_response(result['intent'], proba)}")
            else:
                st.warning("Veuillez entrer une requête.")
    
    with col_btn2:
        if st.button("📋 Exemples de requêtes"):
            examples = [
                "Probabilité de renouvellement pour le client 456",
                "Quels sont les contrats à risque élevé ?",
                "Prime moyenne des polices résiliées",
                "Distribution des risques par région"
            ]
            for ex in examples:
                st.code(ex)

# Section 3: Modèle Prédictif
elif section == "🤖 Modèle Prédictif":
    st.header("🤖 Entraînement du Modèle Prédictif")
    
    tab1, tab2, tab3 = st.tabs(["🎯 Entraînement", "📊 Performance", "🔮 Prédiction"])
    
    with tab1:
        st.subheader("Configuration du modèle")
        
        model_type = st.selectbox(
            "Algorithme",
            ["XGBoost", "Random Forest", "Régression Logistique"]
        )
        
        col_feat1, col_feat2 = st.columns(2)
        with col_feat1:
            target = st.selectbox(
                "Variable cible",
                ["renewed", "lapse", "premium_change"]
            )
        with col_feat2:
            test_size = st.slider("Taille du jeu de test", 0.1, 0.5, 0.2)
        
        if st.button("🚀 Entraîner le modèle", type="primary"):
            with st.spinner("Entraînement en cours..."):
                # Simulation d'entraînement
                import time
                time.sleep(2)
                
                # Métriques simulées
                col_perf1, col_perf2, col_perf3, col_perf4 = st.columns(4)
                with col_perf1:
                    st.metric("Accuracy", "0.89")
                with col_perf2:
                    st.metric("Precision", "0.87")
                with col_perf3:
                    st.metric("Recall", "0.85")
                with col_perf4:
                    st.metric("F1-Score", "0.86")
                
                st.success("✅ Modèle entraîné avec succès !")
    
    with tab2:
        st.subheader("Analyse des performances")
        
        # Matrice de confusion
        fig_conf = go.Figure(data=go.Heatmap(
            z=[[85, 15], [10, 90]],
            x=['Prédit Résilié', 'Prédit Renouvelé'],
            y=['Réel Résilié', 'Réel Renouvelé'],
            text=[['85', '15'], ['10', '90']],
            texttemplate="%{text}",
            textfont={"size": 16},
            colorscale='Blues'
        ))
        st.plotly_chart(fig_conf, use_container_width=True)
        
        # Courbe ROC
        st.subheader("Courbe ROC")
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
            y=[0, 0.3, 0.6, 0.8, 0.95, 1.0],
            mode='lines',
            name='Modèle'
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            line=dict(dash='dash'),
            name='Aléatoire'
        ))
        st.plotly_chart(fig_roc, use_container_width=True)
    
    with tab3:
        st.subheader("Faire une prédiction")
        
        col_pred1, col_pred2 = st.columns(2)
        with col_pred1:
            policy_id = st.text_input("ID de la police")
            premium = st.number_input("Prime (€)", value=1200)
        
        with col_pred2:
            tenure = st.number_input("Ancienneté (jours)", value=365)
            claims = st.number_input("Nombre de sinistres", value=1)
        
        if st.button("📊 Calculer la probabilité", type="primary"):
            # Simulation de prédiction
            proba = min(0.95, 0.7 + (premium/20000) - (claims*0.1))
            
            st.info(f"### Probabilité de renouvellement : {proba:.1%}")
            
            if proba > 0.7:
                st.success("✅ Faible risque - Stratégie standard recommandée")
            elif proba > 0.4:
                st.warning("⚠️ Risque modéré - Offre de fidélisation suggérée")
            else:
                st.error("🔴 Risque élevé - Intervention prioritaire requise")

# Section 4: Insights IA
elif section == "🔍 Insights IA":
    st.header("🔍 Insights Générés par IA")
    
    insight_type = st.selectbox(
        "Type d'analyse",
        ["📋 Rapport automatique", "📈 Visualisations avancées", "🎯 Recommandations"]
    )
    
    if insight_type == "📋 Rapport automatique":
        st.subheader("Rapport d'analyse généré automatiquement")
        
        # Générer un rapport exemple
        report_text = engines['insight'].generate_narrative_report(pd.DataFrame({
            'start_date': pd.date_range('2023-01-01', periods=100),
            'renewed': np.random.choice([0, 1], 100, p=[0.2, 0.8]),
            'current_premium': np.random.normal(1200, 300, 100)
        }))
        
        st.markdown(report_text)
        
        if st.button("🔄 Générer un nouveau rapport"):
            st.rerun()
    
    elif insight_type == "📈 Visualisations avancées":
        st.subheader("Visualisations interactives")
        
        # Créer des données exemple
        np.random.seed(42)
        data = pd.DataFrame({
            'Segment': ['Jeunes', 'Familles', 'Séniors', 'Entreprises'] * 25,
            'Renouvellement': np.random.beta(2, 1, 100),
            'Prime': np.random.normal(1500, 500, 100),
            'Région': np.random.choice(['Nord', 'Sud', 'Est', 'Ouest'], 100)
        })
        
        # Graphique 3D
        fig_3d = px.scatter_3d(
            data,
            x='Renouvellement',
            y='Prime',
            z=data.index,
            color='Segment',
            size='Prime',
            title="Analyse 3D des segments clients"
        )
        st.plotly_chart(fig_3d, use_container_width=True)
        
        # Heatmap de corrélations
        st.subheader("Matrice de corrélation")
        numeric_data = data.select_dtypes(include=[np.number])
        corr_matrix = numeric_data.corr()
        
        fig_heat = px.imshow(
            corr_matrix,
            text_auto=True,
            color_continuous_scale='RdBu',
            title="Corrélations entre variables"
        )
        st.plotly_chart(fig_heat, use_container_width=True)
    
    elif insight_type == "🎯 Recommandations":
        st.subheader("Recommandations stratégiques générées par IA")
        
        recommendations = [
            {
                "priorité": "Haute",
                "recommandation": "Cibler les clients avec prime > 150% de la moyenne",
                "impact": "Réduction de 15% des résiliations",
                "coût": "Faible"
            },
            {
                "priorité": "Moyenne",
                "recommandation": "Programme de fidélisation 60 jours avant expiration",
                "impact": "Amélioration de 8% du taux de renouvellement",
                "coût": "Moyen"
            },
            {
                "priorité": "Basse",
                "recommandation": "Personnalisation des offres par segment géographique",
                "impact": "Augmentation de 5% de la satisfaction client",
                "coût": "Élevé"
            }
        ]
        
        for rec in recommendations:
            with st.expander(f"{rec['priorité']} - {rec['recommandation']}"):
                col_rec1, col_rec2 = st.columns(2)
                with col_rec1:
                    st.metric("Impact estimé", rec['impact'])
                with col_rec2:
                    st.metric("Coût", rec['coût'])
                st.progress(75 if rec['priorité'] == 'Haute' else 50 if rec['priorité'] == 'Moyenne' else 25)

# Section 5: Data Quality
elif section == "📈 Data Quality":
    st.header("📈 Qualité des Données")
    
    # Simulation d'un rapport de qualité
    quality_metrics = {
        "Complétude": 94,
        "Exactitude": 88,
        "Cohérence": 92,
        "Actualité": 96,
        "Unicité": 98
    }
    
    # Scores de qualité
    col_q1, col_q2, col_q3, col_q4, col_q5 = st.columns(5)
    metrics_cols = [col_q1, col_q2, col_q3, col_q4, col_q5]
    
    for idx, (metric, score) in enumerate(quality_metrics.items()):
        with metrics_cols[idx]:
            st.metric(metric, f"{score}%")
    
    # Graphique radar
    fig_radar = go.Figure(data=go.Scatterpolar(
        r=list(quality_metrics.values()),
        theta=list(quality_metrics.keys()),
        fill='toself',
        line_color='blue'
    ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=False,
        title="Score de qualité des données"
    )
    
    st.plotly_chart(fig_radar, use_container_width=True)
    
    # Problèmes détectés
    st.subheader("🔎 Problèmes identifiés")
    
    issues = pd.DataFrame({
        'Type': ['Valeurs manquantes', 'Incohérences', 'Doublons', 'Format incorrect'],
        'Count': [124, 67, 42, 89],
        'Criticité': ['Moyenne', 'Basse', 'Forte', 'Moyenne'],
        'Statut': ['Corrigé', 'En cours', 'À faire', 'Corrigé']
    })
    
    st.dataframe(issues, use_container_width=True)
    
    # Log de nettoyage
    st.subheader("📝 Journal des transformations")
    
    log_entries = [
        "2024-01-15 10:30: Suppression de 42 doublons",
        "2024-01-15 11:15: Imputation des valeurs manquantes (médiane)",
        "2024-01-15 12:00: Conversion des dates au format standard",
        "2024-01-15 14:30: Création de 5 nouvelles variables dérivées"
    ]
    
    for entry in log_entries:
        st.code(entry)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>🤖 <b>RenewAI Dashboard</b> - Hackathon Insurance Analytics 2024</p>
        <p>Moteurs: NLQ • Prédictif • Insights • Data Quality</p>
    </div>
    """,
    unsafe_allow_html=True
)