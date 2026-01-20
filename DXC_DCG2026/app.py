# ============================================================
# app.py — LIK Insurance Analyst
# Orchestrateur central de tous les moteurs
# ============================================================


# Importation des libraries necessaires
# IMPORTS DES BIBLIOTHÈQUES STANDARD ET TIERS

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
import json
import io
import tempfile
import zipfile
import traceback
import warnings
from datetime import datetime
from dotenv import load_dotenv
# Import additionnel pour les métriques d'évaluation des modèles et les sous-figures Plotly
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from plotly.subplots import make_subplots

#Ignorer les warnings pour une sortie console plus propre
warnings.filterwarnings('ignore')

# Import des modules personnalisés de l'application
import streamlit as st
from modules.report_engine import ReportEngine
from modules.custom_llm_client import OpenAIAnalyzer

# ========== INITIALISATION SESSION STATE ==========

#INITIALISATION DES VARIABLES D'ÉTAT DE LA SESSION#

# Ces variables persistent lors de la navigation dans l'application Streamlit.#
if 'openai_client' not in st.session_state:
    st.session_state.openai_client = None

if 'report_engine' not in st.session_state:
    st.session_state.report_engine = None



# ------------------------------------------------------------
# CHARGEMENT DES VARIABLES D'ENVIRONNEMENT
#Tente de charger la clé API OpenAI depuis un fichier .env
#ou les variables d'environnement système.
# ------------------------------------------------------------
OPENAI_API_KEY = ""

# Essayer de charger depuis .env avec plusieurs encodages
encodings_to_try = ['utf-8', 'latin-1', 'utf-16', 'cp1252']

for encoding in encodings_to_try:
    try:
        # Réinitialiser dotenv
        from dotenv import dotenv_values
        env_vars = dotenv_values(".env", encoding=encoding)
        OPENAI_API_KEY = env_vars.get("OPENAI_API_KEY", "")
        if OPENAI_API_KEY:
            print(f" Fichier .env chargé avec encodage: {encoding}")
            break
    except:
        continue

# Si .env échoue, essayer depuis les variables système
if not OPENAI_API_KEY:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Si toujours rien, laisser vide
if not OPENAI_API_KEY:
    print(" Aucune clé API trouvée dans .env ou variables système")
# ------------------------------------------------------------
# DÉFINITION DES CHEMINS D'ACCÈS
# Configure les chemins vers le répertoire de base et les assets.
# ------------------------------------------------------------
BASE_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(BASE_DIR, "modules"))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

# Vérifier si report_engine est disponible
try:
    from report_engine import ReportEngine

    REPORT_ENGINE_AVAILABLE = True
except ImportError:
    REPORT_ENGINE_AVAILABLE = False
    ReportEngine = None


# ------------------------------------------------------------
# CONFIGURATION DE LA PAGE STREAMLIT
# Définit le titre, l'icône, la mise en page et l'état initial de la barre latérale.
# ------------------------------------------------------------
st.set_page_config(
    page_title="LIK Insurance Analyst",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)



# ------------------------------------------------------------
# CHARGEMENT DU FICHIER CSS
# Applique des styles personnalisés pour l'interface utilisateur.
# ------------------------------------------------------------
def load_css():
    css_path = os.path.join(ASSETS_DIR, "style.css")
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        # Style CSS par défaut
        st.markdown("""
        <style>
        .main-header {
            color: #1E3A8A;
            text-align: center;
            font-size: 2.5rem;
            margin-bottom: 1rem;
        }
        .stButton > button {
            width: 100%;
        }
        </style>
        """, unsafe_allow_html=True)


load_css()

# ------------------------------------------------------------
# ÉTAT DE LA SESSION (CENTRAL)
# Initialise toutes les variables d'état de session avec des valeurs par défaut.
# Ces variables stockent les données et l'état de l'application entre les interactions.
# ------------------------------------------------------------
session_defaults = {
    "df": None,
    "dataframe": None,
    "metadata": None,
    "data_ready": False,
    "data_loaded": False,
    "metadata_ready": False,
    "openai_client": None,
    "nlq_engine": None,
    "predictive_engine": None,
    "insight_engine": None,
    "scored_clients": None,
    "client_table": None,
    "raw_data": None,
    "conversation_history": [],
    "business_context": None,
    "column_documentation": None,
    "column_explainer": None,
    "uploaded_file_name": None,
    "df_final": None,
    "report_engine": None,
    "using_mateur": False,
    "generated_report_md": None,
    "generated_report_pdf": None,
    "generated_report_word": None,
    "generated_report_html": None,
    "data_processor": None,
    "df_processed": None,
    "df_enriched": None,
    "df_segmented": None,
    "pipeline_results": None,
    "data_processed": False,
    "predictive_engine": None,
    "df_prepared": None,
    "time_series_data": None,
    'df_final' : None
}
# Initialisation : pour chaque clé,
# si elle n'existe pas dans session_state, l'ajouter avec sa valeur par défaut.

for key, value in session_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ------------------------------------------------------------
# EN-TÊTE DE L'APPLICATION
# Affiche le titre principal et la description.
# ------------------------------------------------------------
st.markdown('<h1 class="main-header">LIK Insurance Analyst</h1>', unsafe_allow_html=True)
st.markdown('<p style="font-size: 16px;">Transformer vos données en décisions viables</p>', unsafe_allow_html=True)
# ============================================================
# BARRE LATÉRALE (SIDEBAR)
# Contient la navigation, la configuration API, l'état de l'application.
# ============================================================
with st.sidebar:
    # Logo centré
    logo_path = os.path.join(ASSETS_DIR, "logo0.png")
    if os.path.exists(logo_path):
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(logo_path, width=300)

        st.markdown(
            """
            <div style="text-align: center;">
                <p style="margin-top: 5px; margin-bottom: 20px; font-size: 14px;">
                    <b>Livrer l'excellence</b>
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown("###  LIK Insurance")

    # Section API masquée - gardée uniquement pour la logique backend
    if st.session_state.get('show_api_config', False):  # Condition pour afficher/masquer
        st.markdown("###  Configuration API")

        api_key_input = st.text_input(
            "Clé OpenAI API",
            type="password",
            value=OPENAI_API_KEY,
            help="Obtenez votre clé sur platform.openai.com",
            placeholder="sk-...",
            key="api_key_input"
        )
    else:
        # Utiliser la clé API sans l'afficher dans l'interface
        api_key_input = OPENAI_API_KEY  # Utiliser directement la valeur du .env

    # Initialiser les clients OpenAI
    # Dans la sidebar de app.py, modifiez cette partie :
    if api_key_input:
        try:
            # from llm_client import OpenAIAnalyzer  # ← COMMENTEZ CETTE LIGNE
            from custom_llm_client import OpenAIAnalyzer  # ← UTILISEZ CE NOUVEAU NOM

            # Vérifier que la clé n'est pas vide
            if api_key_input.strip():
                st.session_state.openai_client = OpenAIAnalyzer(api_key=api_key_input)
                st.success(" Connexion valide")
            else:
                st.warning(" Veuillez entrer une clé API valide")
                st.session_state.openai_client = None

        except ImportError as e:
            st.error(f" Module custom_llm_client non disponible: {e}")
        except ValueError as e:
            st.error(f" Clé API invalide: {e}")
        except Exception as e:
            st.error(f" Erreur d'initialisation: {e}")
    else:
        if 'openai_client' in st.session_state:
            st.session_state.openai_client = None
            st.session_state.nlq_engine = None
            st.session_state.report_engine = None

# Navigation
    from streamlit import container

    # Barre horizontale avec espace réduit
    st.markdown("<hr style='margin: 2px 0 10px 0;'>", unsafe_allow_html=True)

    # Navigation dans un container
    with container():
        st.markdown("<h3 style='text-align: center; margin: 0 0 13px 0; padding: 0;'>Navigation</h3>",
                    unsafe_allow_html=True)
    page = st.radio(
        "",
        [
            " 📤 Chargement des données",
            " 🏷️ Métadonnées",
            " 🔄 Traitement des données",
            " 💬 Assistant IA",
            " 👁️ Visualisation des données",
            " 🧮 Modèles Prédictifs",
            " 📄 Rapport Intelligent",
            "🏢 À Propos"

        ]
    )

    st.markdown("---")

    def ensure_metadata_available():
        """
        Vérifie que les métadonnées sont disponibles, sinon les génère
        """
        if not st.session_state.metadata_ready and st.session_state.data_loaded:
            try:
                from modules.metadata_extractor import MetadataExtractor
                from modules.business_context import BusinessContextProvider

                df = st.session_state.dataframe

                with st.spinner("🔍 Génération des métadonnées en cours..."):
                    metadata_extractor = MetadataExtractor(df)
                    metadata = metadata_extractor.extract_safe_metadata()

                    business_context = BusinessContextProvider.get_context(
                        BusinessContextProvider.infer_domain_from_columns(df.columns)
                    )

                    st.session_state.metadata = metadata
                    st.session_state.business_context = business_context
                    st.session_state.metadata_ready = True

                st.success("✅ Métadonnées générées avec succès")
                return True

            except ImportError:
                st.warning("⚠️ Impossible de générer les métadonnées automatiquement")
                return False
        return st.session_state.metadata_ready


    # État de l'application
    st.markdown("###  État")
    if st.session_state.data_loaded:
        df = st.session_state.dataframe
        st.success(" Données chargées")
        st.caption(f"• {len(df):,} lignes")
        st.caption(f"• {len(df.columns)} colonnes")
    else:
        st.warning(" Aucune donnée")

    if st.session_state.metadata is not None:
        st.success(" Métadonnées prêtes")

    if st.session_state.scored_clients is not None:
        st.success(" Analyse risque complète")

# ============================================================
# 1️⃣ PAGE : CHARGEMENT DES DONNÉES
# Permet à l'utilisateur de téléverser un fichier de données.
# ============================================================
if page == " 📤 Chargement des données":
    st.header(" Chargement des données")
    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Téléversez votre fichier de données",
            type=["csv", "xlsx", "xls", "txt", "dta"],
            help="Formats supportés: CSV, Excel, Texte, Stata"
        )

        if uploaded_file is not None:
            try:
                # Sauvegarder le nom du fichier
                st.session_state.uploaded_file_name = uploaded_file.name

                # Détection du type de fichier
                file_name = uploaded_file.name.lower()

                if file_name.endswith('.csv'):
                    # Essayer différents encodages
                    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
                    df = None
                    for encoding in encodings:
                        try:
                            uploaded_file.seek(0)  # Réinitialiser le pointeur
                            df = pd.read_csv(uploaded_file, encoding=encoding)
                            break
                        except UnicodeDecodeError:
                            continue
                    if df is None:
                        # Dernier essai avec erreurs ignorées
                        uploaded_file.seek(0)
                        df = pd.read_csv(uploaded_file, encoding='utf-8', errors='replace')
                elif file_name.endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(uploaded_file)
                elif file_name.endswith('.txt'):
                    df = pd.read_csv(uploaded_file, sep='\t', encoding='utf-8')
                elif file_name.endswith('.dta'):
                    df = pd.read_stata(uploaded_file)
                else:
                    st.error("Format de fichier non supporté")
                    df = None

                if df is not None:
                    # Préparation des données
                    try:
                        from data_prep_engine import DataPrepEngine

                        prep = DataPrepEngine()
                        df = prep.clean_data(df)
                        df = prep.engineer_features(df)
                        st.session_state.df_final = df
                    except ImportError:
                        # Si le module n'est pas disponible, utiliser les données brutes
                        st.info("Module data_prep_engine non disponible - données brutes utilisées")
                        st.session_state.df_final = df

                    # Affichage des informations de base
                    st.success(f" Fichier chargé: {uploaded_file.name}")

                    with st.expander(" Aperçu des données", expanded=True):
                        st.dataframe(df.head(5), use_container_width=True) # On charge les 5 1eres observations

                    # Statistiques rapides
                    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                    with col_stat1:
                        st.metric("Lignes", f"{len(df):,}")
                    with col_stat2:
                        st.metric("Colonnes", len(df.columns))
                    with col_stat3:
                        st.metric("Valeurs manquantes", f"{df.isna().sum().sum():,}")
                    with col_stat4:
                        st.metric("Doublons", f"{df.duplicated().sum():,}")

                    # Sauvegarde dans la session
                    st.session_state.dataframe = df
                    st.session_state.df = df
                    st.session_state.data_loaded = True
                    st.session_state.data_ready = True

                    # Réinitialiser les analyses existantes
                    st.session_state.metadata = None
                    st.session_state.business_context = None
                    st.session_state.scored_clients = None
                    st.session_state.client_table = None

                    st.success(" Données prêtes pour l'analyse!")

            except Exception as e:
                st.error(f" Erreur lors du chargement: {str(e)}")
                st.code(traceback.format_exc())

    with col2:
        st.markdown("**100% de Sécurité**")
        st.info("""
        Cette application vous permet de capitaliser sur vos objectifs pour le profilage et la gestion des clients en risque.

        **Vos données restent:**
        - En local sur votre ordinateur
        - Ne sont jamais partagées
        - Entièrement sous votre contrôle
        """)
        # Je pense que c'est bien

# ============================================================
#  PAGE : MÉTADONNÉES
# Extrait et affiche les métadonnées des données chargées (structure, types, contexte métier).
# ============================================================
elif page == " 🏷️ Métadonnées":
    st.header(" Extraction des Métadonnées")

    if not st.session_state.data_loaded:
        st.warning(" Veuillez d'abord charger des données")
        st.stop()

    df = st.session_state.dataframe

    try:
        from metadata_extractor import MetadataExtractor
        from business_context import BusinessContextProvider

        with st.spinner("Extraction des métadonnées sécurisées..."):
            metadata_extractor = MetadataExtractor(df)
            metadata = metadata_extractor.extract_safe_metadata()
            schema_json = metadata_extractor.generate_schema_json()

            business_context = BusinessContextProvider.get_context(
                BusinessContextProvider.infer_domain_from_columns(df.columns)
            )

            st.session_state.metadata = metadata
            st.session_state.business_context = business_context

        st.success(" Métadonnées extraites avec succès!")

        tab1, tab2, tab3 = st.tabs([" Vue d'ensemble", " Structure", " Contexte Métier"])

        with tab1:
            st.subheader("Informations Générales")
            general_info = metadata.get('general_info', {})

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Lignes", general_info.get('nombre_lignes', 0))
            with col2:
                st.metric("Colonnes", general_info.get('nombre_colonnes', 0))
            with col3:
                st.metric("Mémoire (Mo)", f"{general_info.get('taille_memoire_mo', 0):.1f}")
            with col4:
                st.metric("Qualité", f"{metadata.get('quality_indicators', {}).get('completude_pct', 0):.1f}%")

            st.subheader("Types de Données")
            dtype_summary = metadata.get('data_types_summary', {})
            if dtype_summary:
                dtype_df = pd.DataFrame({
                    "Type": list(dtype_summary.keys()),
                    "Nombre": list(dtype_summary.values())
                })
                st.dataframe(dtype_df, use_container_width=True)

        with tab2:
            st.subheader("Structure des Colonnes")
            columns_info = metadata.get('structure_columns', [])
            columns_df = pd.DataFrame(columns_info[:10])
            st.dataframe(columns_df, use_container_width=True)

            st.subheader("Profils Statistiques")
            profiles = metadata.get('statistical_profiles', {})

            if profiles.get('variables_numeriques'):
                st.markdown("**Variables Numériques:**")
                for var in profiles['variables_numeriques'][:3]:
                    st.markdown(f"- {var['nom']}: [{var['plage']['min']:.2f}, {var['plage']['max']:.2f}]")

            if profiles.get('variables_categorielles'):
                st.markdown("**Variables Catégorielles:**")
                for var in profiles['variables_categorielles'][:3]:
                    st.markdown(f"- {var['nom']}: {var['categories_count']} catégories")

        with tab3:
            st.subheader("Contexte Métier Inféré")
            context = business_context

            st.markdown(f"**Domaine:** {context.get('domaine', 'Non déterminé')}")
            st.markdown(f"**Description:** {context.get('description', '')}")

            st.markdown("**Concepts Clés:**")
            concepts = context.get('concepts_cles', [])
            for concept in concepts[:5]:
                st.markdown(f"- {concept}")

            st.markdown("**Analyses Courantes:**")
            analyses = context.get('analyses_courantes', [])
            for analyse in analyses[:3]:
                st.markdown(f"- {analyse}")

        st.markdown("---")
        st.subheader(" Export des Métadonnées")
        col_exp1, col_exp2 = st.columns(2)

        with col_exp1:
            json_str = json.dumps(schema_json, indent=2, ensure_ascii=False)
            st.download_button(
                label=" Télécharger JSON",
                data=json_str,
                file_name="metadata.json",
                mime="application/json"
            )

        with col_exp2:
            context_str = json.dumps(context, indent=2, ensure_ascii=False)
            st.download_button(
                label=" Contexte Métier",
                data=context_str,
                file_name="business_context.json",
                mime="application/json"
            )

    except ImportError as e:
        st.error(f" Erreur d'importation: {e}")
        st.info("Vérifiez que les modules sécurisés sont installés dans le dossier 'modules/'")

# ============================================================
# PAGE : TRAITEMENT DES DONNÉES
# Offre des outils avancés pour le nettoyage, l'analyse scientifique et la préparation des données.
# ============================================================

elif page == " 🔄 Traitement des données":
    st.header("🔄 Traitement De Données")

    if not st.session_state.data_loaded:
        st.warning("⚠️ Veuillez d'abord charger des données")
        st.stop()

    # Utiliser les données finales si disponibles, sinon les données brutes
    if st.session_state.df_final is not None:
        df = st.session_state.df_final.copy()
    else:
        df = st.session_state.dataframe.copy()

    # Initialisation du moteur de traitement scientifique
    try:
        from modules.data_processing_engine import DataProcessingEngine

        # CORRECTION : Initialiser le moteur avec une vérification correcte
        if 'data_processor' not in st.session_state or st.session_state.data_processor is None:
            st.session_state.data_processor = DataProcessingEngine()
            st.success("✅ Vous pouvez commencer vos traitements")

        processor = st.session_state.data_processor

    except ImportError as e:
        st.error(f"❌ Module data_processing_engine non disponible: {e}")
        st.info("""
        **Assurez-vous que:**
        1. Le fichier `data_processing_engine.py` est dans le dossier `modules/`
        2. La classe `DataProcessingEngine` est bien définie
        """)
        st.stop()

    # Le reste du code reste inchangé...

    # Onglets de traitement
    tab1, tab2, tab3, tab4 = st.tabs([
        "🧬 Analyse Scientifique",
        "🔧 Prétraitement",
        "🎯 Détection Cibles",
        "📊 Statistiques"
    ])

    # ============================================================
    # TAB 1: ANALYSE SCIENTIFIQUE
    # Détection automatique des types de variables et analyse statistique.
    # ============================================================
    with tab1:
        st.subheader("🧬 Analyse Scientifique des Données")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("""
            **🔍 Détection scientifique des types:**
            - Identification automatique des types de variables
            - Tests statistiques rigoureux
            - Analyse de distribution approfondie
            - Détection d'anomalies scientifiques
            """)

            if st.button("🔬 Exécuter l'analyse scientifique", type="primary", use_container_width=True):
                with st.spinner("🧬 Analyse scientifique en cours..."):
                    try:
                        # Détection des types de colonnes
                        column_types = processor.detect_column_types(df)

                        # Afficher le résumé statistique
                        statistical_summary = processor.get_statistical_summary()

                        st.success("✅ Analyse scientifique terminée!")

                        # Afficher les types de variables
                        st.markdown("#### 📊 Types de variables détectés")

                        type_counts = statistical_summary.get("variable_types", {})
                        if type_counts:
                            type_df = pd.DataFrame({
                                "Type de variable": list(type_counts.keys()),
                                "Nombre": list(type_counts.values())
                            })
                            st.dataframe(type_df, use_container_width=True)

                        # Métriques de qualité
                        st.markdown("#### 🎯 Métriques de qualité")

                        quality_metrics = statistical_summary.get("data_quality", {})
                        col_q1, col_q2, col_q3, col_q4 = st.columns(4)

                        with col_q1:
                            st.metric("Variables totales", quality_metrics.get("total_variables", 0))

                        with col_q2:
                            complete = quality_metrics.get("complete_variables", 0)
                            total = quality_metrics.get("total_variables", 1)
                            percentage = (complete / total * 100) if total > 0 else 0
                            st.metric("Variables complètes", f"{complete} ({percentage:.1f}%)")

                        with col_q3:
                            st.metric("Variables normales", quality_metrics.get("normal_variables", 0))

                        with col_q4:
                            st.metric("Haute qualité", quality_metrics.get("high_quality_variables", 0))

                        # Détails par colonne
                        with st.expander("📄 Détails par colonne", expanded=False):
                            for col, info in list(column_types.items())[:10]:  # Limiter à 10 colonnes
                                with st.expander(f"Colonne: {col}", expanded=False):
                                    col1_info, col2_info = st.columns(2)

                                    with col1_info:
                                        st.markdown(f"**Type:** {info.get('type', 'N/A')}")
                                        st.markdown(f"**Type original:** {info.get('original_dtype', 'N/A')}")
                                        st.markdown(f"**Valeurs uniques:** {info.get('unique_values', 0)}")
                                        st.markdown(f"**Valeurs manquantes:** {info.get('missing_percentage', 0):.1f}%")

                                    with col2_info:
                                        if 'distribution' in info:
                                            dist = info['distribution']
                                            if 'mean' in dist:
                                                st.markdown(f"**Moyenne:** {dist['mean']:.2f}")
                                                st.markdown(f"**Écart-type:** {dist['std']:.2f}")
                                            if 'skewness' in dist:
                                                st.markdown(f"**Asymétrie:** {dist['skewness']:.2f}")

                        if len(column_types) > 10:
                            st.info(f"... et {len(column_types) - 10} autres colonnes analysées")

                    except Exception as e:
                        st.error(f"❌ Erreur lors de l'analyse: {str(e)}")
                        st.code(traceback.format_exc())

        with col2:
            st.info("""
            **🎯 Méthodologie :**

            1. **Tests statistiques** (Shapiro-Wilk, Anderson-Darling)
            2. **Analyse de distribution** complète
            3. **Détection d'anomalies** avec IQR et Z-scores
            4. **Classification automatique** des types de variables
            5. **Validation rigoureuse** des hypothèses
            """)

    # ============================================================
    # TAB 2: PRÉTRAITEMENT
    # Options pour nettoyer et préparer les données pour l'analyse.
    # ============================================================
    with tab2:
        st.subheader("🛠️ Options de prétraitement")

        st.markdown("""
        - Traitement intelligent des valeurs manquantes
        - Détection et correction des anomalies
        - Conservation rigoureuse des types de données
        - Normalisation adaptative selon la distribution
        """)

        # Options de prétraitement
        col_opt1, col_opt2 = st.columns(2)

        with col_opt1:
            strategy = st.selectbox(
                "Stratégie de prétraitement:",
                ["conservative", "balanced", "aggressive"],
                help="Conservative: privilégie la conservation des données\n"
                     "Balanced: équilibre entre conservation et traitement\n"
                     "Aggressive: optimisation maximale pour le machine learning"
            )

            target_column = st.selectbox(
                "Colonne cible (optionnelle):",
                ["Aucune"] + list(df.columns),
                index=0,
                help="Sélectionnez une colonne cible pour un traitement adapté"
            )

        with col_opt2:
            preserve_types = st.checkbox("Conserver les types originaux", value=True)
            handle_missing = st.checkbox("Traiter les valeurs manquantes", value=True)
            handle_outliers = st.checkbox("Traiter les anomalies", value=True)

        if target_column == "Aucune":
            target_column = None

        if st.button("⚡ Exécuter le prétraitement", type="primary", use_container_width=True):
            with st.spinner("🔧 Prétraitement en cours..."):
                try:
                    # Exécuter le prétraitement scientifique
                    df_processed = processor.scientific_preprocess(
                        df,
                        target_column=target_column,
                        strategy=strategy
                    )

                    # Sauvegarder les résultats
                    st.session_state.df_processed = df_processed
                    st.session_state.data_processor = processor

                    st.success(f"✅ Prétraitement terminé: {len(df)} → {len(df_processed)} lignes")

                    # Afficher le rapport scientifique
                    scientific_report = processor.get_scientific_report()

                    # Métriques de prétraitement
                    st.markdown("#### 📊 Métriques de prétraitement")

                    quality_metrics = scientific_report.get("quality_metrics", {})
                    col_met1, col_met2, col_met3 = st.columns(3)

                    with col_met1:
                        completeness = quality_metrics.get("completeness", 0) * 100
                        st.metric("Complétude", f"{completeness:.1f}%")

                    with col_met2:
                        type_rate = quality_metrics.get("type_conservation_rate", 0) * 100
                        st.metric("Types conservés", f"{type_rate:.1f}%")

                    with col_met3:
                        if "numeric_variability" in quality_metrics:
                            variability = quality_metrics["numeric_variability"]
                            st.metric("Variabilité", f"{variability:.3f}")

                    # Étapes appliquées
                    steps_applied = scientific_report.get("steps_applied", [])
                    if steps_applied:
                        st.markdown("#### 📝 Étapes appliquées")

                        for i, step in enumerate(steps_applied[:10], 1):  # Limiter à 10 étapes
                            st.markdown(f"{i}. {step}")

                        if len(steps_applied) > 10:
                            st.info(f"... et {len(steps_applied) - 10} autres étapes")

                    # Aperçu des données traitées
                    with st.expander("👁️ Aperçu des données traitées", expanded=False):
                        st.dataframe(df_processed.head(10), use_container_width=True)

                        # Comparaison avant/après
                        col_comp1, col_comp2 = st.columns(2)

                        with col_comp1:
                            st.markdown("**Avant traitement:**")
                            st.metric("Lignes", len(df))
                            st.metric("Valeurs manquantes", df.isna().sum().sum())

                        with col_comp2:
                            st.markdown("**Après traitement:**")
                            st.metric("Lignes", len(df_processed))
                            st.metric("Valeurs manquantes", df_processed.isna().sum().sum())

                    # Téléchargement des données traitées
                    st.markdown("---")
                    st.markdown("#### 📤 Export des données traitées")

                    csv = df_processed.to_csv(index=False, encoding='utf-8-sig')
                    st.download_button(
                        label="💾 Télécharger CSV",
                        data=csv,
                        file_name="donnees_traitees_scientifiques.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

                except Exception as e:
                    st.error(f"❌ Erreur lors du prétraitement: {str(e)}")
                    st.code(traceback.format_exc())

    # ============================================================
    # TAB 3: DÉTECTION DE CIBLES
    # Identifie les variables potentielles à utiliser comme cible pour la modélisation
    # ============================================================
    with tab3:
        st.subheader("🎯 Détection de Variables Cibles")

        st.markdown("""
        **🔎 Méthodologie de détection:**

        1. **Analyse sémantique** des noms de colonnes
        2. **Critères statistiques** (cardinalité, distribution)
        3. **Tests d'équilibre** pour les variables catégorielles
        4. **Élimination** des variables trop déséquilibrées
        """)

        if st.button("🎯 Détecter les variables cibles", type="primary", use_container_width=True):
            with st.spinner("🔎 Détection en cours..."):
                try:
                    # Détection des variables cibles potentielles
                    potential_targets = processor.detect_potential_targets(df)

                    if potential_targets:
                        st.success(f"✅ {len(potential_targets)} variables cibles potentielles détectées")

                        # Afficher les variables cibles
                        st.markdown("#### 📄 Variables cibles potentielles")

                        targets_df = pd.DataFrame({
                            "Variable": potential_targets,
                            "Type": [str(df[col].dtype) for col in potential_targets],
                            "Valeurs uniques": [df[col].nunique() for col in potential_targets],
                            "Valeurs manquantes": [df[col].isna().sum() for col in potential_targets]
                        })

                        st.dataframe(targets_df, use_container_width=True)

                        # Détails par variable cible
                        st.markdown("#### 📊 Analyse détaillée des variables cibles")

                        for target_col in potential_targets[:5]:  # Limiter à 5 variables
                            with st.expander(f"Analyse de: {target_col}", expanded=False):
                                col_target1, col_target2 = st.columns(2)

                                with col_target1:
                                    # Statistiques de base
                                    st.markdown("**Statistiques:**")
                                    non_null = df[target_col].dropna()

                                    if pd.api.types.is_numeric_dtype(df[target_col]):
                                        st.markdown(f"• Moyenne: {non_null.mean():.2f}")
                                        st.markdown(f"• Écart-type: {non_null.std():.2f}")
                                        st.markdown(f"• Min: {non_null.min():.2f}")
                                        st.markdown(f"• Max: {non_null.max():.2f}")
                                    else:
                                        value_counts = df[target_col].value_counts(normalize=True)
                                        for val, prop in list(value_counts.items())[:5]:
                                            st.markdown(f"• {val}: {prop * 100:.1f}%")

                                with col_target2:
                                    # Visualisation
                                    st.markdown("**Distribution:**")

                                    if pd.api.types.is_numeric_dtype(df[target_col]):
                                        try:
                                            import plotly.express as px

                                            fig = px.histogram(df, x=target_col, title=f"Distribution de {target_col}")
                                            st.plotly_chart(fig, use_container_width=True, height=300)
                                        except:
                                            st.info("📊 Visualisation non disponible")
                                    elif df[target_col].nunique() <= 10:
                                        try:
                                            import plotly.express as px

                                            value_counts = df[target_col].value_counts()
                                            fig = px.pie(values=value_counts.values,
                                                         names=value_counts.index,
                                                         title=f"Distribution de {target_col}")
                                            st.plotly_chart(fig, use_container_width=True, height=300)
                                        except:
                                            st.info("📊 Visualisation non disponible")

                        if len(potential_targets) > 5:
                            st.info(f"... et {len(potential_targets) - 5} autres variables cibles")

                        # Recommandations
                        st.markdown("#### 💡 Recommandations")

                        if len(potential_targets) >= 3:
                            st.success("✅ Plusieurs variables cibles potentielles détectées.")
                            st.info("Pour la modélisation, choisissez une variable avec:")
                            st.markdown("1. **Distribution équilibrée** (pas trop déséquilibrée)")
                            st.markdown("2. **Peu de valeurs manquantes**")
                            st.markdown("3. **Sens métier clair**")
                        else:
                            st.warning("⚠️ Peu de variables cibles détectées.")
                            st.info("Considérez:")
                            st.markdown("1. **Créer une variable cible dérivée**")
                            st.markdown("2. **Utiliser une colonne numérique comme cible**")
                            st.markdown("3. **Recoder une variable existante**")

                    else:
                        st.warning("⚠️ Aucune variable cible potentielle détectée")
                        st.info("""
                        **Suggestions:**
                        1. Vérifiez si vos données contiennent des colonnes comme:
                           - `risque`, `churn`, `resilie`, `renouvelle`
                           - Variables binaires (oui/non, 0/1)
                           - Variables catégorielles à faible cardinalité

                        2. Vous pouvez créer une variable cible manuellement
                        3. Utilisez une variable numérique comme cible de régression
                        """)

                except Exception as e:
                    st.error(f"❌ Erreur lors de la détection: {str(e)}")
                    st.code(traceback.format_exc())

    # ============================================================
    # TAB 4: STATISTIQUES
    # Affiche des statistiques détaillées sur les colonnes et permet leur exploration
    # ============================================================
    with tab4:
        st.subheader("📊 Statistiques Complètes")

        # Options d'affichage
        col_stats1, col_stats2 = st.columns(2)

        with col_stats1:
            show_detailed = st.checkbox("Afficher les statistiques détaillées", value=False)
            include_missing = st.checkbox("Inclure l'analyse des valeurs manquantes", value=True)

        with col_stats2:
            limit_cols = st.slider("Nombre maximum de colonnes à afficher",
                                   min_value=5, max_value=50, value=20)
            sort_by = st.selectbox("Trier par:",
                                   ["Nom", "Type", "Valeurs manquantes", "Valeurs uniques"])

        if st.button("📈 Générer les statistiques", type="primary", use_container_width=True):
            with st.spinner("📊 Calcul des statistiques..."):
                try:
                    # Obtenir les statistiques des colonnes
                    column_stats = processor.get_column_statistics(df)

                    # Convertir en DataFrame pour l'affichage
                    stats_list = []
                    for col, stats in column_stats.items():
                        stats_list.append({
                            "Colonne": col,
                            "Type": stats["dtype"],
                            "Non nul": stats["non_null_count"],
                            "Nul": stats["null_count"],
                            "% Nul": stats["null_percentage"],
                            "Uniques": stats["unique_count"],
                            "% Unique": (stats["unique_count"] / len(df)) * 100 if len(df) > 0 else 0
                        })

                    stats_df = pd.DataFrame(stats_list)

                    # Trier selon le critère sélectionné
                    sort_map = {
                        "Nom": "Colonne",
                        "Type": "Type",
                        "Valeurs manquantes": "% Nul",
                        "Valeurs uniques": "Uniques"
                    }

                    if sort_by in sort_map:
                        stats_df = stats_df.sort_values(sort_map[sort_by],
                                                        ascending=(sort_by == "Nom"))

                    # Limiter le nombre de colonnes affichées
                    stats_df = stats_df.head(limit_cols)

                    st.success(f"✅ Statistiques générées pour {len(column_stats)} colonnes")

                    # Tableau récapitulatif
                    st.markdown("#### 📄 Récapitulatif des colonnes")
                    st.dataframe(stats_df, use_container_width=True)

                    # Statistiques globales
                    st.markdown("#### 🎯 Statistiques globales")

                    col_glob1, col_glob2, col_glob3, col_glob4 = st.columns(4)

                    with col_glob1:
                        total_missing = stats_df["Nul"].sum()
                        total_values = len(df) * len(df.columns)
                        missing_pct = (total_missing / total_values * 100) if total_values > 0 else 0
                        st.metric("Valeurs manquantes", f"{missing_pct:.1f}%")

                    with col_glob2:
                        avg_unique = stats_df["% Unique"].mean()
                        st.metric("Unicité moyenne", f"{avg_unique:.1f}%")

                    with col_glob3:
                        numeric_cols = df.select_dtypes(include=[np.number]).columns
                        st.metric("Colonnes numériques", len(numeric_cols))

                    with col_glob4:
                        categorical_cols = df.select_dtypes(include=['object']).columns
                        st.metric("Colonnes catégorielles", len(categorical_cols))

                    # Analyse détaillée pour les colonnes sélectionnées
                    if show_detailed and len(stats_df) > 0:
                        st.markdown("#### 🔍 Analyse détaillée par colonne")

                        selected_cols = st.multiselect(
                            "Sélectionnez les colonnes à analyser en détail:",
                            stats_df["Colonne"].tolist(),
                            default=stats_df["Colonne"].head(3).tolist()
                        )

                        for col in selected_cols:
                            if col in column_stats:
                                col_info = column_stats[col]

                                with st.expander(f"Analyse détaillée: {col}", expanded=False):
                                    # Informations de base
                                    col_det1, col_det2 = st.columns(2)

                                    with col_det1:
                                        st.markdown("**Informations de base:**")
                                        st.markdown(f"- Type: {col_info['dtype']}")
                                        st.markdown(f"- Non nul: {col_info['non_null_count']}")
                                        st.markdown(
                                            f"- Nul: {col_info['null_count']} ({col_info['null_percentage']:.1f}%)")
                                        st.markdown(f"- Uniques: {col_info['unique_count']}")

                                    with col_det2:
                                        if 'mean' in col_info:
                                            st.markdown("**Statistiques numériques:**")
                                            st.markdown(f"- Moyenne: {col_info['mean']:.2f}")
                                            st.markdown(f"- Écart-type: {col_info['std']:.2f}")
                                            st.markdown(f"- Min: {col_info['min']:.2f}")
                                            st.markdown(f"- Max: {col_info['max']:.2f}")
                                            st.markdown(f"- Médiane: {col_info.get('median', 'N/A')}")

                                    # Échantillon de valeurs
                                    if col_info['sample_values']:
                                        st.markdown("**Échantillon de valeurs:**")
                                        sample_str = ", ".join(str(v) for v in col_info['sample_values'][:10])
                                        st.code(sample_str)

                                    # Visualisation
                                    st.markdown("**Visualisation:**")

                                    if pd.api.types.is_numeric_dtype(df[col]):
                                        try:
                                            import plotly.express as px

                                            tab_viz1, tab_viz2 = st.tabs(["Histogramme", "Boîte à moustaches"])

                                            with tab_viz1:
                                                fig = px.histogram(df, x=col, title=f"Distribution de {col}")
                                                st.plotly_chart(fig, use_container_width=True, height=300)

                                            with tab_viz2:
                                                fig = px.box(df, y=col, title=f"Boîte à moustaches de {col}")
                                                st.plotly_chart(fig, use_container_width=True, height=300)

                                        except:
                                            st.info("📊 Visualisation non disponible")

                                    elif df[col].nunique() <= 20:
                                        try:
                                            import plotly.express as px

                                            value_counts = df[col].value_counts().head(10)
                                            fig = px.bar(x=value_counts.index, y=value_counts.values,
                                                         title=f"Fréquence des catégories - {col}")
                                            fig.update_layout(xaxis_title=col, yaxis_title="Fréquence")
                                            st.plotly_chart(fig, use_container_width=True, height=300)
                                        except:
                                            st.info("📊 Visualisation non disponible")

                    # Export des statistiques
                    st.markdown("---")
                    st.markdown("#### 📤 Export des statistiques")

                    # Convertir en JSON pour l'export
                    import json

                    stats_json = json.dumps(column_stats, indent=2, ensure_ascii=False)

                    col_exp1, col_exp2 = st.columns(2)

                    with col_exp1:
                        st.download_button(
                            label="💾 Télécharger JSON",
                            data=stats_json,
                            file_name="statistiques_colonnes.json",
                            mime="application/json",
                            use_container_width=True
                        )

                    with col_exp2:
                        # Export CSV du récapitulatif
                        csv_stats = stats_df.to_csv(index=False, encoding='utf-8-sig')
                        st.download_button(
                            label="💾 Télécharger CSV",
                            data=csv_stats,
                            file_name="recapitulatif_statistiques.csv",
                            mime="text/csv",
                            use_container_width=True
                        )

                except Exception as e:
                    st.error(f"❌ Erreur lors du calcul des statistiques: {str(e)}")
                    st.code(traceback.format_exc())

# ============================================================
# 🔒 NLQ SÉCURISÉ - VERSION FINALE OPTIMISÉE
# Interface de questions-réponses en langage naturel sur les données.
# Version optimisée avec gestion de la sécurité des données
# ============================================================
elif page == " 💬 Assistant IA":
    st.header("🔒 Assistant IA ")

    # Vérification des données
    if not st.session_state.data_loaded:
        st.warning("⚠️ Veuillez d'abord charger des données dans l'onglet 'Chargement des données'")
        st.stop()

    # ============================================
    # INITIALISATION DU MOTEUR NLQ
    # ============================================
    if st.session_state.nlq_engine is None:
        try:
            from modules.secure_nlq_engine import SecureNLQEngine

            # Récupération de la clé API
            api_key = None
            if hasattr(st.session_state, 'openai_client') and st.session_state.openai_client:
                api_key = st.session_state.openai_client.api_key
            elif 'api_key_input' in st.session_state and st.session_state.api_key_input:
                api_key = st.session_state.api_key_input

            if not api_key:
                st.error("❌ Aucune clé API OpenAI disponible")
                st.info("💡 Veuillez entrer une clé API valide dans la barre latérale")
                st.stop()

            # Initialisation du moteur
            st.session_state.nlq_engine = SecureNLQEngine(api_key=api_key)
            st.success("✅ Obtenez des réponses en toutes simplicité")

        except ImportError as e:
            st.error(f"❌ Erreur : {e}")
            st.info("""
            **📋 Vérifications nécessaires:**
            1. Le fichier `secure_nlq_engine.py` doit être dans `modules/`
            2. Installez: `pip install openai pandas plotly`
            """)
            st.stop()
        except Exception as e:
            st.error(f"❌ Erreur d'initialisation NLQ: {e}")
            st.info("🔑 Vérifiez votre clé API et votre connexion Internet")
            st.stop()

    # Vérification finale
    if st.session_state.nlq_engine is None:
        st.error("❌ Erreur dans le moteur de recherche")
        st.stop()

    # Récupération des objets
    df = st.session_state.dataframe
    nlq_engine = st.session_state.nlq_engine

    # Initialisation de l'historique
    if 'nlq_history' not in st.session_state:
        st.session_state.nlq_history = []

    # ============================================
    # ONGLETS PRINCIPAUX
    # ============================================
    tab1, tab2, tab3, tab4 = st.tabs([
        "📝 Analyse par requête",
        "📊 Métadonnées",
        "🔄 Historique",
        "⚙️ Configuration"
    ])

    # ============================================
    # TAB 1: ANALYSE PAR REQUÊTE
    # Interface principale pour poser des questions en langage naturel.
    # ============================================
    with tab1:
        st.subheader("🔍 Analyse NLQ Sécurisée")

        # Information importante
        st.info("""
        **🎯 ANALYSE INTELLIGENTE PAR IA**

        Posez des questions en langage naturel sur vos données et le contexte
        """)

        # Zone de requête
        col_query, col_tips = st.columns([3, 1])

        with col_query:
            st.markdown("### 💬 Posez vos questions ")

            user_query = st.text_area(
                "",
                height=150,
                placeholder="""
Quelle est la distribution des primes d'assurance par type de véhicule? """,
                help="Formulez votre question en français naturel. Soyez aussi précis que possible.",
                key="nlq_query_textarea"
            )

        with col_tips:
            st.markdown("**📋 Guide de formulation**")
            st.markdown("""
            **✅ Bonnes pratiques:**
            - Soyez spécifique
            - Mentionnez les variables clés
            - Définissez l'objectif
            - Précisez le contexte qui vous intéresse

            **❌ À éviter:**
            - Questions trop vagues
            - Termes ambigus
            - Multiples questions en une
            """)

        # Options d'analyse
        with st.expander("⚙️ Options d'analyse avancées", expanded=False):
            col_opt1, col_opt2 = st.columns(2)

            with col_opt1:
                use_data = st.checkbox(
                    "Utiliser les données réelles (génère des graphiques)",
                    value=True,
                    help="Active la génération de graphiques basés sur vos données"
                )

                max_samples = st.slider(
                    "Échantillons pour l'analyse",
                    min_value=100,
                    max_value=5000,
                    value=1000,
                    step=100,
                    help="Nombre d'échantillons à utiliser pour optimiser la performance"
                )

            with col_opt2:
                analysis_depth = st.select_slider(
                    "Profondeur d'analyse",
                    options=["Rapide", "Standard", "Approfondi"],
                    value="Standard"
                )

                include_viz = st.checkbox(
                    "Inclure les suggestions de visualisation",
                    value=True
                )

        # Bouton d'analyse
        st.markdown("---")

        if st.button("🚀 Demander", type="primary", use_container_width=True):
            if user_query and user_query.strip():
                with st.spinner("🧠 Analyse est en cours... Cela peut prendre juste quelques secondes."):
                    try:
                        # Choix du mode d'analyse
                        if use_data:
                            # Analyse avec données réelles
                            result = nlq_engine.analyze_query_with_data(
                                user_query=user_query,
                                dataframe=df,
                                max_samples=max_samples
                            )
                        else:
                            # Analyse avec métadonnées uniquement
                            if not st.session_state.metadata:
                                # Générer métadonnées basiques
                                from modules.metadata_extractor import MetadataExtractor

                                metadata_extractor = MetadataExtractor(df)
                                st.session_state.metadata = metadata_extractor.extract_safe_metadata()

                            result = nlq_engine.analyze_query_with_metadata(
                                user_query=user_query,
                                metadata=st.session_state.metadata
                            )

                        # Sauvegarder dans l'historique
                        st.session_state.nlq_history.append({
                            "query": user_query,
                            "timestamp": datetime.now().isoformat(),
                            "result": result,
                            "mode": "with_data" if use_data else "metadata_only"
                        })

                        # Afficher les résultats
                        if result.get("status") == "error":
                            st.error(f"❌ Erreur: {result.get('error', 'Erreur inconnue')}")
                        else:
                            st.success("✅ Analyse terminée avec succès!")

                            analysis = result.get("analysis", {})

                            # Intention et synthèse
                            st.markdown("### 🎯 Synthèse de l'analyse")
                            intention = analysis.get("intention", "Analyse générée")
                            st.info(f"**Objectif identifié:** {intention}")

                            # Stratégie d'analyse
                            st.markdown("### 📋 Méthodologie")
                            strategie = analysis.get("strategie_analyse", "")
                            if strategie:
                                st.markdown(strategie)

                            # Réponse détaillée
                            st.markdown("### 📝 Analyse complète")
                            reponse = analysis.get("reponse_detaillee", "")
                            if reponse:
                                st.markdown(reponse)

                            # Insights clés
                            insights = analysis.get("insights_cles", [])
                            if insights:
                                st.markdown("### 💡 Insights clés")
                                for i, insight in enumerate(insights, 1):
                                    st.markdown(f"{i}. {insight}")

                            # Recommandations
                            recommandations = analysis.get("recommandations", [])
                            if recommandations:
                                st.markdown("### 🎯 Recommandations")
                                for i, reco in enumerate(recommandations, 1):
                                    st.markdown(f"{i}. {reco}")

                            # Graphiques générés
                            graphs = result.get("graphs", {})
                            if graphs and graphs.get("generated"):
                                st.markdown("### 📊 Visualisations générées")

                                for graph in graphs["generated"]:
                                    st.markdown(f"#### {graph.get('description', graph['type'])}")
                                    st.markdown(f"*Variables: {', '.join(graph['variables'])}*")

                                    # Afficher le graphique HTML
                                    import streamlit.components.v1 as components

                                    components.html(graph["html"], height=500, scrolling=True)

                            # Informations sur l'échantillon
                            if "sample_info" in result:
                                with st.expander("ℹ️ Informations sur l'analyse", expanded=False):
                                    sample_info = result["sample_info"]
                                    col_info1, col_info2, col_info3 = st.columns(3)

                                    with col_info1:
                                        st.metric("Lignes analysées", f"{sample_info.get('sampled_rows', 0):,}")
                                    with col_info2:
                                        st.metric("Lignes totales", f"{sample_info.get('original_rows', 0):,}")
                                    with col_info3:
                                        st.metric("Colonnes", sample_info.get('columns', 0))

                    except Exception as e:
                        st.error(f"❌ Erreur lors de l'analyse: {str(e)}")
                        with st.expander("🔍 Détails de l'erreur"):
                            st.code(traceback.format_exc())
            else:
                st.warning("⚠️ Veuillez entrer une question")

    # ============================================
    # TAB 2: MÉTADONNÉES
    # # Affiche les métadonnées disponibles pour l'analyse NLQ.
    # ============================================
    with tab2:
        st.subheader("📊 Métadonnées disponibles pour l'analyse")

        if st.session_state.metadata:
            metadata = st.session_state.metadata

            # Vue d'ensemble
            col_meta1, col_meta2, col_meta3, col_meta4 = st.columns(4)

            general_info = metadata.get('general_info', {})
            with col_meta1:
                st.metric("Colonnes", general_info.get('nombre_colonnes', 0))
            with col_meta2:
                st.metric("Lignes", f"{general_info.get('nombre_lignes', 0):,}")
            with col_meta3:
                domaine = metadata.get('business_context_hints', {}).get('domaine', 'N/A')
                st.metric("Domaine", domaine.title())
            with col_meta4:
                completude = 100 - general_info.get('pourcentage_manquants_global', 0)
                st.metric("Complétude", f"{completude:.1f}%")

            # Variables clés
            st.markdown("#### 🔑 Variables clés identifiées")
            key_vars = metadata.get('business_context_hints', {}).get('variables_cles', [])
            if key_vars:
                cols_key = st.columns(min(len(key_vars), 5))
                for i, var in enumerate(key_vars[:5]):
                    with cols_key[i]:
                        st.info(f"**{var}**")
            else:
                st.info("Aucune variable clé identifiée automatiquement")

            # Structure des colonnes
            st.markdown("#### 🏗️ Structure des colonnes")

            columns_info = metadata.get('structure_columns', [])
            if columns_info:
                # Créer un DataFrame pour affichage
                display_data = []
                for col_info in columns_info[:20]:  # Limiter à 20
                    display_data.append({
                        "Colonne": col_info['nom'],
                        "Type": col_info['type_donnee'],
                        "Valeurs uniques": col_info.get('valeurs_uniques', 'N/A'),
                        "% Manquants": f"{col_info.get('pourcentage_manquants', 0):.1f}%"
                    })

                st.dataframe(pd.DataFrame(display_data), use_container_width=True)

                if len(columns_info) > 20:
                    st.info(f"... et {len(columns_info) - 20} autres colonnes")
        else:
            st.warning("⚠️ Aucune métadonnée disponible")
            st.info("Générez les métadonnées dans l'onglet 'Métadonnées' pour activer cette fonctionnalité")

    # ============================================
    # TAB 3: HISTORIQUE
    # Affiche l'historique des analyses NLQ effectuées.
    # ============================================
    with tab3:
        st.subheader("🔄 Historique des analyses NLQ")

        if st.session_state.nlq_history:
            st.info(f"📋 {len(st.session_state.nlq_history)} analyse(s) effectuée(s)")

            # Bouton pour effacer l'historique
            if st.button("🗑️ Effacer l'historique", type="secondary"):
                st.session_state.nlq_history = []
                st.success("✅ Historique effacé")
                st.rerun()

            st.markdown("---")

            # Afficher les analyses (les plus récentes en premier)
            for i, item in enumerate(reversed(st.session_state.nlq_history)):
                timestamp = item.get('timestamp', 'Date inconnue')
                query = item['query']
                mode = item.get('mode', 'unknown')

                # Icône selon le mode
                mode_icon = "📊" if mode == "with_data" else "📋"

                with st.expander(f"{mode_icon} {timestamp[:19]} - {query[:60]}...",
                                 expanded=(i == 0)):

                    st.markdown(f"**Question complète:** {query}")
                    st.caption(f"Mode: {'Avec données réelles' if mode == 'with_data' else 'Métadonnées uniquement'}")

                    result = item.get('result', {})

                    if result.get('status') == 'error':
                        st.error(f"❌ Erreur: {result.get('error', 'Erreur inconnue')}")
                    elif 'analysis' in result:
                        analysis = result['analysis']

                        # Intention
                        if 'intention' in analysis:
                            st.markdown(f"**🎯 Intention:** {analysis['intention']}")

                        # Stratégie
                        if 'strategie_analyse' in analysis:
                            st.markdown(f"**📋 Méthodologie:** {analysis['strategie_analyse'][:200]}...")

                        # Nombre de graphiques
                        graphs = result.get('graphs', {})
                        nb_graphs = len(graphs.get('generated', []))
                        if nb_graphs > 0:
                            st.success(f"✅ {nb_graphs} graphique(s) généré(s)")

                        # Bouton pour régénérer
                        if st.button(f"🔄 Régénérer cette analyse",
                                     key=f"regen_{i}",
                                     use_container_width=True):
                            st.session_state.nlq_quick_query = query
                            st.rerun()
        else:
            st.info("📭 Aucune analyse dans l'historique")
            st.markdown("Lancez votre première analyse dans l'onglet **'Analyse par requête'**")

    # ============================================
    # TAB 4: CONFIGURATION
    # Affiche la configuration du moteur NLQ et des options systèmes
    # ============================================
    with tab4:
        st.subheader("⚙️ Configuration du moteur NLQ")

        # Statut
        st.markdown("#### 🔧 Statut du système")

        col_status1, col_status2 = st.columns(2)

        with col_status1:
            if st.session_state.nlq_engine:
                st.success("✅ Moteur NLQ: Opérationnel")
                st.info(f"🤖 Modèle: {st.session_state.nlq_engine.model}")
            else:
                st.error("❌ Moteur NLQ: Non initialisé")

        with col_status2:
            if st.session_state.openai_client:
                st.success("✅ Client OpenAI: Connecté")
            else:
                st.warning("⚠️ Client OpenAI: Non connecté")

        # Sécurité
        st.markdown("#### 🔒 Niveau de sécurité")
        st.success("""
        **Mode sécurisé activé:**
        - ✅ Analyse sur métadonnées uniquement (mode par défaut)
        - ✅ Aucune donnée brute transmise à l'API
        - ✅ Anonymisation des requêtes
        - ✅ Données en local uniquement
        - ✅ Conformité RGPD garantie
        """)

        # Statistiques d'utilisation
        st.markdown("#### 📊 Statistiques d'utilisation")

        col_stats1, col_stats2, col_stats3 = st.columns(3)

        with col_stats1:
            st.metric("Analyses totales", len(st.session_state.nlq_history))

        with col_stats2:
            analyses_reussies = sum(1 for item in st.session_state.nlq_history
                                    if item.get('result', {}).get('status') == 'success')
            st.metric("Analyses réussies", analyses_reussies)

        with col_stats3:
            if st.session_state.nlq_history:
                dernier = st.session_state.nlq_history[-1]['timestamp']
                st.metric("Dernière analyse", dernier[:19])

        # Actions
        st.markdown("#### 🛠️ Actions")

        col_action1, col_action2 = st.columns(2)

        with col_action1:
            if st.button("🔄 Réinitialiser le moteur NLQ", use_container_width=True):
                st.session_state.nlq_engine = None
                st.success("✅ Moteur réinitialisé")
                st.info("Le moteur sera rechargé à la prochaine utilisation")

        with col_action2:
            if st.button("📥 Exporter l'historique (JSON)", use_container_width=True):
                if st.session_state.nlq_history:
                    import json

                    history_json = json.dumps(st.session_state.nlq_history,
                                              indent=2, ensure_ascii=False)
                    st.download_button(
                        label="💾 Télécharger l'historique",
                        data=history_json,
                        file_name=f"nlq_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                else:
                    st.warning("Aucun historique à exporter")

        # Aide
        with st.expander("❓ Aide et documentation", expanded=False):
            st.markdown("""
            ### Guide d'utilisation du moteur NLQ

            **1. Modes d'analyse:**
            - **Métadonnées uniquement**: Analyse rapide et sécurisée sur la structure des données
            - **Avec données réelles**: Génère des graphiques et analyses approfondies

            **2. Formulation des questions:**
            - Utilisez un langage naturel en français
            - Soyez précis sur les variables d'intérêt
            - Mentionnez l'objectif métier

            **3. Optimisation:**
            - Limitez le nombre d'échantillons pour des analyses rapides
            - Utilisez les exemples rapides pour démarrer
            - Consultez l'historique pour retrouver vos analyses

            **4. Sécurité:**
            - Vos données ne quittent jamais votre environnement
            - Seules les métadonnées sont utilisées (mode par défaut)
            - Conformité RGPD assurée
            """)
# ============================================================
# INSIGHTS AVANCÉS - VERSION COMPLÈTE AVEC TOUS LES GRAPHIQUES
# ============================================================
elif page == " 👁️ Visualisation des données":
    st.header(" 👁️ Visualisation des données et Analyse du Risque Client")

    # Vérifier qu'on a des données chargées
    if not st.session_state.data_loaded:
        st.warning("⚠️ Veuillez d'abord charger des données dans l'onglet 'Chargement des données'.")
        st.stop()

    # Prendre les données préparées si elles existent, sinon les données brutes
    if st.session_state.df_final is not None:
        df_final = st.session_state.df_final
    else:
        df_final = st.session_state.dataframe

    # Vérifier que les colonnes essentielles existent
    required_columns = ['ncli', 'nomncli', 'Prime', 'nb_jour_couv']
    available_columns = df_final.columns.tolist()
    missing_columns = [col for col in required_columns if col not in available_columns]

    if missing_columns:
        st.error(f"❌ Colonnes requises manquantes: {missing_columns}")
        st.info(f"Colonnes disponibles: {', '.join(available_columns[:10])}...")
        st.stop()

    # Initialiser le moteur d'insights s'il n'existe pas encore
    if st.session_state.insight_engine is None:
        try:
            from modules.insight_engine import InsightEngine

            st.session_state.insight_engine = InsightEngine()
            st.success("✅ Moteur d'insights initialisé")
        except ImportError as e:
            st.error(f"❌ Impossible de charger insight_engine: {e}")
            st.info("Assurez-vous que insight_engine.py est dans le dossier modules/")
            st.stop()

    insight_engine = st.session_state.insight_engine

    # Construction de la table client et calcul des scores de risque
    with st.spinner("🔍 Construction de la table client et calcul des risques..."):
        try:
            # Construire la table agrégée par client
            client_table = insight_engine.build_client_risk_table(df_final)

            # Calculer la médiane de la prime par jour pour référence
            ppj_median = client_table["prime_par_jour_moy"].median()

            # Calculer le score de risque pour chaque client
            scored_clients = insight_engine.compute_risk_score(client_table)

            # Générer un insight personnalisé pour chaque client
            scored_clients["insight"] = scored_clients.apply(
                lambda row: insight_engine.generate_client_insight(row, ppj_median),
                axis=1
            )

            # Sauvegarder tout ça dans la session
            st.session_state.scored_clients = scored_clients
            st.session_state.client_table = client_table
            st.session_state.raw_data = df_final

            st.success(f"✅ Analyse des risques complétée pour {len(scored_clients)} clients!")

        except Exception as e:
            st.error(f"❌ Erreur lors de l'analyse des risques: {str(e)}")
            st.code(traceback.format_exc())
            st.stop()

    st.markdown("---")

    # Menu pour choisir le type d'analyse
    analysis_type = st.radio(
        "**Sélectionnez le type d'analyse :**",
        [
            "📊 Vue d'ensemble",
            "📈 Analyse Univariée",
            "📉 Analyse Bivariée",
            "🎯 Analyse Multivariée",
            "📄 Rapport Narratif"
        ],
        horizontal=True
    )

    # ============================================================
    # VUE D'ENSEMBLE
    # ============================================================
    if analysis_type == "📊 Vue d'ensemble":
        st.subheader("📊 Vue d'ensemble du portefeuille")

        # Afficher les métriques principales
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Clients", f"{len(scored_clients):,}")
        with col2:
            st.metric("Prime totale", f"{scored_clients['prime_totale'].sum():,.0f} MAD")
        with col3:
            st.metric("Score risque moyen", f"{scored_clients['score_risque'].mean():.1f}/100")
        with col4:
            high_risk = (scored_clients['niveau_risque'] == 'Élevé').sum()
            st.metric("Risque élevé", f"{high_risk} clients")

        # Générer et afficher les insights clés
        st.subheader("🎯 Insights clés")
        insights = insight_engine.generate_insights(scored_clients)
        for ins in insights:
            st.markdown(f"• {ins}")

        # Graphiques de distribution
        col_viz1, col_viz2 = st.columns(2)

        with col_viz1:
            # Graphique en camembert de la répartition des risques
            st.subheader("📊 Distribution des risques")
            risk_dist = scored_clients["niveau_risque"].value_counts()
            fig_pie = go.Figure(data=[go.Pie(
                labels=risk_dist.index,
                values=risk_dist.values,
                hole=.3,
                marker_colors=['#2ECC71', '#F39C12', '#E74C3C'],
                textinfo='label+percent',
                textposition='inside'
            )])
            fig_pie.update_layout(
                title="Répartition par niveau de risque",
                height=400
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_viz2:
            # Histogramme des scores de risque
            st.subheader("📈 Distribution des scores")
            fig_hist = px.histogram(
                scored_clients,
                x='score_risque',
                nbins=30,
                title="Distribution des scores de risque",
                color_discrete_sequence=['#3498DB']
            )
            fig_hist.update_layout(
                xaxis_title="Score de risque",
                yaxis_title="Nombre de clients",
                height=400
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        # Graphique de répartition par tranches
        st.subheader("📊 Répartition par tranches de risque")

        # Créer des tranches de score
        scored_clients['tranche_score'] = pd.cut(
            scored_clients['score_risque'],
            bins=[0, 25, 50, 75, 100],
            labels=['0-25', '26-50', '51-75', '76-100']
        )

        tranche_counts = scored_clients['tranche_score'].value_counts().sort_index()
        fig_tranche = go.Figure(data=[go.Bar(
            x=tranche_counts.index.astype(str),
            y=tranche_counts.values,
            marker_color=['#2ECC71', '#F1C40F', '#E67E22', '#E74C3C'],
            text=tranche_counts.values,
            textposition='auto'
        )])
        fig_tranche.update_layout(
            title="Nombre de clients par tranche de score",
            xaxis_title="Tranche de score",
            yaxis_title="Nombre de clients",
            height=400
        )
        st.plotly_chart(fig_tranche, use_container_width=True)

        # Top 10 des clients à risque élevé
        st.subheader("🔴 Top 10 clients à risque élevé")
        high_risk_clients = scored_clients[scored_clients['niveau_risque'] == 'Élevé'].sort_values(
            'score_risque', ascending=False
        ).head(10)

        # Colonnes à afficher
        display_columns = ['nomncli', 'score_risque', 'prime_totale', 'frequence_sinistre', 'insight']
        available_display = [col for col in display_columns if col in high_risk_clients.columns]

        if available_display:
            st.dataframe(
                high_risk_clients[available_display],
                use_container_width=True
            )
        else:
            st.info("⚠️ Aucune donnée disponible pour les clients à haut risque")

    # ============================================================
    # ANALYSE UNIVARIÉE ENRICHIE
    # ============================================================
    elif analysis_type == "📈 Analyse Univariée":
        st.subheader("📈 Analyse Univariée Approfondie")

        col1, col2 = st.columns(2)

        with col1:
            # Lister les variables numériques disponibles
            numeric_cols = scored_clients.select_dtypes(include=[np.number]).columns.tolist()
            selected_var = st.selectbox(
                "Sélectionnez une variable numérique :",
                options=numeric_cols,
                index=numeric_cols.index('score_risque') if 'score_risque' in numeric_cols else 0
            )

        with col2:
            chart_type = st.selectbox(
                "Type de visualisation :",
                ["Tous les graphiques", "Histogramme", "Boîte à moustaches", "Violin plot", "Statistiques descriptives"]
            )

        # Statistiques descriptives de la variable
        st.subheader(f"📊 Statistiques de : {selected_var}")

        col_stat1, col_stat2, col_stat3, col_stat4, col_stat5 = st.columns(5)
        with col_stat1:
            st.metric("Moyenne", f"{scored_clients[selected_var].mean():.2f}")
        with col_stat2:
            st.metric("Médiane", f"{scored_clients[selected_var].median():.2f}")
        with col_stat3:
            st.metric("Écart-type", f"{scored_clients[selected_var].std():.2f}")
        with col_stat4:
            st.metric("Min", f"{scored_clients[selected_var].min():.2f}")
        with col_stat5:
            st.metric("Max", f"{scored_clients[selected_var].max():.2f}")

        # Afficher les graphiques selon le type choisi
        if chart_type == "Tous les graphiques":
            col_g1, col_g2 = st.columns(2)

            with col_g1:
                # Histogramme
                fig_hist = px.histogram(
                    scored_clients,
                    x=selected_var,
                    nbins=30,
                    title=f"Histogramme - {selected_var}",
                    color_discrete_sequence=['#3498DB']
                )
                fig_hist.update_layout(height=350)
                st.plotly_chart(fig_hist, use_container_width=True)

                # Violin plot
                fig_violin = px.violin(
                    scored_clients,
                    y=selected_var,
                    box=True,
                    title=f"Violin Plot - {selected_var}",
                    color_discrete_sequence=['#9B59B6']
                )
                fig_violin.update_layout(height=350)
                st.plotly_chart(fig_violin, use_container_width=True)

            with col_g2:
                # Boîte à moustaches
                fig_box = px.box(
                    scored_clients,
                    y=selected_var,
                    title=f"Boîte à moustaches - {selected_var}",
                    color_discrete_sequence=['#E74C3C']
                )
                fig_box.update_layout(height=350)
                st.plotly_chart(fig_box, use_container_width=True)

                # Courbe de densité
                fig_density = go.Figure()
                fig_density.add_trace(go.Histogram(
                    x=scored_clients[selected_var],
                    histnorm='probability density',
                    name='Densité',
                    marker_color='#1ABC9C',
                    nbinsx=30
                ))
                fig_density.update_layout(
                    title=f"Courbe de densité - {selected_var}",
                    height=350
                )
                st.plotly_chart(fig_density, use_container_width=True)

        elif chart_type == "Histogramme":
            fig = px.histogram(
                scored_clients,
                x=selected_var,
                nbins=30,
                title=f"Histogramme - {selected_var}",
                color_discrete_sequence=['#3498DB']
            )
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Boîte à moustaches":
            fig = px.box(
                scored_clients,
                y=selected_var,
                title=f"Boîte à moustaches - {selected_var}",
                color_discrete_sequence=['#E74C3C']
            )
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Violin plot":
            fig = px.violin(
                scored_clients,
                y=selected_var,
                box=True,
                title=f"Violin Plot - {selected_var}",
                color_discrete_sequence=['#9B59B6']
            )
            st.plotly_chart(fig, use_container_width=True)

        else:  # Statistiques descriptives
            stats_df = scored_clients[selected_var].describe()
            st.dataframe(stats_df, use_container_width=True)

        # Analyse des variables catégorielles
        st.markdown("---")
        st.subheader("📊 Analyse des variables catégorielles")

        categorical_cols = scored_clients.select_dtypes(include=['object', 'category']).columns.tolist()

        if categorical_cols:
            cat_var = st.selectbox(
                "Sélectionnez une variable catégorielle :",
                options=categorical_cols,
                index=categorical_cols.index('niveau_risque') if 'niveau_risque' in categorical_cols else 0
            )

            if cat_var in scored_clients.columns:
                cat_dist = scored_clients[cat_var].value_counts()

                col_cat1, col_cat2 = st.columns(2)

                with col_cat1:
                    # Graphique en barres
                    fig_bar = go.Figure(data=[go.Bar(
                        x=cat_dist.index,
                        y=cat_dist.values,
                        marker_color='#3498DB',
                        text=cat_dist.values,
                        textposition='auto'
                    )])
                    fig_bar.update_layout(
                        title=f"Distribution de {cat_var}",
                        xaxis_title=cat_var,
                        yaxis_title="Nombre",
                        height=400
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)

                with col_cat2:
                    # Graphique en secteurs
                    fig_pie_cat = px.pie(
                        values=cat_dist.values,
                        names=cat_dist.index,
                        title=f"Répartition de {cat_var}"
                    )
                    fig_pie_cat.update_layout(height=400)
                    st.plotly_chart(fig_pie_cat, use_container_width=True)
        else:
            st.info("Aucune variable catégorielle disponible")

    # ============================================================
    # ANALYSE BIVARIÉE ENRICHIE
    # ============================================================
    elif analysis_type == "📉 Analyse Bivariée":
        st.subheader("📉 Analyse Bivariée Approfondie")

        numeric_cols = scored_clients.select_dtypes(include=[np.number]).columns.tolist()

        col1, col2, col3 = st.columns(3)
        with col1:
            x_var = st.selectbox(
                "Variable X :",
                options=numeric_cols,
                index=numeric_cols.index('prime_par_jour_moy') if 'prime_par_jour_moy' in numeric_cols else 0
            )

        with col2:
            y_var = st.selectbox(
                "Variable Y :",
                options=numeric_cols,
                index=numeric_cols.index('score_risque') if 'score_risque' in numeric_cols else 1
            )

        with col3:
            color_var = st.selectbox(
                "Variable de couleur :",
                options=['Aucune'] + scored_clients.columns.tolist(),
                index=0
            )

        # Nuage de points principal
        st.subheader("📊 Nuage de points")
        fig_scatter = px.scatter(
            scored_clients,
            x=x_var,
            y=y_var,
            color=color_var if color_var != 'Aucune' else None,
            title=f"Relation entre {x_var} et {y_var}",
            trendline="ols",
            height=500
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        # Calculer et afficher la corrélation
        try:
            correlation = scored_clients[[x_var, y_var]].corr().iloc[0, 1]

            col_corr1, col_corr2, col_corr3 = st.columns(3)
            with col_corr1:
                st.metric("Coefficient de corrélation", f"{correlation:.3f}")

            with col_corr2:
                # Interpréter la force de la corrélation
                if correlation > 0.7:
                    st.info("📈 Forte corrélation positive")
                elif correlation > 0.3:
                    st.info("↗️ Corrélation positive modérée")
                elif correlation < -0.7:
                    st.info("📉 Forte corrélation négative")
                elif correlation < -0.3:
                    st.info("↘️ Corrélation négative modérée")
                else:
                    st.info("➡️ Faible ou pas de corrélation")

            with col_corr3:
                # Coefficient de détermination
                r2 = correlation ** 2
                st.metric("R² (variance expliquée)", f"{r2:.3f}")

            # Interprétation
            st.markdown("---")
            st.subheader("💡 Interprétation")
            if correlation > 0:
                st.markdown(f"**Relation positive :** Quand {x_var} augmente, {y_var} tend à augmenter")
            elif correlation < 0:
                st.markdown(f"**Relation négative :** Quand {x_var} augmente, {y_var} tend à diminuer")
            else:
                st.markdown("**Pas de relation linéaire évidente** entre les deux variables")

            # Graphiques supplémentaires
            st.markdown("---")
            st.subheader("📊 Analyses complémentaires")

            col_supp1, col_supp2 = st.columns(2)

            with col_supp1:
                # Heatmap de densité
                fig_density_2d = go.Figure(go.Histogram2d(
                    x=scored_clients[x_var],
                    y=scored_clients[y_var],
                    colorscale='Blues'
                ))
                fig_density_2d.update_layout(
                    title=f"Carte de densité - {x_var} vs {y_var}",
                    xaxis_title=x_var,
                    yaxis_title=y_var,
                    height=400
                )
                st.plotly_chart(fig_density_2d, use_container_width=True)

            with col_supp2:
                # Box plot groupé si une variable de couleur est sélectionnée
                if color_var != 'Aucune' and scored_clients[color_var].nunique() <= 10:
                    fig_box_grouped = px.box(
                        scored_clients,
                        x=color_var,
                        y=y_var,
                        title=f"{y_var} par {color_var}",
                        color=color_var
                    )
                    fig_box_grouped.update_layout(height=400)
                    st.plotly_chart(fig_box_grouped, use_container_width=True)
                else:
                    # Scatter plot avec taille de points
                    fig_bubble = px.scatter(
                        scored_clients,
                        x=x_var,
                        y=y_var,
                        size=abs(scored_clients[y_var]),
                        title=f"Bubble chart - {x_var} vs {y_var}",
                        opacity=0.6
                    )
                    fig_bubble.update_layout(height=400)
                    st.plotly_chart(fig_bubble, use_container_width=True)

        except Exception as e:
            st.warning(f"Impossible de calculer la corrélation: {str(e)}")

    # ============================================================
    # ANALYSE MULTIVARIÉE COMPLÈTE
    # ============================================================
    elif analysis_type == "🎯 Analyse Multivariée":
        st.subheader("🎯 Analyse Multivariée Complète")

        # Menu pour choisir le type d'analyse multivariée
        multivariate_choice = st.selectbox(
            "Choisissez l'analyse multivariée :",
            [
                "Analyse en Composantes Principales (ACP)",
                "Analyse des Correspondances Multiples (ACM)",
                "Clustering (K-means)",
                "Matrice de corrélation"
            ]
        )

        if multivariate_choice == "Analyse en Composantes Principales (ACP)":
            st.subheader("🔮 Analyse en Composantes Principales (ACP)")

            # Récupérer les variables disponibles pour l'ACP
            available_vars = insight_engine.get_available_variables_for_pca(scored_clients)

            if len(available_vars) < 2:
                st.warning(f"⚠️ Pas assez de variables numériques pour l'ACP. Variables disponibles: {available_vars}")
                st.stop()

            # Permettre à l'utilisateur de sélectionner les variables
            selected_vars = st.multiselect(
                "Sélectionnez les variables pour l'ACP :",
                options=available_vars,
                default=available_vars[:min(5, len(available_vars))]
            )

            if len(selected_vars) < 2:
                st.error("❌ Veuillez sélectionner au moins 2 variables pour l'ACP.")
                st.stop()

            # Paramètres de l'ACP
            col_param1, col_param2 = st.columns(2)
            with col_param1:
                n_components = st.slider(
                    "Nombre de composantes principales :",
                    min_value=2,
                    max_value=min(10, len(selected_vars)),
                    value=min(3, len(selected_vars))
                )
            with col_param2:
                scale_data = st.checkbox("Standardiser les données", value=True)

            if st.button("🔬 Lancer l'ACP", type="primary"):
                with st.spinner("Calcul de l'ACP en cours..."):
                    try:
                        from sklearn.decomposition import PCA
                        from sklearn.preprocessing import StandardScaler

                        # Préparer les données
                        X_pca = scored_clients[selected_vars].dropna()

                        if scale_data:
                            scaler = StandardScaler()
                            X_scaled = scaler.fit_transform(X_pca)
                        else:
                            X_scaled = X_pca.values

                        # Appliquer l'ACP
                        pca = PCA(n_components=n_components)
                        components = pca.fit_transform(X_scaled)

                        st.success("✅ ACP terminée avec succès !")

                        # Variance expliquée
                        st.subheader("📊 Variance expliquée")

                        col_var1, col_var2 = st.columns(2)

                        with col_var1:
                            # Graphique de la variance expliquée
                            fig_var = go.Figure()
                            fig_var.add_trace(go.Bar(
                                x=[f'CP{i + 1}' for i in range(n_components)],
                                y=pca.explained_variance_ratio_ * 100,
                                marker_color='#3498DB',
                                text=[f'{v:.1f}%' for v in pca.explained_variance_ratio_ * 100],
                                textposition='auto'
                            ))
                            fig_var.update_layout(
                                title="Variance expliquée par composante",
                                xaxis_title="Composante",
                                yaxis_title="Variance expliquée (%)",
                                height=400
                            )
                            st.plotly_chart(fig_var, use_container_width=True)

                        with col_var2:
                            # Variance cumulée
                            cumsum = np.cumsum(pca.explained_variance_ratio_ * 100)
                            fig_cumsum = go.Figure()
                            fig_cumsum.add_trace(go.Scatter(
                                x=[f'CP{i + 1}' for i in range(n_components)],
                                y=cumsum,
                                mode='lines+markers',
                                marker=dict(size=10, color='#E74C3C'),
                                line=dict(width=3)
                            ))
                            fig_cumsum.update_layout(
                                title="Variance cumulée",
                                xaxis_title="Composante",
                                yaxis_title="Variance cumulée (%)",
                                height=400
                            )
                            st.plotly_chart(fig_cumsum, use_container_width=True)

                        # Contributions des variables
                        st.subheader("🎯 Contributions des variables")

                        # Calculer les loadings
                        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
                        loadings_df = pd.DataFrame(
                            loadings,
                            columns=[f'CP{i + 1}' for i in range(n_components)],
                            index=selected_vars
                        )

                        # Heatmap des contributions
                        fig_loadings = px.imshow(
                            loadings_df.T,
                            title="Contributions des variables aux composantes",
                            labels=dict(x="Variable", y="Composante", color="Contribution"),
                            color_continuous_scale='RdBu',
                            aspect='auto'
                        )
                        fig_loadings.update_layout(height=400)
                        st.plotly_chart(fig_loadings, use_container_width=True)

                        # Tableau des contributions
                        st.dataframe(
                            loadings_df.style.background_gradient(cmap='RdBu', axis=0),
                            use_container_width=True
                        )

                        # Projection des individus
                        st.subheader("📍 Projection des individus")

                        # Créer un dataframe avec les composantes
                        pca_df = pd.DataFrame(
                            components[:, :2],
                            columns=['CP1', 'CP2']
                        )

                        # Ajouter les informations des clients
                        pca_df['client'] = X_pca.index
                        if 'niveau_risque' in scored_clients.columns:
                            pca_df['niveau_risque'] = scored_clients.loc[X_pca.index, 'niveau_risque'].values

                        # Nuage de points des 2 premières composantes
                        fig_proj = px.scatter(
                            pca_df,
                            x='CP1',
                            y='CP2',
                            color='niveau_risque' if 'niveau_risque' in pca_df.columns else None,
                            title="Projection sur les 2 premières composantes principales",
                            hover_data=['client']
                        )
                        fig_proj.update_layout(height=500)
                        st.plotly_chart(fig_proj, use_container_width=True)

                        # Cercle de corrélation (biplot)
                        if n_components >= 2:
                            st.subheader("🔵 Cercle de corrélation")

                            fig_circle = go.Figure()

                            # Ajouter le cercle
                            theta = np.linspace(0, 2 * np.pi, 100)
                            fig_circle.add_trace(go.Scatter(
                                x=np.cos(theta),
                                y=np.sin(theta),
                                mode='lines',
                                line=dict(color='gray', dash='dash'),
                                showlegend=False
                            ))

                            # Ajouter les vecteurs des variables
                            for i, var in enumerate(selected_vars):
                                fig_circle.add_trace(go.Scatter(
                                    x=[0, loadings[i, 0]],
                                    y=[0, loadings[i, 1]],
                                    mode='lines+markers+text',
                                    name=var,
                                    text=['', var],
                                    textposition='top center',
                                    marker=dict(size=8)
                                ))

                            fig_circle.update_layout(
                                title="Cercle de corrélation (CP1 vs CP2)",
                                xaxis_title=f"CP1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)",
                                yaxis_title=f"CP2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)",
                                height=600,
                                showlegend=True
                            )
                            fig_circle.update_xaxes(range=[-1.1, 1.1], zeroline=True)
                            fig_circle.update_yaxes(range=[-1.1, 1.1], zeroline=True)
                            st.plotly_chart(fig_circle, use_container_width=True)

                    except Exception as e:
                        st.error(f"❌ Erreur lors de l'ACP: {str(e)}")
                        st.code(traceback.format_exc())
        elif multivariate_choice == "Analyse des Correspondances Multiples (ACM)":
            st.subheader("🎭 Analyse des Correspondances Multiples (ACM)")

            # Récupérer les variables catégorielles
            available_vars = insight_engine.get_available_variables_for_acm(scored_clients)

            if len(available_vars) < 2:
                st.warning(f"⚠️ Pas assez de variables catégorielles pour l'ACM. Variables disponibles: {available_vars}")
                st.stop()

            # Sélection des variables
            selected_vars = st.multiselect(
                "Sélectionnez les variables catégorielles pour l'ACM :",
                options=available_vars,
                default=available_vars[:min(5, len(available_vars))]
            )

            if len(selected_vars) < 2:
                st.error("❌ Veuillez sélectionner au moins 2 variables pour l'ACM.")
                st.stop()

            if st.button("🔬 Lancer l'ACM", type="primary"):
                with st.spinner("Calcul de l'ACM en cours..."):
                    try:
                        # Exécuter l'ACM
                        mca_result = insight_engine.perform_acm_analysis(
                            scored_clients,
                            selected_vars
                        )

                        if 'error' in mca_result:
                            st.error(f"❌ Erreur ACM: {mca_result['error']}")
                        else:
                            st.success("✅ ACM terminée avec succès !")

                            # Visualiser les résultats
                            fig_mca = insight_engine.create_mca_visualization(mca_result, scored_clients)
                            if fig_mca:
                                st.plotly_chart(fig_mca, use_container_width=True)

                    except Exception as e:
                        st.error(f"❌ Erreur lors de l'ACM: {str(e)}")
                        st.code(traceback.format_exc())

        elif multivariate_choice == "Clustering (K-means)":
            st.subheader("👥 Clustering (K-means)")

            # Préparer les variables pour le clustering
            numeric_cols = scored_clients.select_dtypes(include=[np.number]).columns.tolist()
            exclude_cols = ['ncli', 'score_risque']
            analysis_cols = [col for col in numeric_cols if col not in exclude_cols]

            if len(analysis_cols) >= 2:
                # Proposer des variables recommandées basées sur vos colonnes
                recommended_vars = ['prime_par_jour_moy', 'duree_moyenne', 'nb_avenants',
                                    'frequence_sinistre', 'retard_paiement_moyen']
                available_recommended = [v for v in recommended_vars if v in analysis_cols]

                clustering_features = st.multiselect(
                    "Sélectionnez les variables pour le clustering :",
                    options=analysis_cols,
                    default=available_recommended[
                        :min(3, len(available_recommended))] if available_recommended else analysis_cols[
                        :min(3, len(analysis_cols))]
                )

                if len(clustering_features) < 2:
                    st.error("❌ Veuillez sélectionner au moins 2 variables pour le clustering.")
                    st.stop()

                # Paramètres du clustering
                col_clust1, col_clust2 = st.columns(2)
                with col_clust1:
                    n_clusters = st.slider("Nombre de clusters :", min_value=2, max_value=10, value=3)
                with col_clust2:
                    show_elbow = st.checkbox("Afficher la méthode du coude", value=True)

                if st.button("🔬 Lancer le Clustering", type="primary"):
                    with st.spinner("Clustering en cours..."):
                        try:
                            from sklearn.cluster import KMeans
                            from sklearn.preprocessing import StandardScaler

                            # Préparer les données
                            X_cluster = scored_clients[clustering_features].fillna(0)

                            # Normaliser les données
                            scaler = StandardScaler()
                            X_scaled = scaler.fit_transform(X_cluster)

                            # Méthode du coude si demandée
                            if show_elbow:
                                st.subheader("📉 Méthode du coude")
                                inertias = []
                                K_range = range(2, min(11, len(X_cluster)))

                                for k in K_range:
                                    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
                                    kmeans_temp.fit(X_scaled)
                                    inertias.append(kmeans_temp.inertia_)

                                fig_elbow = go.Figure()
                                fig_elbow.add_trace(go.Scatter(
                                    x=list(K_range),
                                    y=inertias,
                                    mode='lines+markers',
                                    marker=dict(size=10, color='#E74C3C'),
                                    line=dict(width=3)
                                ))
                                fig_elbow.update_layout(
                                    title="Méthode du coude - Détermination du nombre optimal de clusters",
                                    xaxis_title="Nombre de clusters",
                                    yaxis_title="Inertie",
                                    height=400
                                )
                                st.plotly_chart(fig_elbow, use_container_width=True)

                            # Appliquer K-means
                            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                            clusters = kmeans.fit_predict(X_scaled)

                            # Ajouter les clusters au dataframe
                            scored_clients_clustered = scored_clients.copy()
                            scored_clients_clustered['Cluster'] = clusters

                            st.success(f"✅ {n_clusters} clusters identifiés")

                            # Distribution des clusters
                            st.subheader("📊 Distribution des clusters")

                            col_dist1, col_dist2 = st.columns(2)

                            with col_dist1:
                                cluster_counts = pd.Series(clusters).value_counts().sort_index()
                                fig_clusters = go.Figure(data=[go.Bar(
                                    x=cluster_counts.index,
                                    y=cluster_counts.values,
                                    marker_color='#3498DB',
                                    text=cluster_counts.values,
                                    textposition='auto'
                                )])
                                fig_clusters.update_layout(
                                    title=f'Distribution des {n_clusters} clusters',
                                    xaxis_title='Cluster',
                                    yaxis_title='Nombre de clients',
                                    height=400
                                )
                                st.plotly_chart(fig_clusters, use_container_width=True)

                            with col_dist2:
                                # Pie chart des clusters
                                fig_pie_clust = px.pie(
                                    values=cluster_counts.values,
                                    names=[f'Cluster {i}' for i in cluster_counts.index],
                                    title="Répartition en %"
                                )
                                fig_pie_clust.update_layout(height=400)
                                st.plotly_chart(fig_pie_clust, use_container_width=True)

                            # Caractérisation des clusters
                            st.subheader("📊 Caractérisation des clusters")

                            # Adapter l'agrégation selon les colonnes disponibles
                            agg_dict = {}
                            if 'score_risque' in scored_clients_clustered.columns:
                                agg_dict['score_risque'] = 'mean'
                            if 'prime_totale' in scored_clients_clustered.columns:
                                agg_dict['prime_totale'] = 'mean'
                            if 'frequence_sinistre' in scored_clients_clustered.columns:
                                agg_dict['frequence_sinistre'] = 'mean'
                            if 'niveau_risque' in scored_clients_clustered.columns:
                                agg_dict['niveau_risque'] = lambda x: x.mode()[0] if not x.mode().empty else 'N/A'
                            if 'nomncli' in scored_clients_clustered.columns:
                                agg_dict['nomncli'] = 'count'

                            if agg_dict:
                                cluster_summary = scored_clients_clustered.groupby('Cluster').agg(agg_dict).round(2)

                                # Renommer les colonnes
                                rename_dict = {
                                    'score_risque': 'Score moyen',
                                    'prime_totale': 'Prime moyenne',
                                    'frequence_sinistre': 'Sinistres moyens',
                                    'niveau_risque': 'Risque dominant',
                                    'nomncli': 'Nombre clients'
                                }
                                cluster_summary = cluster_summary.rename(columns=rename_dict)

                                st.dataframe(
                                    cluster_summary.style.background_gradient(cmap='YlOrRd', axis=0),
                                    use_container_width=True
                                )

                            # Visualisation 2D des clusters
                            st.subheader("🗺️ Visualisation des clusters")

                            if len(clustering_features) >= 2:
                                fig_scatter_clust = px.scatter(
                                    scored_clients_clustered,
                                    x=clustering_features[0],
                                    y=clustering_features[1],
                                    color='Cluster',
                                    title=f"Clusters projetés sur {clustering_features[0]} vs {clustering_features[1]}",
                                    height=500
                                )

                                # Ajouter les centres des clusters
                                centers = scaler.inverse_transform(kmeans.cluster_centers_)
                                fig_scatter_clust.add_trace(go.Scatter(
                                    x=centers[:, 0],
                                    y=centers[:, 1],
                                    mode='markers',
                                    marker=dict(
                                        size=20,
                                        color='red',
                                        symbol='x',
                                        line=dict(width=2, color='white')
                                    ),
                                    name='Centres'
                                ))

                                st.plotly_chart(fig_scatter_clust, use_container_width=True)

                            # Profils radar des clusters
                            st.subheader("🎯 Profils radar des clusters")

                            if len(clustering_features) >= 3:
                                fig_radar = go.Figure()

                                for cluster_id in range(n_clusters):
                                    cluster_data = scored_clients_clustered[
                                        scored_clients_clustered['Cluster'] == cluster_id]
                                    values = [cluster_data[feat].mean() for feat in clustering_features]

                                    # Normaliser entre 0 et 1
                                    values_norm = [(v - scored_clients_clustered[feat].min()) /
                                                   (scored_clients_clustered[feat].max() - scored_clients_clustered[
                                                       feat].min())
                                                   for v, feat in zip(values, clustering_features)]

                                    fig_radar.add_trace(go.Scatterpolar(
                                        r=values_norm + [values_norm[0]],
                                        theta=clustering_features + [clustering_features[0]],
                                        fill='toself',
                                        name=f'Cluster {cluster_id}'
                                    ))

                                fig_radar.update_layout(
                                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                                    showlegend=True,
                                    title="Profils moyens des clusters (normalisés)",
                                    height=500
                                )
                                st.plotly_chart(fig_radar, use_container_width=True)

                        except Exception as e:
                            st.error(f"❌ Erreur lors du clustering: {str(e)}")
                            st.code(traceback.format_exc())
            else:
                st.warning("⚠️ Pas assez de variables numériques pour le clustering")

        elif multivariate_choice == "Matrice de corrélation":
            st.subheader("🔗 Matrice de corrélation complète")

            # Sélection des variables
            numeric_cols = scored_clients.select_dtypes(include=[np.number]).columns.tolist()

            selected_corr_vars = st.multiselect(
                "Sélectionnez les variables pour la matrice de corrélation :",
                options=numeric_cols,
                default=numeric_cols[:min(10, len(numeric_cols))]
            )

            if len(selected_corr_vars) >= 2:
                # Calculer la matrice de corrélation
                corr_matrix = scored_clients[selected_corr_vars].corr()

                # Heatmap de corrélation
                fig_corr = px.imshow(
                    corr_matrix,
                    title="Matrice de corrélation",
                    color_continuous_scale='RdBu',
                    zmin=-1,
                    zmax=1,
                    aspect='auto'
                )
                fig_corr.update_layout(height=600)
                st.plotly_chart(fig_corr, use_container_width=True)

                # Top des corrélations
                st.subheader("🔝 Top 15 des corrélations les plus fortes")

                # Extraire les corrélations
                correlations = corr_matrix.unstack()
                # Enlever les auto-corrélations
                correlations = correlations[correlations != 1]
                # Enlever les doublons
                correlations = correlations[correlations.index.get_level_values(0) < correlations.index.get_level_values(1)]
                # Trier par valeur absolue
                top_correlations = correlations.abs().sort_values(ascending=False).head(15)

                # Créer un dataframe pour l'affichage
                corr_df = pd.DataFrame({
                    'Variable 1': [idx[0] for idx in top_correlations.index],
                    'Variable 2': [idx[1] for idx in top_correlations.index],
                    'Corrélation': [correlations[idx] for idx in top_correlations.index]
                })

                st.dataframe(
                    corr_df.style.background_gradient(cmap='RdBu', subset=['Corrélation'], vmin=-1, vmax=1),
                    use_container_width=True
                )

                # Graphique des top corrélations
                fig_top_corr = go.Figure(go.Bar(
                    x=corr_df['Corrélation'],
                    y=[f"{row['Variable 1']} - {row['Variable 2']}" for _, row in corr_df.iterrows()],
                    orientation='h',
                    marker_color=['red' if x < 0 else 'blue' for x in corr_df['Corrélation']]
                ))
                fig_top_corr.update_layout(
                    title="Top 15 des corrélations",
                    xaxis_title="Coefficient de corrélation",
                    height=500
                )
                st.plotly_chart(fig_top_corr, use_container_width=True)
        else:
            st.warning("⚠️ Sélectionnez au moins 2 variables")

    # ============================================================
    # RAPPORT NARRATIF
    # ============================================================
    elif analysis_type == "📄 Rapport Narratif":
        st.subheader("📄 Rapport narratif pour décideur")

        # Générer le rapport complet
        report = insight_engine.generate_narrative_report(scored_clients)

        # Afficher avec un formatage propre
        st.markdown("### 📄 Rapport d'Analyse Complet")
        st.markdown("---")
        st.markdown(report)

        # Options d'export
        st.markdown("---")
        st.subheader("📤 Exporter le rapport")

        col1, col2 = st.columns(2)

        with col1:
            # Export en Markdown
            report_md = report
            st.download_button(
                label="📄 Télécharger en Markdown",
                data=report_md,
                file_name="rapport_risque_clients.md",
                mime="text/markdown"
            )

        with col2:
            # Export des scores en CSV - adapter selon colonnes disponibles
            export_cols = ['nomncli', 'score_risque', 'niveau_risque']

            # Ajouter les colonnes optionnelles si elles existent
            optional_cols = ['prime_totale', 'frequence_sinistre', 'insight']
            for col in optional_cols:
                if col in scored_clients.columns:
                    export_cols.append(col)

            csv_data = scored_clients[export_cols].to_csv(index=False, encoding='utf-8-sig')

            st.download_button(
                label="📊 Télécharger les scores",
                data=csv_data,
                file_name="scores_risque_clients.csv",
                mime="text/csv"
            )
# ============================================================
# 4️⃣ MODÈLES PRÉDICTIFS AVANCÉS - VERSION AVEC SÉLECTION DE VARIABLES
# ============================================================
elif page == " 🧮 Modèles Prédictifs":
    st.header("🎯 Modélisation Prédictive Avancée")

    if not st.session_state.data_loaded:
        st.warning("⚠️ Veuillez d'abord charger des données")
        st.stop()

    # Utiliser les données préparées si disponibles, sinon les données brutes
    if st.session_state.df_final is not None:
        df_final = st.session_state.df_final
    else:
        df_final = st.session_state.dataframe.copy()

    # Initialisation du moteur prédictif avancé
    if st.session_state.predictive_engine is None:
        try:
            try:
                from modules.predictive_engine import AdvancedPredictiveEngine
            except ImportError:
                sys.path.append(os.path.join(BASE_DIR, "modules"))
                from predictive_engine import AdvancedPredictiveEngine

            st.session_state.predictive_engine = AdvancedPredictiveEngine()
            st.success("✅ Moteur prédictif avancé initialisé")
        except ImportError as e:
            st.error(f"❌ Impossible d'importer le module predictive_engine: {e}")
            st.info("""
            **Assurez-vous que:**
            1. Le fichier `predictive_engine.py` est dans le dossier `modules/`
            2. Les dépendances sont installées:
            ```bash
            pip install scikit-learn statsmodels prophet xgboost catboost plotly
            ```
            """)
            st.stop()

    predictive_engine = st.session_state.predictive_engine

    # Onglets pour différents types de modélisation
    tab1, tab2, tab3 = st.tabs([
        "🔍 Analyse Exploratoire",
        "🎯 Classification",
        "📈 Régression"
    ])

    # ============================================================
    # TAB 1: ANALYSE EXPLORATOIRE
    # ============================================================
    with tab1:
        st.subheader("🔍 Analyse Exploratoire des Données")

        # Sélection de la variable cible pour l'exploration
        col_detect1, col_detect2 = st.columns(2)

        with col_detect1:
            all_columns = list(df_final.columns)
            target_col = st.selectbox(
                "Variable cible (target)",
                options=["Aucune"] + all_columns,
                help="Sélectionnez la variable à analyser",
                key="target_col_exploratory"
            )

        with col_detect2:
            if target_col != "Aucune" and st.button("🔍 Analyser la variable", type="primary"):
                with st.spinner("Analyse en cours..."):
                    y_series = df_final[target_col]
                    n_unique = y_series.nunique()

                    if n_unique == 2:
                        st.success("✅ Classification binaire détectée")
                        st.info(f"Distribution: {y_series.value_counts().to_dict()}")

                        # Graphique de distribution
                        fig = px.pie(
                            values=y_series.value_counts().values,
                            names=y_series.value_counts().index,
                            title=f"Distribution de {target_col}"
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    elif 2 < n_unique <= 10:
                        st.success("✅ Classification multi-classes détectée")
                        st.info(f"{n_unique} classes détectées")

                        # Graphique en barres
                        fig = px.bar(
                            x=y_series.value_counts().index,
                            y=y_series.value_counts().values,
                            title=f"Distribution de {target_col}",
                            labels={'x': 'Classe', 'y': 'Nombre'}
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    elif pd.api.types.is_numeric_dtype(y_series):
                        st.success("✅ Variable numérique continue")

                        # Statistiques
                        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                        with col_stat1:
                            st.metric("Moyenne", f"{y_series.mean():.2f}")
                        with col_stat2:
                            st.metric("Écart-type", f"{y_series.std():.2f}")
                        with col_stat3:
                            st.metric("Min", f"{y_series.min():.2f}")
                        with col_stat4:
                            st.metric("Max", f"{y_series.max():.2f}")

                        # Histogramme
                        fig = px.histogram(
                            df_final,
                            x=target_col,
                            title=f"Distribution de {target_col}",
                            nbins=50
                        )
                        st.plotly_chart(fig, use_container_width=True)

        # Analyse des corrélations
        st.subheader("📊 Analyse des Corrélations")

        # Sélection des variables pour l'analyse de corrélation
        numeric_cols = df_final.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) > 1:
            selected_corr_vars = st.multiselect(
                "Sélectionnez les variables pour l'analyse de corrélation:",
                options=numeric_cols,
                default=numeric_cols[:min(10, len(numeric_cols))],
                help="Sélectionnez les variables numériques à analyser"
            )

            if len(selected_corr_vars) > 1:
                if st.button("🔗 Calculer les corrélations", type="primary"):
                    # Calculer la matrice de corrélation
                    corr_matrix = df_final[selected_corr_vars].corr()

                    # Heatmap de corrélation
                    fig = px.imshow(
                        corr_matrix,
                        title="Matrice de Corrélation",
                        color_continuous_scale='RdBu',
                        zmin=-1,
                        zmax=1
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Top des corrélations
                    st.markdown("**Top 10 des corrélations les plus fortes:**")
                    correlations = corr_matrix.unstack().sort_values(ascending=False)
                    # Enlever les auto-corrélations (valeur 1)
                    correlations = correlations[correlations != 1]

                    top_correlations = correlations.head(10)
                    for idx, value in top_correlations.items():
                        var1, var2 = idx
                        st.info(f"{var1} ↔ {var2}: **{value:.3f}**")
        else:
            st.warning("⚠️ Pas assez de colonnes numériques pour l'analyse de corrélation")

    # ============================================================
    # IMPORTATIONS NÉCESSAIRES
    # ============================================================

    import pandas as pd
    import numpy as np
    import streamlit as st
    import plotly.express as px
    import plotly.graph_objects as go
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.feature_selection import VarianceThreshold
    import traceback
    from datetime import datetime


    # ============================================================
    # FONCTION UTILITAIRE AMÉLIORÉE POUR PRÉPARER LES DONNÉES
    # ============================================================

    def prepare_data_for_ml(df, target_col, selected_features, test_size=0.2, handle_missing="imputer_median",
                            scale_data=True, task_type="classification"):
        """
        Prépare les données pour le machine learning avec gestion robuste des types de données
        """
        try:
            # 1. VALIDATION DES ENTRÉES
            if target_col not in df.columns:
                raise ValueError(f"La variable cible '{target_col}' n'existe pas dans le DataFrame")

            missing_features = [f for f in selected_features if f not in df.columns]
            if missing_features:
                selected_features = [f for f in selected_features if f in df.columns]
                st.warning(f"⚠️ Features manquantes exclues: {missing_features}")

            # 2. SÉLECTION DES DONNÉES
            features_to_use = [f for f in selected_features if f != target_col and f in df.columns]
            if not features_to_use:
                raise ValueError("Aucune feature valide sélectionnée")

            df_selected = df[[target_col] + features_to_use].copy()

            # 3. EXCLUSION DES COLONNES NON UTILISABLES
            # Exclure les dates, timedelta, et types complexes
            date_cols = []
            for col in features_to_use:
                col_dtype = str(df_selected[col].dtype)
                if 'datetime' in col_dtype or 'timedelta' in col_dtype:
                    date_cols.append(col)

            if date_cols:
                df_selected = df_selected.drop(columns=date_cols)
                st.warning(f"⚠️ Colonnes de type date exclues: {date_cols}")

            # Mettre à jour les features après exclusion
            features_to_use = [col for col in df_selected.columns if col != target_col]

            # 4. SÉPARATION FEATURES/TARGET
            X = df_selected[features_to_use]
            y = df_selected[target_col]

            # 5. ENCODAGE DES VARIABLES CATÉGORIELLES
            # Identifier les colonnes catégorielles restantes
            cat_cols = []
            for col in X.columns:
                if X[col].dtype == 'object' or (hasattr(X[col], 'dtype') and X[col].dtype.name == 'category'):
                    unique_vals = X[col].nunique()
                    if unique_vals <= 50:  # Limite pour éviter l'explosion dimensionnelle
                        cat_cols.append(col)
                    else:
                        X = X.drop(columns=[col])
                        st.warning(f"⚠️ Colonne '{col}' exclue (trop de valeurs uniques: {unique_vals})")

            if cat_cols:
                X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True, dtype=int)
            else:
                X_encoded = X.copy()

            # 6. CONVERSION DES BOOLÉENS
            bool_cols = X_encoded.select_dtypes(include=['bool']).columns
            X_encoded[bool_cols] = X_encoded[bool_cols].astype(int)

            # 7. GESTION DES VALEURS MANQUANTES
            numeric_cols = X_encoded.select_dtypes(include=[np.number]).columns

            if handle_missing == "imputer_median" and len(numeric_cols) > 0:
                imputer = SimpleImputer(strategy='median')
                X_encoded[numeric_cols] = imputer.fit_transform(X_encoded[numeric_cols])
            elif handle_missing == "imputer_mean" and len(numeric_cols) > 0:
                imputer = SimpleImputer(strategy='mean')
                X_encoded[numeric_cols] = imputer.fit_transform(X_encoded[numeric_cols])
            elif handle_missing == "drop":
                mask = X_encoded.notna().all(axis=1) & y.notna()
                X_encoded = X_encoded[mask]
                y = y[mask]
            else:
                # Par défaut, imputer avec la médiane
                if len(numeric_cols) > 0:
                    imputer = SimpleImputer(strategy='median')
                    X_encoded[numeric_cols] = imputer.fit_transform(X_encoded[numeric_cols])

            # 8. VÉRIFICATION ET NETTOYAGE FINAL
            # Supprimer les colonnes avec variance nulle
            if len(X_encoded.columns) > 0:
                selector = VarianceThreshold(threshold=0.01)
                try:
                    X_encoded_arr = selector.fit_transform(X_encoded)
                    kept_indices = selector.get_support(indices=True)
                    X_encoded = X_encoded.iloc[:, kept_indices]
                    if len(kept_indices) < X_encoded.shape[1]:
                        st.info(f"📉 {X_encoded.shape[1] - len(kept_indices)} colonnes à faible variance supprimées")
                except Exception as e:
                    st.warning(f"⚠️ Impossible d'appliquer le filtre de variance: {str(e)}")

            # 9. NORMALISATION
            if scale_data and len(X_encoded.columns) > 0:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_encoded)
                X_encoded = pd.DataFrame(
                    X_scaled,
                    columns=X_encoded.columns,
                    index=X_encoded.index
                )

            # 10. SPLIT DES DONNÉES AVEC STRATIFICATION INTELLIGENTE
            if len(X_encoded) == 0:
                raise ValueError("Aucune donnée valide après prétraitement")

            if task_type == "classification":
                # Vérifier si la stratification est possible
                y_unique = y.nunique()
                if y_unique >= 2:
                    class_counts = y.value_counts()
                    min_class_size = class_counts.min()

                    # Stratification seulement si toutes les classes ont au moins 2 échantillons
                    if min_class_size >= 2 and y_unique <= 20:  # Limite pour éviter trop de classes
                        try:
                            X_train, X_test, y_train, y_test = train_test_split(
                                X_encoded, y, test_size=test_size,
                                random_state=42, stratify=y
                            )
                            st.success("✅ Stratification appliquée avec succès")
                        except Exception as e:
                            st.warning(f"⚠️ Stratification échouée: {str(e)}. Utilisation sans stratification.")
                            X_train, X_test, y_train, y_test = train_test_split(
                                X_encoded, y, test_size=test_size,
                                random_state=42
                            )
                    else:
                        X_train, X_test, y_train, y_test = train_test_split(
                            X_encoded, y, test_size=test_size,
                            random_state=42
                        )
                        if min_class_size < 2:
                            st.warning("⚠️ Stratification désactivée (certaines classes ont < 2 échantillons)")
                else:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_encoded, y, test_size=test_size,
                        random_state=42
                    )
            else:  # Régression
                X_train, X_test, y_train, y_test = train_test_split(
                    X_encoded, y, test_size=test_size,
                    random_state=42
                )

            # Vérification finale
            if X_train.shape[1] == 0:
                raise ValueError("Aucune feature valide après prétraitement")

            return X_train, X_test, y_train, y_test, X_encoded.columns.tolist()

        except Exception as e:
            st.error(f"❌ Erreur dans prepare_data_for_ml: {str(e)}")
            raise


    # ============================================================
    # FONCTION POUR FILTRER LES COLONNES UTILISABLES
    # ============================================================

    def get_usable_columns(df, exclude_dates=True, max_categories=20):
        """
        Retourne les colonnes utilisables pour le ML
        """
        usable_cols = []

        for col in df.columns:
            # Exclure les dates si demandé
            if exclude_dates:
                col_dtype = str(df[col].dtype)
                if 'datetime' in col_dtype or 'timedelta' in col_dtype:
                    continue

            # Pour les colonnes catégorielles, vérifier le nombre de valeurs uniques
            if df[col].dtype == 'object' or (hasattr(df[col], 'dtype') and df[col].dtype.name == 'category'):
                unique_count = df[col].nunique()
                if unique_count <= max_categories:
                    usable_cols.append(col)
                else:
                    # Vérifier si c'est vraiment numérique mais stocké comme object
                    try:
                        pd.to_numeric(df[col], errors='raise')
                        usable_cols.append(col)
                    except:
                        continue
            else:
                usable_cols.append(col)

        return usable_cols


    # ============================================================
    # TAB 2: CLASSIFICATION OPTIMISÉE
    # ============================================================
    with tab2:
        st.subheader("🎯 Classification Supervisée")

        # Initialisation des variables
        target_col = None
        selected_features = []

        # Section 1: Sélection des variables
        st.markdown("### 🔧 Configuration du problème")

        col_sel1, col_sel2 = st.columns(2)

        with col_sel1:
            # Obtenir les colonnes utilisables
            usable_cols = get_usable_columns(df_final, exclude_dates=True, max_categories=20)

            # Variables cibles potentielles (catégorielles avec nombre raisonnable de classes)
            target_candidates = []
            for col in usable_cols:
                unique_count = df_final[col].nunique()
                if 2 <= unique_count <= 20:
                    target_candidates.append((col, unique_count))

            # Trier par nombre de classes
            target_candidates.sort(key=lambda x: x[1])
            target_options = [col for col, _ in target_candidates]

            if not target_options:
                st.error("❌ Aucune variable cible catégorielle valide trouvée")
                st.stop()

            target_col = st.selectbox(
                "🎯 Variable cible:",
                options=target_options,
                help="Variable catégorielle à prédire (2-20 classes)",
                key="target_col_class"
            )

            # Afficher les statistiques de la cible
            if target_col:
                col_stats1, col_stats2, col_stats3 = st.columns(3)
                with col_stats1:
                    st.metric("Classes", df_final[target_col].nunique())
                with col_stats2:
                    missing = df_final[target_col].isna().sum()
                    st.metric("Manquantes", missing)
                with col_stats3:
                    total = len(df_final)
                    st.metric("Total", total)

                # Distribution des classes
                class_dist = df_final[target_col].value_counts().head(10)
                if len(class_dist) > 0:
                    fig = px.bar(
                        x=class_dist.index.astype(str),
                        y=class_dist.values,
                        title=f"Top 10 classes de '{target_col}'",
                        labels={'x': 'Classe', 'y': 'Nombre'},
                        color=class_dist.values,
                        color_continuous_scale='Blues'
                    )
                    st.plotly_chart(fig, use_container_width=True)

        with col_sel2:
            # Variables prédictives (exclure la cible)
            if target_col:
                feature_candidates = [col for col in usable_cols if col != target_col]
            else:
                feature_candidates = usable_cols

            st.markdown("**📊 Variables prédictives:**")

            # Option de sélection rapide
            col_opt1, col_opt2 = st.columns(2)
            with col_opt1:
                use_all = st.checkbox("Toutes les variables", value=True, key="use_all_class")
            with col_opt2:
                use_numeric_only = st.checkbox("Numériques seulement", value=False, key="numeric_only_class")

            if use_numeric_only:
                feature_candidates = [col for col in feature_candidates
                                      if pd.api.types.is_numeric_dtype(df_final[col])]

            if not use_all:
                selected_features = st.multiselect(
                    "Sélectionnez les features:",
                    options=feature_candidates,
                    default=feature_candidates[:min(10, len(feature_candidates))],
                    help="Choisissez les variables pour prédire la cible",
                    key="features_class"
                )
            else:
                selected_features = feature_candidates

            # Aperçu des features sélectionnées
            if selected_features:
                st.info(f"✅ {len(selected_features)} features sélectionnées")

                # Types de données
                numeric_count = sum(1 for f in selected_features
                                    if pd.api.types.is_numeric_dtype(df_final[f]))
                cat_count = len(selected_features) - numeric_count

                col_type1, col_type2, col_type3 = st.columns(3)
                with col_type1:
                    st.metric("Numériques", numeric_count)
                with col_type2:
                    st.metric("Catégorielles", cat_count)
                with col_type3:
                    # Estimation de la dimension après one-hot encoding
                    estimated_dim = numeric_count
                    for f in selected_features:
                        if not pd.api.types.is_numeric_dtype(df_final[f]):
                            unique_count = min(df_final[f].nunique(), 10)
                            estimated_dim += unique_count - 1
                    st.metric("Dim. estimée", estimated_dim)

        # Section 2: Configuration avancée
        st.markdown("### ⚙️ Paramètres avancés")

        col_adv1, col_adv2, col_adv3 = st.columns(3)

        with col_adv1:
            model_type = st.selectbox(
                "🤖 Algorithme:",
                options=["random_forest", "xgboost", "logistic_regression", "gradient_boosting"],
                help="Sélectionnez l'algorithme de classification",
                key="model_type_class"
            )

            test_size = st.slider(
                "📊 Taille test (%):",
                min_value=10, max_value=40, value=20,
                help="Pourcentage pour la validation",
                key="test_size_class"
            ) / 100

        with col_adv2:
            handle_missing = st.selectbox(
                "🔄 Valeurs manquantes:",
                options=["imputer_median", "imputer_mean", "drop"],
                help="Stratégie de gestion des NaN",
                key="handle_missing_class"
            )

            scale_features = st.selectbox(
                "📏 Normalisation:",
                options=["standard", "minmax", "aucune"],
                help="Méthode de normalisation",
                key="scale_class"
            )

        with col_adv3:
            optimize_params = st.checkbox("🔍 Optimisation hyperparamètres", value=True, key="optimize_class")
            cv_folds = st.slider("🌀 Validation croisée:", 2, 10, 5, key="cv_class")
            balance_classes = st.checkbox("⚖️ Équilibrer les classes", value=False, key="balance_class")

        # Section 3: Entraînement
        st.markdown("### 🚀 Entraînement du modèle")

        if not target_col or not selected_features:
            st.warning("⚠️ Sélectionnez une variable cible et des features")
        else:
            # Résumé de la configuration
            with st.expander("📄 Résumé de la configuration", expanded=True):
                col_sum1, col_sum2, col_sum3 = st.columns(3)
                with col_sum1:
                    st.metric("Cible", target_col)
                    st.metric("Classes", df_final[target_col].nunique())
                with col_sum2:
                    st.metric("Features", len(selected_features))
                    st.metric("Échantillons", len(df_final))
                with col_sum3:
                    st.metric("Test size", f"{test_size * 100:.0f}%")
                    st.metric("Modèle", model_type)

            col_train1, col_train2, col_train3 = st.columns([1, 2, 1])
            with col_train2:
                train_button = st.button(
                    "🎯 Démarrer l'entraînement",
                    type="primary",
                    use_container_width=True,
                    key="train_button_class"
                )

        # PROCESSUS D'ENTRAÎNEMENT
        if train_button and target_col and selected_features:
            try:
                # Préparation des données
                with st.spinner("🔄 Préparation des données en cours..."):
                    # Adapter scale_data pour la nouvelle fonction
                    scale_bool = scale_features != "aucune"

                    X_train, X_test, y_train, y_test, feature_names = prepare_data_for_ml(
                        df_final,
                        target_col,
                        selected_features,
                        test_size=test_size,
                        handle_missing=handle_missing,
                        scale_data=scale_bool,
                        task_type="classification"
                    )

                    # Vérifications finales
                    if X_train.shape[1] == 0:
                        st.error("❌ Aucune feature valide après prétraitement")
                        st.stop()

                    if len(y_train) == 0:
                        st.error("❌ Aucun échantillon d'entraînement valide")
                        st.stop()

                    # Afficher les informations
                    st.success("✅ Données préparées avec succès")

                    info_col1, info_col2, info_col3, info_col4 = st.columns(4)
                    with info_col1:
                        st.metric("Train", f"{len(X_train):,}")
                    with info_col2:
                        st.metric("Test", f"{len(X_test):,}")
                    with info_col3:
                        st.metric("Features", X_train.shape[1])
                    with info_col4:
                        class_balance = y_train.value_counts().min() / y_train.value_counts().max()
                        st.metric("Balance", f"{class_balance:.2%}")

                # ENTRAÎNEMENT DU MODÈLE
                with st.spinner(f"🎯 Entraînement du modèle {model_type}..."):
                    # Import des modèles de classification
                    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
                    from sklearn.linear_model import LogisticRegression
                    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                                 f1_score, confusion_matrix, classification_report,
                                                 roc_auc_score, roc_curve)

                    # Initialisation du modèle selon le type
                    if model_type == "random_forest":
                        model = RandomForestClassifier(
                            n_estimators=100,
                            max_depth=10,
                            min_samples_split=5,
                            min_samples_leaf=2,
                            random_state=42,
                            n_jobs=-1,
                            class_weight='balanced' if balance_classes else None
                        )

                    elif model_type == "xgboost":
                        try:
                            from xgboost import XGBClassifier

                            model = XGBClassifier(
                                n_estimators=100,
                                max_depth=6,
                                learning_rate=0.1,
                                random_state=42,
                                use_label_encoder=False,
                                eval_metric='logloss'
                            )
                        except ImportError:
                            st.warning("XGBoost non installé, utilisation de Random Forest")
                            model = RandomForestClassifier(n_estimators=100, random_state=42)

                    elif model_type == "gradient_boosting":
                        model = GradientBoostingClassifier(
                            n_estimators=100,
                            learning_rate=0.1,
                            max_depth=3,
                            random_state=42
                        )

                    elif model_type == "logistic_regression":
                        model = LogisticRegression(
                            max_iter=1000,
                            random_state=42,
                            class_weight='balanced' if balance_classes else None,
                            solver='lbfgs'
                        )

                    else:
                        model = RandomForestClassifier(n_estimators=100, random_state=42)

                    # Optimisation des hyperparamètres si demandée
                    if optimize_params:
                        with st.spinner("🔍 Optimisation des hyperparamètres..."):
                            from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

                            if model_type == "random_forest":
                                param_grid = {
                                    'n_estimators': [50, 100, 200],
                                    'max_depth': [5, 10, 15, None],
                                    'min_samples_split': [2, 5, 10],
                                    'min_samples_leaf': [1, 2, 4]
                                }
                                search = GridSearchCV(
                                    model, param_grid,
                                    cv=min(cv_folds, 5),  # Limiter à 5 folds pour la performance
                                    scoring='f1_weighted',
                                    n_jobs=-1,
                                    verbose=0
                                )

                            elif model_type == "xgboost":
                                param_grid = {
                                    'n_estimators': [50, 100, 150],
                                    'max_depth': [3, 6, 9],
                                    'learning_rate': [0.01, 0.1, 0.3],
                                    'subsample': [0.8, 1.0]
                                }
                                search = RandomizedSearchCV(
                                    model, param_grid,
                                    n_iter=10,
                                    cv=min(cv_folds, 5),
                                    scoring='f1_weighted',
                                    n_jobs=-1,
                                    random_state=42,
                                    verbose=0
                                )

                            elif model_type == "logistic_regression":
                                param_grid = {
                                    'C': [0.01, 0.1, 1, 10, 100],
                                    'penalty': ['l2'],
                                    'solver': ['lbfgs', 'liblinear']
                                }
                                search = GridSearchCV(
                                    model, param_grid,
                                    cv=min(cv_folds, 5),
                                    scoring='f1_weighted',
                                    n_jobs=-1,
                                    verbose=0
                                )

                            else:
                                # Pas d'optimisation pour les autres modèles
                                search = None

                            if search is not None:
                                search.fit(X_train, y_train)
                                model = search.best_estimator_
                                st.success(f"✅ Meilleurs paramètres: {search.best_params_}")

                    # Entraînement du modèle
                    model.fit(X_train, y_train)

                    # Prédictions
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

                    # Sauvegarde dans session state
                    st.session_state.classification_model = {
                        'model': model,
                        'features': feature_names,
                        'target': target_col,
                        'model_type': model_type,
                        'X_train': X_train,
                        'X_test': X_test,
                        'y_train': y_train,
                        'y_test': y_test,
                        'y_pred': y_pred,
                        'y_pred_proba': y_pred_proba
                    }

                st.success("✅ Modèle entraîné avec succès !")

                # ============================================
                # AFFICHAGE DES RÉSULTATS
                # ============================================

                # 1. Métriques de performance
                st.subheader("📊 Performance du modèle")

                # Calcul des métriques
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

                # Affichage des métriques
                col_met1, col_met2, col_met3, col_met4 = st.columns(4)
                with col_met1:
                    st.metric("Accuracy", f"{accuracy:.3f}")
                with col_met2:
                    st.metric("Precision", f"{precision:.3f}")
                with col_met3:
                    st.metric("Recall", f"{recall:.3f}")
                with col_met4:
                    st.metric("F1-Score", f"{f1:.3f}")

                # 2. Matrice de confusion
                st.subheader("📄 Matrice de confusion")

                cm = confusion_matrix(y_test, y_pred)
                class_labels = sorted(y_test.unique())

                # Création de la heatmap
                fig_cm = px.imshow(
                    cm,
                    labels=dict(x="Prédit", y="Réel", color="Count"),
                    x=[f'Prédit {label}' for label in class_labels],
                    y=[f'Réel {label}' for label in class_labels],
                    text_auto=True,
                    color_continuous_scale='Blues',
                    aspect="auto"
                )
                fig_cm.update_layout(title="Matrice de Confusion")
                st.plotly_chart(fig_cm, use_container_width=True)

                # 3. Rapport de classification détaillé
                with st.expander("📄 Rapport de classification détaillé"):
                    report = classification_report(y_test, y_pred, output_dict=True)
                    report_df = pd.DataFrame(report).transpose()
                    st.dataframe(report_df.style.format("{:.3f}").background_gradient(cmap='Blues'),
                                 use_container_width=True)

                # 4. Importance des features
                if hasattr(model, 'feature_importances_'):
                    st.subheader("🎯 Importance des features")

                    # Création du dataframe d'importance
                    importance_df = pd.DataFrame({
                        'Feature': feature_names[:len(model.feature_importances_)],
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=False)

                    # Graphique des top 15 features
                    fig_importance = px.bar(
                        importance_df.head(15),
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title='Top 15 des features les plus importantes',
                        color='Importance',
                        color_continuous_scale='Viridis'
                    )
                    fig_importance.update_layout(yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig_importance, use_container_width=True)

                    # Tableau complet des importances
                    with st.expander("📄 Voir toutes les importances"):
                        st.dataframe(importance_df, use_container_width=True)


                # 6. Prédictions détaillées
                st.subheader("🔮 Prédictions sur le jeu de test")

                if st.button("📊 Afficher les prédictions détaillées", key="show_preds_class"):
                    results_df = pd.DataFrame({
                        'Vraie_valeur': y_test.values,
                        'Prédiction': y_pred
                    })

                    if y_pred_proba is not None:
                        for i, class_label in enumerate(sorted(y_test.unique())):
                            results_df[f'Probabilité_Classe_{class_label}'] = y_pred_proba[:, i]

                    results_df['Correct'] = results_df['Vraie_valeur'] == results_df['Prédiction']

                    # Affichage avec coloration
                    st.dataframe(
                        results_df.head(50).style.apply(
                            lambda x: ['background-color: #d4edda' if x['Correct'] else 'background-color: #f8d7da' for
                                       _ in x],
                            axis=1
                        ),
                        use_container_width=True
                    )

                    # Téléchargement
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Télécharger toutes les prédictions",
                        data=csv,
                        file_name=f"predictions_classification_{target_col}_{model_type}.csv",
                        mime="text/csv",
                        key="download_preds_class"
                    )

                # 7. Export du modèle
                st.markdown("---")
                st.subheader("💾 Export du modèle")

                col_exp1, col_exp2 = st.columns(2)

                with col_exp1:
                    if st.button("💾 Sauvegarder le modèle", key="save_model_class"):
                        import joblib
                        import os

                        os.makedirs("models", exist_ok=True)
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        model_path = f"models/classification_{target_col}_{model_type}_{timestamp}.pkl"

                        model_data = {
                            'model': model,
                            'features': feature_names,
                            'target': target_col,
                            'model_type': model_type,
                            'accuracy': accuracy,
                            'metadata': {
                                'training_date': timestamp,
                                'n_samples_train': len(X_train),
                                'n_samples_test': len(X_test),
                                'n_features': len(feature_names),
                                'test_size': test_size
                            }
                        }

                        joblib.dump(model_data, model_path)
                        st.success(f"✅ Modèle sauvegardé: `{model_path}`")

                with col_exp2:
                    # Code pour reproduire le modèle
                    with st.expander("📝 Code de reproduction"):
                        st.code(f"""
    # Code pour reproduire le modèle {model_type}
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    # Charger les données
    # df = pd.read_csv('votre_fichier.csv')

    # Sélectionner les mêmes variables
    X = df[{selected_features}]
    y = df['{target_col}']

    # Encodage one-hot pour les variables catégorielles
    X = pd.get_dummies(X, drop_first=True)

    # Imputation des valeurs manquantes
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X)

    # Normalisation
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split des données
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size={test_size}, random_state=42
    )

    # Entraîner le modèle
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Évaluation
    y_pred = model.predict(X_test)
    accuracy = (y_pred == y_test).mean()
    print(f"Accuracy: {{accuracy:.3f}}")
                        """)

            except Exception as e:
                st.error(f"❌ Erreur lors de l'entraînement: {str(e)}")
                st.code(traceback.format_exc())

    # ============================================================
    # TAB 3: RÉGRESSION OPTIMISÉE
    # ============================================================
    with tab3:
        st.subheader("📈 Régression Supervisée")

        # Initialisation des variables
        target_col_reg = None
        selected_features_reg = []

        # Section 1: Sélection des variables
        st.markdown("### 🔧 Configuration du problème")

        col_sel1, col_sel2 = st.columns(2)

        with col_sel1:
            # Colonnes utilisables pour la régression
            usable_cols_reg = get_usable_columns(df_final, exclude_dates=True)

            # Variables cibles numériques
            numeric_targets = []
            for col in usable_cols_reg:
                try:
                    if pd.api.types.is_numeric_dtype(df_final[col]):
                        # Vérifier qu'il y a suffisamment de valeurs uniques
                        if df_final[col].nunique() > 5:  # Éviter les variables quasi-catégorielles
                            numeric_targets.append(col)
                except:
                    continue

            if not numeric_targets:
                st.error("❌ Aucune variable numérique valide pour la régression")
                st.stop()

            target_col_reg = st.selectbox(
                "🎯 Variable cible:",
                options=numeric_targets,
                help="Variable numérique continue à prédire",
                key="target_col_reg"
            )

            # Statistiques de la cible
            if target_col_reg:
                target_stats = df_final[target_col_reg].describe()

                col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                with col_stat1:
                    st.metric("Moyenne", f"{target_stats['mean']:.2f}")
                with col_stat2:
                    st.metric("Std", f"{target_stats['std']:.2f}")
                with col_stat3:
                    st.metric("Min", f"{target_stats['min']:.2f}")
                with col_stat4:
                    st.metric("Max", f"{target_stats['max']:.2f}")

                # Distribution
                fig = px.histogram(df_final, x=target_col_reg,
                                   title=f"Distribution de '{target_col_reg}'",
                                   nbins=50,
                                   color_discrete_sequence=['#636EFA'])
                st.plotly_chart(fig, use_container_width=True)

        with col_sel2:
            # Variables prédictives
            if target_col_reg:
                feature_candidates = [col for col in usable_cols_reg if col != target_col_reg]
            else:
                feature_candidates = usable_cols_reg

            st.markdown("**📊 Variables prédictives:**")

            # Options de filtrage
            col_filt1, col_filt2 = st.columns(2)
            with col_filt1:
                use_all_reg = st.checkbox("Toutes les variables", value=True, key="use_all_reg")
            with col_filt2:
                filter_correlation = st.checkbox("Filtrer par corrélation", value=False, key="filter_corr")

            if not use_all_reg:
                selected_features_reg = st.multiselect(
                    "Sélectionnez les features:",
                    options=feature_candidates,
                    default=feature_candidates[:min(10, len(feature_candidates))],
                    help="Choisissez les prédicteurs",
                    key="features_reg"
                )
            else:
                selected_features_reg = feature_candidates

            # Filtrage par corrélation si activé
            if filter_correlation and target_col_reg and selected_features_reg:
                numeric_features = [f for f in selected_features_reg
                                    if pd.api.types.is_numeric_dtype(df_final[f])]
                if numeric_features:
                    corr_values = []
                    for feat in numeric_features:
                        try:
                            corr = df_final[[target_col_reg, feat]].corr().iloc[0, 1]
                            if not pd.isna(corr):
                                corr_values.append((feat, abs(corr)))
                        except:
                            pass

                    # Garder les features les plus corrélées
                    if corr_values:
                        corr_values.sort(key=lambda x: x[1], reverse=True)
                        top_features = [f[0] for f in corr_values[:20]]  # Top 20
                        selected_features_reg = top_features
                        st.success(f"✅ {len(selected_features_reg)} features sélectionnées par corrélation")

        # Section 2: Configuration
        st.markdown("### ⚙️ Paramètres avancés")

        col_conf1, col_conf2, col_conf3 = st.columns(3)

        with col_conf1:
            model_type_reg = st.selectbox(
                "🤖 Algorithme:",
                options=["random_forest", "xgboost", "linear_regression", "gradient_boosting"],
                help="Sélectionnez l'algorithme de régression",
                key="model_type_reg"
            )

            test_size_reg = st.slider(
                "📊 Taille test (%):",
                min_value=10, max_value=40, value=20, step=5,
                key="test_size_reg"
            ) / 100

        with col_conf2:
            handle_outliers = st.selectbox(
                "📊 Outliers:",
                options=["garder", "supprimer", "winsorize"],
                help="Traitement des valeurs extrêmes",
                key="outliers_reg"
            )

            scale_method = st.selectbox(
                "📏 Normalisation:",
                options=["standard", "minmax", "robust", "aucune"],
                help="Méthode de scaling",
                key="scale_reg"
            )

        with col_conf3:
            optimize_reg = st.checkbox("🔍 Optimisation", value=True, key="optimize_reg")
            cv_reg = st.slider("🌀 Validation croisée:", 2, 10, 5, key="cv_reg")
            remove_collinear = st.checkbox("📉 Supprimer colinéarité", value=True, key="collinear_reg")

        # Bouton d'entraînement
        train_button_reg = st.button("🚀 Entraîner modèle", type="primary", key="train_button_reg")

        if train_button_reg:
            if not target_col_reg or not selected_features_reg:
                st.error("⚠️ Sélectionnez une variable cible et des features")
                st.stop()

            try:
                with st.spinner("🔄 Préparation des données en cours..."):
                    # Copie des données pour traitement
                    df_processed = df_final[[target_col_reg] + selected_features_reg].copy()

                    # Traitement des outliers
                    if handle_outliers == "supprimer":
                        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
                        initial_count = len(df_processed)
                        for col in numeric_cols:
                            if col != target_col_reg:  # Ne pas traiter la cible
                                Q1 = df_processed[col].quantile(0.25)
                                Q3 = df_processed[col].quantile(0.75)
                                IQR = Q3 - Q1
                                if IQR > 0:  # Éviter division par zéro
                                    lower = Q1 - 1.5 * IQR
                                    upper = Q3 + 1.5 * IQR
                                    df_processed = df_processed[
                                        (df_processed[col] >= lower) & (df_processed[col] <= upper)
                                        ]

                        removed_count = initial_count - len(df_processed)
                        if removed_count > 0:
                            st.info(
                                f"📊 {removed_count} outliers supprimés ({removed_count / initial_count * 100:.1f}%)")

                    # Préparation avec la fonction utilitaire
                    scale_bool = scale_method != "aucune"

                    X_train, X_test, y_train, y_test, feature_names = prepare_data_for_ml(
                        df_processed,
                        target_col_reg,
                        selected_features_reg,
                        test_size=test_size_reg,
                        handle_missing="imputer_median",
                        scale_data=scale_bool,
                        task_type="regression"
                    )

                    # Afficher les informations
                    st.success("✅ Données préparées avec succès")

                    col_info1, col_info2, col_info3, col_info4 = st.columns(4)
                    with col_info1:
                        st.metric("Train", f"{len(X_train):,}")
                    with col_info2:
                        st.metric("Test", f"{len(X_test):,}")
                    with col_info3:
                        st.metric("Features", X_train.shape[1])
                    with col_info4:
                        # Calcul de la corrélation moyenne
                        try:
                            corr_matrix = pd.concat([pd.DataFrame(X_train), pd.Series(y_train, name=target_col_reg)],
                                                    axis=1).corr()
                            avg_corr = corr_matrix[target_col_reg].abs().mean()
                            st.metric("Corr. moyenne", f"{avg_corr:.3f}")
                        except:
                            st.metric("Corr. moyenne", "N/A")

                # ENTRAÎNEMENT DU MODÈLE DE RÉGRESSION
                with st.spinner(f"📈 Entraînement du modèle {model_type_reg}..."):
                    # Import des modèles de régression
                    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
                    from sklearn.linear_model import LinearRegression
                    from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                                                 r2_score, mean_absolute_percentage_error)

                    # Initialisation du modèle selon le type
                    if model_type_reg == "random_forest":
                        model = RandomForestRegressor(
                            n_estimators=100,
                            max_depth=10,
                            min_samples_split=5,
                            random_state=42,
                            n_jobs=-1
                        )

                    elif model_type_reg == "xgboost":
                        try:
                            from xgboost import XGBRegressor

                            model = XGBRegressor(
                                n_estimators=100,
                                max_depth=6,
                                learning_rate=0.1,
                                random_state=42
                            )
                        except ImportError:
                            st.warning("XGBoost non installé, utilisation de Random Forest")
                            model = RandomForestRegressor(n_estimators=100, random_state=42)

                    elif model_type_reg == "gradient_boosting":
                        model = GradientBoostingRegressor(
                            n_estimators=100,
                            learning_rate=0.1,
                            max_depth=3,
                            random_state=42
                        )

                    elif model_type_reg == "linear_regression":
                        model = LinearRegression()

                    else:
                        model = RandomForestRegressor(n_estimators=100, random_state=42)

                    # Optimisation des hyperparamètres si demandée
                    if optimize_reg:
                        with st.spinner("🔍 Optimisation des hyperparamètres..."):
                            from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

                            if model_type_reg == "random_forest":
                                param_grid = {
                                    'n_estimators': [50, 100, 200],
                                    'max_depth': [5, 10, 15, None],
                                    'min_samples_split': [2, 5, 10]
                                }
                                search = GridSearchCV(
                                    model, param_grid,
                                    cv=min(cv_reg, 5),
                                    scoring='r2',
                                    n_jobs=-1,
                                    verbose=0
                                )

                            elif model_type_reg == "xgboost":
                                param_grid = {
                                    'n_estimators': [50, 100, 150],
                                    'max_depth': [3, 6, 9],
                                    'learning_rate': [0.01, 0.1, 0.3]
                                }
                                search = RandomizedSearchCV(
                                    model, param_grid,
                                    n_iter=10,
                                    cv=min(cv_reg, 5),
                                    scoring='r2',
                                    n_jobs=-1,
                                    random_state=42,
                                    verbose=0
                                )

                            elif model_type_reg == "linear_regression":
                                param_grid = {
                                    'fit_intercept': [True, False]
                                }
                                search = GridSearchCV(
                                    model, param_grid,
                                    cv=min(cv_reg, 5),
                                    scoring='r2',
                                    n_jobs=-1,
                                    verbose=0
                                )

                            else:
                                search = None

                            if search is not None:
                                search.fit(X_train, y_train)
                                model = search.best_estimator_
                                st.success(f"✅ Meilleurs paramètres: {search.best_params_}")

                    # Entraînement du modèle
                    model.fit(X_train, y_train)

                    # Prédictions
                    y_pred = model.predict(X_test)

                    # Sauvegarde dans session state
                    st.session_state.regression_model = {
                        'model': model,
                        'features': feature_names,
                        'target': target_col_reg,
                        'model_type': model_type_reg,
                        'X_train': X_train,
                        'X_test': X_test,
                        'y_train': y_train,
                        'y_test': y_test,
                        'y_pred': y_pred
                    }

                st.success("✅ Modèle de régression entraîné avec succès !")

                # ============================================
                # AFFICHAGE DES RÉSULTATS
                # ============================================

                # 1. Métriques de performance
                st.subheader("📊 Performance du modèle")

                # Calcul des métriques
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                try:
                    mape = mean_absolute_percentage_error(y_test, y_pred) * 100
                except:
                    mape = None

                # Affichage des métriques
                col_met1, col_met2, col_met3, col_met4 = st.columns(4)
                with col_met1:
                    st.metric("R² Score", f"{r2:.3f}")
                with col_met2:
                    st.metric("RMSE", f"{rmse:.3f}")
                with col_met3:
                    st.metric("MAE", f"{mae:.3f}")
                with col_met4:
                    if mape is not None:
                        st.metric("MAPE", f"{mape:.1f}%")
                    else:
                        st.metric("MAPE", "N/A")

                # 2. Graphique des prédictions vs vraies valeurs
                st.subheader("📈 Prédictions vs Vraies valeurs")

                # Création du graphique
                fig_scatter = go.Figure()

                # Nuage de points des prédictions
                fig_scatter.add_trace(go.Scatter(
                    x=y_test,
                    y=y_pred,
                    mode='markers',
                    name='Prédictions',
                    marker=dict(
                        size=8,
                        color='blue',
                        opacity=0.6,
                        line=dict(width=1, color='DarkSlateGrey')
                    ),
                    hovertemplate='<b>Vraie valeur</b>: %{x:.2f}<br><b>Prédiction</b>: %{y:.2f}<br><b>Erreur</b>: %{customdata:.2f}<extra></extra>',
                    customdata=np.abs(y_test - y_pred)
                ))

                # Ligne de parfaite prédiction
                min_val = min(y_test.min(), y_pred.min())
                max_val = max(y_test.max(), y_pred.max())
                fig_scatter.add_trace(go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    name='Parfait',
                    line=dict(color='red', width=2, dash='dash')
                ))

                fig_scatter.update_layout(
                    title='Comparaison des prédictions avec les vraies valeurs',
                    xaxis_title='Vraies valeurs',
                    yaxis_title='Prédictions',
                    showlegend=True,
                    width=800,
                    height=600
                )

                st.plotly_chart(fig_scatter, use_container_width=True)

                # 3. Distribution des erreurs
                st.subheader("📊 Distribution des erreurs")

                errors = y_test - y_pred

                col_err1, col_err2 = st.columns(2)

                with col_err1:
                    # Histogramme des erreurs
                    fig_err_hist = px.histogram(
                        x=errors,
                        nbins=50,
                        title="Distribution des erreurs",
                        labels={'x': 'Erreur', 'y': 'Fréquence'},
                        color_discrete_sequence=['#FF6B6B']
                    )
                    fig_err_hist.add_vline(x=0, line_dash="dash", line_color="red")
                    st.plotly_chart(fig_err_hist, use_container_width=True)

                with col_err2:
                    # QQ plot des erreurs
                    from scipy import stats

                    qq_data = stats.probplot(errors, dist="norm")
                    x = qq_data[0][0]
                    y = qq_data[0][1]

                    fig_qq = go.Figure()
                    fig_qq.add_trace(go.Scatter(
                        x=x, y=y,
                        mode='markers',
                        name='Erreurs',
                        marker=dict(color='blue', size=6)
                    ))

                    # Ligne de référence
                    fig_qq.add_trace(go.Scatter(
                        x=[x.min(), x.max()],
                        y=[x.min(), x.max()],
                        mode='lines',
                        name='Normale',
                        line=dict(color='red', dash='dash')
                    ))

                    fig_qq.update_layout(
                        title="QQ Plot des erreurs",
                        xaxis_title="Quantiles théoriques",
                        yaxis_title="Quantiles observés"
                    )
                    st.plotly_chart(fig_qq, use_container_width=True)

                # 4. Importance des features (si disponible)
                if hasattr(model, 'feature_importances_'):
                    st.subheader("🎯 Importance des features")

                    # Création du dataframe d'importance
                    importance_df = pd.DataFrame({
                        'Feature': feature_names[:len(model.feature_importances_)],
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=False)

                    # Graphique des top 15 features
                    fig_importance = px.bar(
                        importance_df.head(15),
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title='Top 15 des features les plus importantes',
                        color='Importance',
                        color_continuous_scale='Viridis'
                    )
                    fig_importance.update_layout(yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig_importance, use_container_width=True)

                # 5. Prédictions détaillées
                st.subheader("🔮 Prédictions détaillées")

                results_df = pd.DataFrame({
                    'Vraie_valeur': y_test.values,
                    'Prédiction': y_pred,
                    'Erreur': errors,
                    'Erreur_abs': np.abs(errors),
                    'Erreur_pourcentage': np.abs(errors / y_test * 100) if (y_test != 0).all() else np.nan
                })

                # Affichage des prédictions
                with st.expander("📄 Voir les prédictions"):
                    st.dataframe(
                        results_df.sort_values('Erreur_abs', ascending=False).head(50)
                        .style.format({
                            'Vraie_valeur': '{:.2f}',
                            'Prédiction': '{:.2f}',
                            'Erreur': '{:.2f}',
                            'Erreur_abs': '{:.2f}',
                            'Erreur_pourcentage': '{:.1f}%'
                        })
                        .background_gradient(subset=['Erreur_abs'], cmap='Reds'),
                        use_container_width=True
                    )

                # Téléchargement des prédictions
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Télécharger toutes les prédictions",
                    data=csv,
                    file_name=f"predictions_regression_{target_col_reg}_{model_type_reg}.csv",
                    mime="text/csv",
                    key="download_preds_reg"
                )

                # 6. Export du modèle
                st.markdown("---")
                st.subheader("💾 Export du modèle")

                col_exp1, col_exp2 = st.columns(2)

                with col_exp1:
                    if st.button("💾 Sauvegarder le modèle", key="save_model_reg"):
                        import joblib
                        import os

                        os.makedirs("models", exist_ok=True)
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        model_path = f"models/regression_{target_col_reg}_{model_type_reg}_{timestamp}.pkl"

                        model_data = {
                            'model': model,
                            'features': feature_names,
                            'target': target_col_reg,
                            'model_type': model_type_reg,
                            'r2_score': r2,
                            'rmse': rmse,
                            'metadata': {
                                'training_date': timestamp,
                                'n_samples_train': len(X_train),
                                'n_samples_test': len(X_test),
                                'n_features': len(feature_names),
                                'test_size': test_size_reg
                            }
                        }

                        joblib.dump(model_data, model_path)
                        st.success(f"✅ Modèle sauvegardé: `{model_path}`")

                with col_exp2:
                    # Code pour reproduire le modèle
                    with st.expander("📝 Code de reproduction"):
                        st.code(f"""
    # Code pour reproduire le modèle {model_type_reg}
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    # Charger les données
    # df = pd.read_csv('votre_fichier.csv')

    # Sélectionner les mêmes variables
    X = df[{selected_features_reg}]
    y = df['{target_col_reg}']

    # Encodage one-hot pour les variables catégorielles
    X = pd.get_dummies(X, drop_first=True)

    # Imputation des valeurs manquantes
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X)

    # Normalisation
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split des données
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size={test_size_reg}, random_state=42
    )

    # Entraîner le modèle
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Évaluation
    y_pred = model.predict(X_test)
    r2 = model.score(X_test, y_test)
    print(f"R² Score: {{r2:.3f}}")
                        """)

            except Exception as e:
                st.error(f"❌ Erreur: {str(e)}")
                st.code(traceback.format_exc())
# ============================================================
# 📄 Rapport Intelligent
# ============================================================
elif page == " 📄 Rapport Intelligent":
    st.header(" Génération de rapport IA")

    if not REPORT_ENGINE_AVAILABLE:
        st.error(" Le module report_engine n'est pas disponible. Installez les dépendances nécessaires.")
        st.info("""
        **Dépendances nécessaires :**
        ```bash
        pip install reportlab python-docx markdown
        ```
        """)
        st.stop()

    if st.session_state.report_engine is None:
        try:
            st.session_state.report_engine = ReportEngine(ai_client=None)
        except Exception as e:
            st.error(f" Impossible d'initialiser le moteur de rapport: {e}")
            st.stop()

    if hasattr(st.session_state, 'using_mateur') and st.session_state.using_mateur:
        st.info(" **Moteur : Mateur AI** - Analyse 100% locale, aucune donnée externe")
    elif st.session_state.openai_client is not None:
        st.info(" **Téléchargez** le rapport en un clique !")
    else:
        st.info("**Moteur : Local** - Génération basique")

    if not st.session_state.data_ready:
        st.warning(" Veuillez charger les données")
        st.stop()

    title = st.text_input(
        "Titre du rapport",
        "Rapport d'analyse – Assurance Automobile"
    )

    audience = st.selectbox(
        "Public cible",
        ["Direction générale", "Direction métier", "Équipe data", "Audit", "Comité de pilotage"]
    )

    sections = st.multiselect(
        "Sections à inclure",
        [
            "executive_summary",
            "data_context",
            "data_quality",
            "statistics",
            "models",
            "scoring",
            "insights",
            "recommendations",
            "limitations",
            "annexes"
        ],
        default=["executive_summary", "scoring", "recommendations"]
    )

    custom_instructions = st.text_area(
        "Instructions personnalisées",
        placeholder="Ex : Insister sur la rentabilité, mentionner les risques réglementaires, proposer un plan d'action concret...",
        height=100
    )

    st.subheader(" Options d'export")
    col_export1, col_export2, col_export3, col_export4 = st.columns(4)
    with col_export1:
        export_md = st.checkbox("Markdown (.md)", value=True)
    with col_export2:
        export_pdf = st.checkbox("PDF (.pdf)", value=True)
    with col_export3:
        export_word = st.checkbox("Word (.docx)", value=True)
    with col_export4:
        export_html = st.checkbox("HTML (.html)", value=True)

    if st.button(" Générer le rapport complet", type="primary"):
        with st.spinner("Génération du rapport en cours..."):
            try:
                data_summary = {
                    "rows": st.session_state.dataframe.shape[0],
                    "columns": st.session_state.dataframe.shape[1],
                    "key_variables": list(st.session_state.dataframe.columns[:10]),
                    "completeness": round((1 - st.session_state.dataframe.isna().sum().sum() /
                                           (st.session_state.dataframe.shape[0] * st.session_state.dataframe.shape[
                                               1])) * 100, 1)
                }

                analysis_summary = "Analyse descriptive + scoring client"

                if st.session_state.scored_clients is not None:
                    scored_clients = st.session_state.scored_clients
                    analysis_summary += f"\n- Clients analysés : {len(scored_clients):,}"
                    analysis_summary += f"\n- Score risque moyen : {scored_clients['score_risque'].mean():.1f}/100"
                    high_risk = (scored_clients['niveau_risque'] == 'Élevé').sum()
                    analysis_summary += f"\n- Clients à risque élevé : {high_risk}"
                    data_summary["high_risk_count"] = high_risk

                insights = None
                if st.session_state.scored_clients is not None:
                    try:
                        from modules.insight_engine import InsightEngine

                        insight_engine = InsightEngine()
                        insights = insight_engine.generate_insights(st.session_state.scored_clients)
                    except ImportError as e:
                        insights = ["Insights sur les risques clients disponibles dans l'onglet 'Insights Avancés'"]
                    except Exception as e:
                        insights = [f"Insights : {str(e)}"]

                report_md = st.session_state.report_engine.generate_report(
                    title=title,
                    audience=audience,
                    sections=sections,
                    data_summary=data_summary,
                    analysis_summary=analysis_summary,
                    model_results=None,
                    insights=insights,
                    custom_instructions=custom_instructions,
                    detail_level=4
                )

                st.session_state.generated_report_md = report_md
                st.success(" Rapport markdown généré avec succès!")

                if export_pdf:
                    with st.spinner("Génération du PDF..."):
                        try:
                            pdf_buffer = st.session_state.report_engine.to_pdf(report_md, title)
                            st.session_state.generated_report_pdf = pdf_buffer.getvalue()
                            st.success(" PDF généré avec succès!")
                        except Exception as e:
                            st.warning(f" PDF non généré: {str(e)}")
                            st.session_state.generated_report_pdf = None

                if export_word:
                    with st.spinner("Génération du document Word..."):
                        try:
                            word_buffer = st.session_state.report_engine.to_word(report_md, title)
                            st.session_state.generated_report_word = word_buffer.getvalue()
                            st.success(" Document Word généré avec succès!")
                        except Exception as e:
                            st.warning(f" Word non généré: {str(e)}")
                            st.session_state.generated_report_word = None

                if export_html:
                    try:
                        html_report = st.session_state.report_engine.to_html(report_md)
                        st.session_state.generated_report_html = html_report
                        st.success(" HTML généré avec succès!")
                    except Exception as e:
                        st.warning(f" HTML non généré: {str(e)}")
                        st.session_state.generated_report_html = None

            except Exception as e:
                st.error(f" Erreur lors de la génération du rapport: {str(e)}")
                st.code(traceback.format_exc())

    if hasattr(st.session_state, 'generated_report_md') and st.session_state.generated_report_md:
        st.markdown("---")
        st.subheader(" Aperçu du rapport")

        with st.expander(" Voir le rapport complet", expanded=False):
            st.markdown(st.session_state.generated_report_md)

        st.subheader(" Téléchargements")
        cols = st.columns(4)

        with cols[0]:
            filename = f"rapport_{datetime.now().strftime('%Y%m%d_%H%M')}"
            st.download_button(
                label=" Markdown",
                data=st.session_state.generated_report_md,
                file_name=f"{filename}.md",
                mime="text/markdown",
                help="Format texte avec mise en forme"
            )

        if hasattr(st.session_state, 'generated_report_pdf') and st.session_state.generated_report_pdf:
            with cols[1]:
                st.download_button(
                    label=" PDF",
                    data=st.session_state.generated_report_pdf,
                    file_name=f"{filename}.pdf",
                    mime="application/pdf",
                    help="Document formaté pour impression"
                )

        if hasattr(st.session_state, 'generated_report_word') and st.session_state.generated_report_word:
            with cols[2]:
                st.download_button(
                    label=" Word",
                    data=st.session_state.generated_report_word,
                    file_name=f"{filename}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    help="Document éditable Microsoft Word"
                )

        if hasattr(st.session_state, 'generated_report_html') and st.session_state.generated_report_html:
            with cols[3]:
                st.download_button(
                    label=" HTML",
                    data=st.session_state.generated_report_html,
                    file_name=f"{filename}.html",
                    mime="text/html",
                    help="Page web autonome"
                )

        st.info("""
        **Formats disponibles :**
        - **Markdown** : Format texte simple, éditable (.Rmd)
        - **PDF** : Document formaté pour impression et partage
        - **Word** : Document éditable (Word)
        - **HTML** : Page web 
        """)

# ============================================================
# 🏢 PAGE "À PROPOS"
# ============================================================
elif page == "🏢 À Propos":
    st.header("🏢 À Propos de LIK Insurance Analyst")

    # Bannière avec logo et message d'accroche
    st.markdown("""
    <div style='text-align: center; padding: 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 15px; margin-bottom: 30px;'>
        <h1 style='margin-bottom: 10px;'>LIK Insurance Analyst</h1>
        <h3 style='font-weight: normal; margin-top: 0;'>Intelligence Artificielle au Service de l'Assurance</h3>
    </div>
    """, unsafe_allow_html=True)

    # Présentation en 2 colonnes
    col_pres1, col_pres2 = st.columns([2, 1])

    with col_pres1:
        st.markdown("""
        ### 🎯 Notre Mission

        **Transformer les données brutes en décisions stratégiques**  

        LIK Insurance Analyst est une plateforme innovante qui combine l'expertise du secteur de l'assurance avec les dernières avancées en intelligence artificielle. Notre solution permet aux assureurs d'optimiser leur gestion des risques, d'améliorer leur rentabilité et d'offrir une expérience client exceptionnelle.
        """)

        st.markdown("""
        ### 🚀 Notre Vision

        **Devenir le partenaire privilégié des assureurs dans leur transformation digitale**  

        Nous aspirons à démocratiser l'accès aux technologies d'IA avancées pour tous les acteurs du secteur de l'assurance, des petites mutuelles aux grands groupes internationaux.
        """)

    with col_pres2:
        # Statistiques clés
        st.markdown("""
        <div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px; text-align: center;'>
            <h3 style='color: #1E3A8A;'>📊 En Chiffres</h3>
            <div style='font-size: 36px; font-weight: bold; color: #667eea;'>100%</div>
            <p>Sécurité des données</p>
            <div style='font-size: 36px; font-weight: bold; color: #667eea;'>+30%</div>
            <p>Précision des modèles</p>
            <div style='font-size: 36px; font-weight: bold; color: #667eea;'>24/7</div>
            <p>Support disponible</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Nos Valeurs
    st.subheader("💎 Nos Valeurs Fondamentales")

    valeurs = [
        {
            "emoji": "🔒",
            "titre": "Confidentialité",
            "description": "Vos données restent 100% locales et sécurisées",
            "couleur": "#e8f4fd"
        },
        {
            "emoji": "⚡",
            "titre": "Innovation",
            "description": "Technologies de pointe en IA et Machine Learning",
            "couleur": "#f0f9ff"
        },
        {
            "emoji": "🎯",
            "titre": "Précision",
            "description": "Analyses scientifiques et résultats fiables",
            "couleur": "#f8f9fa"
        },
        {
            "emoji": "🤝",
            "titre": "Collaboration",
            "description": "Partage d'expertise avec nos partenaires",
            "couleur": "#e8f7ec"
        },
        {
            "emoji": "📈",
            "titre": "Performance",
            "description": "Optimisation continue des résultats",
            "couleur": "#fff3cd"
        },
        {
            "emoji": "🌍",
            "titre": "Accessibilité",
            "description": "Solutions adaptées à tous les budgets",
            "couleur": "#d1ecf1"
        }
    ]

    cols = st.columns(3)
    for i, valeur in enumerate(valeurs):
        with cols[i % 3]:
            st.markdown(f"""
            <div style='background-color: {valeur['couleur']}; padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 15px; min-height: 150px; display: flex; flex-direction: column; justify-content: center;'>
                <div style='font-size: 40px; margin-bottom: 10px;'>{valeur['emoji']}</div>
                <h4 style='margin: 0 0 10px 0; color: #1E3A8A;'>{valeur['titre']}</h4>
                <p style='margin: 0; font-size: 14px;'>{valeur['description']}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Technologies et Méthodologie
    st.subheader("🛠️ Notre Stack Technologique")

    tech_cols = st.columns(3)

    with tech_cols[0]:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; padding: 20px; border-radius: 10px;'>
            <h4 style='margin-top: 0;'>🤖 Intelligence Artificielle</h4>
            <p>• OpenAI GPT-4/GPT-3.5</p>
            <p>• Scikit-learn & TensorFlow</p>
            <p>• XGBoost & CatBoost</p>
            <p>• Transformers NLP</p>
        </div>
        """, unsafe_allow_html=True)

    with tech_cols[1]:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white; padding: 20px; border-radius: 10px;'>
            <h4 style='margin-top: 0;'>📊 Data Science</h4>
            <p>• Pandas & NumPy</p>
            <p>• Plotly & Altair</p>
            <p>• Statsmodels</p>
            <p>• Prophet & ARIMA</p>
        </div>
        """, unsafe_allow_html=True)

    with tech_cols[2]:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); color: white; padding: 20px; border-radius: 10px;'>
            <h4 style='margin-top: 0;'>💻 Développement</h4>
            <p>• Streamlit & FastAPI</p>
            <p>• Python 3.11+</p>
            <p>• Docker & Kubernetes</p>
            <p>• Git & CI/CD</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Notre Approche
    st.subheader("🔬 Notre Approche Scientifique")

    approche_steps = [
        ("1️⃣", "Analyse Exploratoire", "Compréhension approfondie de vos données et du contexte métier"),
        ("2️⃣", "Préparation Scientifique", "Nettoyage et transformation rigoureuse des données"),
        ("3️⃣", "Modélisation Avancée", "Développement de modèles prédictifs adaptés"),
        ("4️⃣", "Validation Rigoureuse", "Tests statistiques et validation métier"),
        ("5️⃣", "Déploiement Sécurisé", "Intégration dans vos processus existants"),
        ("6️⃣", "Suivi Continu", "Monitoring et amélioration continue")
    ]

    for emoji, titre, description in approche_steps:
        st.markdown(f"""
        <div style='display: flex; align-items: flex-start; margin-bottom: 15px; padding: 15px; background-color: #f8f9fa; border-radius: 10px;'>
            <div style='font-size: 30px; margin-right: 15px;'>{emoji}</div>
            <div>
                <h4 style='margin: 0 0 5px 0; color: #1E3A8A;'>{titre}</h4>
                <p style='margin: 0; font-size: 14px;'>{description}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Section Sécurité
    st.subheader("🔒 Notre Engagement Sécurité")

    sec_cols = st.columns(2)

    with sec_cols[0]:
        st.markdown("""
        <div style='background-color: #d4edda; padding: 20px; border-radius: 10px;'>
            <h4 style='color: #155724; margin-top: 0;'>✅ Ce que nous garantissons</h4>
            <p>• Données 100% locales et sécurisées</p>
            <p>• Conformité RGPD et réglementations locales</p>
            <p>• Chiffrement AES-256 pour toutes les données</p>
            <p>• Authentification multi-facteurs</p>
            <p>• Sauvegardes régulières et chiffrées</p>
        </div>
        """, unsafe_allow_html=True)

    with sec_cols[1]:
        st.markdown("""
        <div style='background-color: #f8d7da; padding: 20px; border-radius: 10px;'>
            <h4 style='color: #721c24; margin-top: 0;'>❌ Ce que nous ne faisons pas</h4>
            <p>• Vendre ou partager vos données</p>
            <p>• Envoyer vos données vers le cloud sans consentement</p>
            <p>• Stocker des données sensibles non chiffrées</p>
            <p>• Utiliser des logiciels non sécurisés</p>
            <p>• Accès non autorisé à vos systèmes</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Section Partenariat DXC
    st.subheader("🤝 Notre Partenariat Stratégique")

    st.markdown("""
    <div style='text-align: center; padding: 30px; background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%); color: white; border-radius: 15px; margin: 20px 0;'>
        <h2 style='margin-bottom: 10px;'>💼 Partenariat avec DXC Technology</h2>
        <p style='font-size: 18px; margin-bottom: 20px;'>Expertise globale en transformation digitale</p>
        <a href='https://dxc.com/' target='_blank' style='display: inline-block; background-color: white; color: #1E3A8A; padding: 12px 30px; text-decoration: none; border-radius: 25px; font-weight: bold; margin-top: 10px;'>
            🌐 Découvrir DXC Technology
        </a>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ### 💡 Pourquoi ce partenariat ?

    Notre collaboration avec DXC Technology nous permet de :

    - **Bénéficier d'une expertise internationale** en transformation digitale
    - **Accéder aux dernières technologies** en matière de cloud et de sécurité
    - **Élargir notre portefeuille de solutions** pour mieux répondre à vos besoins
    - **Assurer un support technique** de niveau entreprise
    - **Maintenir notre avance technologique** grâce à la R&D commune
    """)

    st.markdown("---")

    # Contact
    st.subheader("📞 Contactez-nous")

    contact_cols = st.columns(3)

    with contact_cols[0]:
        st.markdown("""
        <div style='text-align: center; padding: 20px; background-color: #e8f4fd; border-radius: 10px;'>
            <div style='font-size: 40px; margin-bottom: 10px;'>📧</div>
            <h4>Email</h4>
            <p style='margin: 5px 0;'>contact@lik-insurance.ma</p>
            <p style='margin: 5px 0;'>support@lik-insurance.ma</p>
        </div>
        """, unsafe_allow_html=True)

    with contact_cols[1]:
        st.markdown("""
        <div style='text-align: center; padding: 20px; background-color: #f0f9ff; border-radius: 10px;'>
            <div style='font-size: 40px; margin-bottom: 10px;'>📱</div>
            <h4>Téléphone</h4>
            <p style='margin: 5px 0;'>+212 5 XX XX XX XX</p>
            <p style='margin: 5px 0;'>Lundi - Vendredi</p>
            <p style='margin: 5px 0;'>9h00 - 18h00</p>
        </div>
        """, unsafe_allow_html=True)

    with contact_cols[2]:
        st.markdown("""
        <div style='text-align: center; padding: 20px; background-color: #f8f9fa; border-radius: 10px;'>
            <div style='font-size: 40px; margin-bottom: 10px;'>🏢</div>
            <h4>Adresse</h4>
            <p style='margin: 5px 0;'>Tour Hassan Tower</p>
            <p style='margin: 5px 0;'>Bureau 1504, 15ème étage</p>
            <p style='margin: 5px 0;'>Rabat, Maroc</p>
        </div>
        """, unsafe_allow_html=True)
#=================================================
#FOOTER AMÉLIORÉ
#============================================================

