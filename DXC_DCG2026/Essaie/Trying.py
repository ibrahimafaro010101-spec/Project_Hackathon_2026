# =================================================
# predictive_engine.py — Modèles prédictifs
# =================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve, matthews_corrcoef,
    cohen_kappa_score, balanced_accuracy_score, log_loss,
    brier_score_loss, average_precision_score,
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error
)
import joblib
import plotly.graph_objects as go
import plotly.express as px
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import warnings
import streamlit as st
from datetime import datetime, timedelta
import os

warnings.filterwarnings('ignore')

# Importation conditionnelle des bibliothèques avancées
try:
    from catboost import CatBoostClassifier, CatBoostRegressor

    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    st.warning("⚠️ CatBoost n'est pas installé. Installation recommandée: pip install catboost")

try:
    from xgboost import XGBClassifier, XGBRegressor

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    st.warning("⚠️ XGBoost n'est pas installé. Installation recommandée: pip install xgboost")

try:
    import statsmodels.api as sm
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    from statsmodels.tsa.seasonal import seasonal_decompose, STL

    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    st.warning("⚠️ Statsmodels n'est pas installé. Installation recommandée: pip install statsmodels")

try:
    from prophet import Prophet

    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    st.warning("⚠️ Prophet n'est pas installé. Installation recommandée: pip install prophet")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    st.warning("⚠️ PyTorch n'est pas installé. Installation recommandée: pip install torch")


class AdvancedPredictiveEngine:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.label_encoders = {}
        self.feature_importance = None
        self.threshold_optimized = 0.5
        self.model_type = None
        self.target_binary = False
        self.cat_features = []
        self.train_metrics = None
        self.test_metrics = None
        self.time_series_model = None
        self.forecast_results = None

        # Modèles pour différents types de problèmes
        self.classification_model = None
        self.regression_model = None

        # Configuration
        self.config = {
            'random_state': 42,
            'test_size': 0.2,
            'cv_folds': 5,
            'forecast_horizon': 30,
            'time_series_freq': 'D'
        }

    # =================================================
    # MÉTHODES EXISTANTES (classification et régression)
    # =================================================

    def prepare_training_data(self, df, target_col, test_size=0.2, random_state=42):
        """
        Prépare les données pour l'entraînement avec gestion des déséquilibres
        """
        # Vérifier que la colonne cible existe
        if target_col not in df.columns:
            raise ValueError(f"La colonne cible '{target_col}' n'existe pas dans les données.")

        # Créer une copie pour éviter les modifications sur l'original
        df_processed = df.copy()

        # ANALYSE DE LA VARIABLE CIBLE
        y = df_processed[target_col]
        unique_values = y.nunique()
        value_counts = y.value_counts()

        st.info(f"Analyse de la variable cible '{target_col}':")
        st.info(f"- Nombre de classes uniques: {unique_values}")
        st.info(f"- Distribution: {dict(value_counts.head(10))}")

        # Détection automatique du type de problème
        if unique_values == 2:
            # Problème de classification binaire
            self.target_binary = True
            st.success("✅ Classification binaire détectée")

        elif unique_values > 2 and unique_values <= 10:
            # Problème de classification multi-classes (max 10 classes)
            self.target_binary = False
            if value_counts.min() < 5:
                st.warning(f"⚠️ Certaines classes ont très peu d'échantillons (min: {value_counts.min()})")
            st.info(f"Classification multi-classes ({unique_values} classes)")

        else:
            # Trop de classes - créer une variable binaire à partir de la distribution
            st.warning(f"⚠️ Trop de classes ({unique_values}). Création d'une variable cible binaire...")

            # Option 1: Si la colonne contient des codes risque
            if any(keyword in target_col.lower() for keyword in ['risque', 'risk', 'niveau', 'classe', 'grade']):
                # Garder les 20% des classes les plus rares comme risque élevé
                threshold = value_counts.quantile(0.2)
                rare_classes = value_counts[value_counts <= threshold].index
                y_binary = y.isin(rare_classes).astype(int)
                st.info(
                    f"Classes rares considérées comme risque ({len(rare_classes)} classes): {list(rare_classes)[:10]}")

            # Option 2: Basé sur les quantiles (si numérique)
            elif pd.api.types.is_numeric_dtype(y):
                threshold = y.quantile(0.75)  # Top 25% = risque élevé
                y_binary = (y >= threshold).astype(int)
                st.info(f"Seuil de risque: valeurs >= {threshold:.2f}")

            # Option 3: Par défaut, risque = 20% des échantillons les plus rares
            else:
                # Sélectionner 20% des échantillons comme risque
                n_risk = int(len(y) * 0.2)
                # Identifier les classes avec le moins d'occurrences
                rare_samples = y.value_counts().nsmallest(n_risk).index
                y_binary = y.isin(rare_samples).astype(int)
                st.info(f"20% des échantillons les plus rares considérés comme risque")

            # Remplacer y par la variable binaire
            y = y_binary
            df_processed[target_col] = y
            self.target_binary = True
            st.success(f"✅ Variable cible binaire créée. Distribution: {y.value_counts().to_dict()}")

        # Identifier les features catégorielles pour CatBoost
        categorical_cols = df_processed.select_dtypes(include=['object', 'category']).columns.tolist()
        if target_col in categorical_cols:
            categorical_cols.remove(target_col)
        self.cat_features = categorical_cols

        # Séparer features et target
        X = df_processed.drop(columns=[target_col])

        # Encodage des variables catégorielles (pour RandomForest et XGBoost)
        X = self._encode_categorical_features(X)

        # Imputation des valeurs manquantes
        X_imputed = self.imputer.fit_transform(X)
        X = pd.DataFrame(X_imputed, columns=X.columns)

        # Normalisation des features (sauf pour les arbres qui n'en ont pas besoin)
        # Mais utile pour la cohérence
        X_scaled = self.scaler.fit_transform(X)
        X = pd.DataFrame(X_scaled, columns=X.columns)

        # Diviser en train/test - SANS stratification si trop peu d'échantillons par classe
        if self.target_binary and y.nunique() == 2:
            min_class_count = y.value_counts().min()
            if min_class_count >= 2:
                # Stratification possible
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state, stratify=y
                )
            else:
                # Pas assez d'échantillons pour stratification
                st.warning("⚠️ Pas assez d'échantillons pour la stratification")
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state
                )
        else:
            # Pas de stratification pour multi-classes ou classification non binaire
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )

        # Gérer le déséquilibre des classes avec SMOTE (seulement pour classification binaire)
        if self.target_binary and y_train.nunique() == 2:
            class_counts = y_train.value_counts()
            st.info(f"Distribution avant SMOTE: {class_counts.to_dict()}")

            if class_counts.min() >= 2:  # Au moins 2 échantillons par classe
                try:
                    smote = SMOTE(random_state=random_state, sampling_strategy='auto')
                    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

                    st.info(f"Distribution après SMOTE: {pd.Series(y_train_resampled).value_counts().to_dict()}")

                    return X_train_resampled, X_test, y_train_resampled, y_test
                except Exception as e:
                    st.warning(f"SMOTE échoué: {e}. Utilisation des données originales.")

        return X_train, X_test, y_train, y_test

    def _encode_categorical_features(self, X):
        """Encode les variables catégorielles pour RandomForest et XGBoost"""
        X_encoded = X.copy()

        for col in X_encoded.select_dtypes(include=['object', 'category']).columns:
            # Ne pas encoder les colonnes avec trop de catégories
            if X_encoded[col].nunique() > 50:
                st.warning(f"Colonne '{col}' a trop de catégories ({X_encoded[col].nunique()}). Suppression.")
                X_encoded = X_encoded.drop(columns=[col])
                continue

            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
            self.label_encoders[col] = le

        return X_encoded

    def train_classification_model(self, X_train, y_train, model_type='random_forest', optimize=True):
        """
        Entraîne un modèle de classification
        """
        self.model_type = model_type
        self.target_binary = (y_train.nunique() == 2)

        # Vérifier la taille des données
        n_samples = len(X_train)
        n_features = X_train.shape[1]
        st.info(f"Nombre d'échantillons d'entraînement: {n_samples}")
        st.info(f"Nombre de features: {n_features}")

        if model_type == 'random_forest':
            # Définir les hyperparamètres adaptés à la taille des données
            if n_samples < 100:
                param_grid = {
                    'n_estimators': [50, 100],
                    'max_depth': [3, 5, 10],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2],
                    'max_features': ['sqrt', 'log2']
                }
            else:
                param_grid = {
                    'n_estimators': [100, 200],
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2'],
                    'class_weight': ['balanced', 'balanced_subsample'] if self.target_binary else [None]
                }

            base_model = RandomForestClassifier(
                random_state=42,
                n_jobs=-1,
                oob_score=True,
                class_weight='balanced' if self.target_binary else None
            )

        elif model_type == 'catboost':
            if not CATBOOST_AVAILABLE:
                st.error("❌ CatBoost n'est pas installé. Utilisation de RandomForest à la place.")
                return self.train_classification_model(X_train, y_train, 'random_forest', optimize)

            # Identifier les indices des colonnes catégorielles
            cat_features_indices = []
            for i, col in enumerate(X_train.columns):
                if col in self.cat_features:
                    cat_features_indices.append(i)
            st.info(f"CatBoost utilisera {len(cat_features_indices)} features catégorielles")

            # Paramètres pour CatBoost
            param_grid = {
                'iterations': [100, 200, 300],
                'depth': [4, 6, 8],
                'learning_rate': [0.01, 0.05, 0.1],
                'l2_leaf_reg': [1, 3, 5],
                'border_count': [32, 64, 128]
            }

            if n_samples < 100:
                param_grid = {
                    'iterations': [50, 100],
                    'depth': [3, 4, 6],
                    'learning_rate': [0.05, 0.1],
                    'l2_leaf_reg': [3, 5]
                }

            base_model = CatBoostClassifier(
                random_state=42,
                verbose=0,
                cat_features=cat_features_indices,
                auto_class_weights='Balanced' if self.target_binary else None
            )

        elif model_type == 'xgboost':
            if not XGBOOST_AVAILABLE:
                st.error("❌ XGBoost n'est pas installé. Utilisation de RandomForest à la place.")
                return self.train_classification_model(X_train, y_train, 'random_forest', optimize)

            # Paramètres pour XGBoost
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0],
                'gamma': [0, 0.1, 0.2]
            }

            if n_samples < 100:
                param_grid = {
                    'n_estimators': [50, 100],
                    'max_depth': [3, 5],
                    'learning_rate': [0.05, 0.1]
                }

            base_model = XGBClassifier(
                random_state=42,
                n_jobs=-1,
                use_label_encoder=False,
                eval_metric='logloss',
                scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train) if self.target_binary and sum(
                    y_train) > 0 else 1
            )

        else:
            raise ValueError(f"Type de modèle non supporté: {model_type}")

        # Optimisation des hyperparamètres
        if optimize and n_samples >= 30 and X_train.shape[1] >= 2:
            try:
                if n_samples < 100:
                    simplified_grid = {}
                    for key in list(param_grid.keys()):
                        simplified_grid[key] = [param_grid[key][0]]
                    param_grid = simplified_grid

                n_folds = min(3, max(2, n_samples // 10))

                grid_search = GridSearchCV(
                    estimator=base_model,
                    param_grid=param_grid,
                    cv=n_folds,
                    scoring='f1' if self.target_binary else 'accuracy',
                    n_jobs=-1,
                    verbose=0
                )

                with st.spinner(f"Optimisation des hyperparamètres ({model_type})..."):
                    grid_search.fit(X_train, y_train)

                self.classification_model = grid_search.best_estimator_

                st.success(f"✅ Meilleurs paramètres: {grid_search.best_params_}")
                st.success(f"✅ Meilleur score CV: {grid_search.best_score_:.3f}")

            except Exception as e:
                st.warning(f"⚠️ Échec de l'optimisation: {e}. Utilisation du modèle par défaut.")
                self.classification_model = base_model
                with st.spinner(f"Entraînement du modèle {model_type}..."):
                    self.classification_model.fit(X_train, y_train)
        else:
            self.classification_model = base_model
            with st.spinner(f"Entraînement du modèle {model_type}..."):
                self.classification_model.fit(X_train, y_train)

        # Calcul de l'importance des features
        if hasattr(self.classification_model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': self.classification_model.feature_importances_
            }).sort_values('importance', ascending=False)

            st.info("📊 Importance des features calculée")

        # Optimisation du seuil de classification (seulement pour classification binaire)
        if self.target_binary and hasattr(self.classification_model, 'predict_proba'):
            self._optimize_threshold(X_train, y_train)

        st.success(f"✅ Modèle {model_type} entraîné avec succès!")
        return self.classification_model

    def train_regression_model(self, X_train, y_train, model_type='random_forest', optimize=True):
        """
        Entraîne un modèle de régression
        """
        self.model_type = model_type

        n_samples = len(X_train)
        n_features = X_train.shape[1]

        st.info(f"Entraînement modèle de régression ({model_type})")
        st.info(f"Échantillons: {n_samples}, Features: {n_features}")

        if model_type == 'random_forest':
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
            base_model = RandomForestRegressor(random_state=42, n_jobs=-1)

        elif model_type == 'xgboost' and XGBOOST_AVAILABLE:
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.3],
                'subsample': [0.8, 1.0]
            }
            base_model = XGBRegressor(random_state=42, n_jobs=-1)

        elif model_type == 'catboost' and CATBOOST_AVAILABLE:
            param_grid = {
                'iterations': [100, 200],
                'depth': [6, 8, 10],
                'learning_rate': [0.03, 0.1],
                'l2_leaf_reg': [3, 5, 7]
            }
            base_model = CatBoostRegressor(random_state=42, verbose=0)

        else:
            st.warning(f"Modèle {model_type} non disponible, utilisation de Random Forest")
            base_model = RandomForestRegressor(random_state=42, n_jobs=-1)
            param_grid = {}

        # Optimisation
        if optimize and n_samples >= 20 and len(param_grid) > 0:
            try:
                n_folds = min(3, max(2, n_samples // 10))
                grid_search = GridSearchCV(
                    base_model,
                    param_grid,
                    cv=n_folds,
                    scoring='r2',
                    n_jobs=-1,
                    verbose=0
                )

                with st.spinner("Optimisation en cours..."):
                    grid_search.fit(X_train, y_train)

                self.regression_model = grid_search.best_estimator_
                st.success(f"Meilleurs paramètres: {grid_search.best_params_}")

            except Exception as e:
                st.warning(f"Optimisation échouée: {e}")
                self.regression_model = base_model
                self.regression_model.fit(X_train, y_train)
        else:
            self.regression_model = base_model
            self.regression_model.fit(X_train, y_train)

        st.success(f"✅ Modèle de régression {model_type} entraîné")
        return self.regression_model

    # =================================================
    # SÉRIES TEMPORELLES - NOUVELLE SECTION AVANCÉE
    # =================================================

    def prepare_time_series_data(self, df, date_col, target_col, freq='D'):
        """
        Prépare les données pour l'analyse de séries temporelles
        """
        try:
            # Vérifier et convertir la colonne de date
            if date_col not in df.columns:
                raise ValueError(f"Colonne de date '{date_col}' non trouvée")

            if target_col not in df.columns:
                raise ValueError(f"Colonne cible '{target_col}' non trouvée")

            # Créer une copie
            df_ts = df.copy()

            # Convertir la date
            df_ts[date_col] = pd.to_datetime(df_ts[date_col], errors='coerce')

            # Trier par date
            df_ts = df_ts.sort_values(date_col)

            # Créer la série temporelle
            ts_series = df_ts.set_index(date_col)[target_col]

            # Redresser la fréquence
            ts_series = ts_series.asfreq(freq)

            # Remplir les valeurs manquantes
            ts_series = ts_series.fillna(method='ffill').fillna(method='bfill')

            st.success(f"✅ Série temporelle préparée: {len(ts_series)} points")
            st.info(f"Période: {ts_series.index[0]} à {ts_series.index[-1]}")

            return ts_series

        except Exception as e:
            st.error(f"❌ Erreur lors de la préparation des séries temporelles: {e}")
            return None

    def analyze_time_series(self, ts_series, freq='D'):
        """
        Analyse exploratoire d'une série temporelle
        """
        try:
            st.subheader("📊 Analyse exploratoire de la série temporelle")

            # Statistiques descriptives
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Points de données", len(ts_series))
            with col2:
                st.metric("Période", f"{(ts_series.index[-1] - ts_series.index[0]).days} jours")
            with col3:
                st.metric("Valeur moyenne", f"{ts_series.mean():.2f}")
            with col4:
                st.metric("Écart-type", f"{ts_series.std():.2f}")

            # Graphique de la série
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=ts_series.index,
                y=ts_series.values,
                mode='lines',
                name='Série originale',
                line=dict(color='blue', width=2)
            ))
            fig.update_layout(
                title='Série temporelle originale',
                xaxis_title='Date',
                yaxis_title='Valeur',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

            # Test de stationnarité (Dickey-Fuller)
            if STATSMODELS_AVAILABLE and len(ts_series) > 10:
                try:
                    adf_result = adfuller(ts_series.dropna())
                    st.info(f"Test de Dickey-Fuller: p-value = {adf_result[1]:.4f}")

                    if adf_result[1] > 0.05:
                        st.warning("⚠️ La série n'est pas stationnaire (p > 0.05)")
                    else:
                        st.success("✅ La série est stationnaire (p ≤ 0.05)")
                except:
                    pass

            # Décomposition saisonnière
            if STATSMODELS_AVAILABLE and len(ts_series) > 50:
                try:
                    # Déterminer la période saisonnière
                    if freq == 'D':
                        period = 7  # hebdomadaire
                    elif freq == 'M':
                        period = 12  # annuel
                    else:
                        period = min(30, len(ts_series) // 2)

                    decomposition = seasonal_decompose(ts_series, model='additive', period=period)

                    # Créer le graphique de décomposition
                    fig_decomp = make_subplots(
                        rows=4, cols=1,
                        subplot_titles=['Série originale', 'Tendance', 'Saisonnalité', 'Résidus'],
                        vertical_spacing=0.1
                    )

                    fig_decomp.add_trace(
                        go.Scatter(x=ts_series.index, y=ts_series.values, mode='lines', name='Original'),
                        row=1, col=1
                    )
                    fig_decomp.add_trace(
                        go.Scatter(x=decomposition.trend.index, y=decomposition.trend, mode='lines', name='Tendance'),
                        row=2, col=1
                    )
                    fig_decomp.add_trace(
                        go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal, mode='lines',
                                   name='Saisonnalité'),
                        row=3, col=1
                    )
                    fig_decomp.add_trace(
                        go.Scatter(x=decomposition.resid.index, y=decomposition.resid, mode='lines', name='Résidus'),
                        row=4, col=1
                    )

                    fig_decomp.update_layout(height=600, showlegend=False)
                    st.plotly_chart(fig_decomp, use_container_width=True)

                except Exception as e:
                    st.info(f"Décomposition non disponible: {e}")

            # Autocorrélation et autocorrélation partielle
            if STATSMODELS_AVAILABLE and len(ts_series) > 30:
                try:
                    col_acf, col_pacf = st.columns(2)

                    with col_acf:
                        fig_acf = go.Figure()
                        acf_values = acf(ts_series.dropna(), nlags=min(40, len(ts_series) // 2))
                        fig_acf.add_trace(go.Bar(
                            x=list(range(len(acf_values))),
                            y=acf_values,
                            name='ACF'
                        ))
                        fig_acf.update_layout(title='Autocorrélation (ACF)', height=300)
                        st.plotly_chart(fig_acf, use_container_width=True)

                    with col_pacf:
                        fig_pacf = go.Figure()
                        pacf_values = pacf(ts_series.dropna(), nlags=min(40, len(ts_series) // 2))
                        fig_pacf.add_trace(go.Bar(
                            x=list(range(len(pacf_values))),
                            y=pacf_values,
                            name='PACF'
                        ))
                        fig_pacf.update_layout(title='Autocorrélation Partielle (PACF)', height=300)
                        st.plotly_chart(fig_pacf, use_container_width=True)

                except:
                    pass

            return True

        except Exception as e:
            st.error(f"❌ Erreur lors de l'analyse: {e}")
            return False

    def train_time_series_model(self, ts_series, model_type='prophet', freq='D', forecast_horizon=30):
        """
        Entraîne un modèle de série temporelle
        """
        try:
            st.subheader(f"⏰ Entraînement du modèle {model_type}")

            # S'assurer que la série a assez de données
            if len(ts_series) < 20:
                st.error("❌ Pas assez de données pour l'entraînement (minimum 20 points)")
                return None, None, None

            # Séparer train/test (80/20)
            train_size = int(len(ts_series) * 0.8)
            train_series = ts_series.iloc[:train_size]
            test_series = ts_series.iloc[train_size:]

            st.info(f"Train: {len(train_series)} points, Test: {len(test_series)} points")

            if model_type == 'prophet':
                if not PROPHET_AVAILABLE:
                    st.error("❌ Prophet n'est pas installé")
                    return None, None, None

                # Préparer les données pour Prophet
                prophet_df = pd.DataFrame({
                    'ds': train_series.index,
                    'y': train_series.values
                })

                # Créer et entraîner le modèle
                prophet_model = Prophet(
                    yearly_seasonality=True,
                    weekly_seasonality=(freq == 'D'),
                    daily_seasonality=False,
                    changepoint_prior_scale=0.05,
                    seasonality_prior_scale=10.0
                )

                with st.spinner("Entraînement Prophet en cours..."):
                    prophet_model.fit(prophet_df)

                self.time_series_model = prophet_model

                # Prévisions
                future = prophet_model.make_future_dataframe(periods=forecast_horizon, freq=freq)
                forecast = prophet_model.predict(future)

                # Extraire les prévisions
                forecast_series = pd.Series(
                    forecast['yhat'].values,
                    index=forecast['ds']
                )

                # Prévisions pour la période de test
                test_forecast = forecast.iloc[-len(test_series) - forecast_horizon:-forecast_horizon]

            elif model_type == 'arima':
                if not STATSMODELS_AVAILABLE:
                    st.error("❌ Statsmodels n'est pas installé")
                    return None, None, None

                # ARIMA automatique
                with st.spinner("Recherche des meilleurs paramètres ARIMA..."):
                    # Essayez différentes combinaisons
                    best_aic = np.inf
                    best_order = None
                    best_model = None

                    # Grille de recherche simplifiée
                    p_range = range(0, 3)
                    d_range = range(0, 2)
                    q_range = range(0, 3)

                    for p in p_range:
                        for d in d_range:
                            for q in q_range:
                                try:
                                    model = ARIMA(train_series, order=(p, d, q))
                                    model_fit = model.fit()

                                    if model_fit.aic < best_aic:
                                        best_aic = model_fit.aic
                                        best_order = (p, d, q)
                                        best_model = model_fit
                                except:
                                    continue

                    if best_model is None:
                        # Utiliser des paramètres par défaut
                        best_order = (1, 1, 1)
                        model = ARIMA(train_series, order=best_order)
                        best_model = model.fit()

                st.success(f"✅ ARIMA{best_order} sélectionné (AIC: {best_aic:.2f})")

                self.time_series_model = best_model

                # Prévisions
                forecast = best_model.forecast(steps=len(test_series) + forecast_horizon)
                forecast_series = pd.Series(
                    forecast,
                    index=pd.date_range(
                        start=train_series.index[-1] + pd.Timedelta(days=1),
                        periods=len(test_series) + forecast_horizon,
                        freq=freq
                    )
                )

                test_forecast = forecast_series.iloc[:len(test_series)]

            elif model_type == 'exponential_smoothing':
                if not STATSMODELS_AVAILABLE:
                    st.error("❌ Statsmodels n'est pas installé")
                    return None, None, None

                # Holt-Winters Exponential Smoothing
                with st.spinner("Entraînement Exponential Smoothing..."):
                    # Déterminer la période saisonnière
                    if freq == 'D':
                        seasonal_periods = 7
                    elif freq == 'M':
                        seasonal_periods = 12
                    else:
                        seasonal_periods = None

                    if seasonal_periods and len(train_series) > 2 * seasonal_periods:
                        # Avec saisonnalité
                        model = ExponentialSmoothing(
                            train_series,
                            seasonal_periods=seasonal_periods,
                            trend='add',
                            seasonal='add'
                        )
                    else:
                        # Sans saisonnalité
                        model = ExponentialSmoothing(
                            train_series,
                            trend='add'
                        )

                    model_fit = model.fit()

                self.time_series_model = model_fit

                # Prévisions
                forecast = model_fit.forecast(len(test_series) + forecast_horizon)
                forecast_series = pd.Series(
                    forecast,
                    index=pd.date_range(
                        start=train_series.index[-1] + pd.Timedelta(days=1),
                        periods=len(test_series) + forecast_horizon,
                        freq=freq
                    )
                )

                test_forecast = forecast_series.iloc[:len(test_series)]

            elif model_type == 'random_forest' and len(ts_series) > 50:
                # Random Forest pour séries temporelles avec features de lag
                with st.spinner("Préparation des features temporelles..."):
                    # Créer des features de lag
                    max_lag = min(7, len(train_series) // 10)
                    df_features = self._create_time_series_features(train_series, max_lag)

                    X_train = df_features.drop(columns=['target'])
                    y_train = df_features['target']

                    # Entraîner le modèle
                    rf_model = RandomForestRegressor(
                        n_estimators=100,
                        random_state=42,
                        n_jobs=-1
                    )
                    rf_model.fit(X_train, y_train)

                self.time_series_model = rf_model
                self.time_series_lags = max_lag

                # Prévisions récursives
                test_forecast = self._recursive_forecast_rf(
                    train_series, rf_model, max_lag, len(test_series) + forecast_horizon
                )

                forecast_series = pd.Series(
                    test_forecast,
                    index=pd.date_range(
                        start=train_series.index[-1] + pd.Timedelta(days=1),
                        periods=len(test_series) + forecast_horizon,
                        freq=freq
                    )
                )

                test_forecast = forecast_series.iloc[:len(test_series)]

            else:
                st.error(f"❌ Modèle {model_type} non disponible ou pas assez de données")
                return None, None, None

            # Évaluation sur l'ensemble de test
            if len(test_series) > 0:
                evaluation = self.evaluate_time_series_model(test_series, test_forecast)
                self._display_time_series_metrics(evaluation)

            # Sauvegarder les résultats
            self.forecast_results = {
                'model_type': model_type,
                'train_series': train_series,
                'test_series': test_series,
                'forecast_series': forecast_series,
                'full_forecast': forecast_series.iloc[-forecast_horizon:],
                'evaluation': evaluation if len(test_series) > 0 else None
            }

            return self.time_series_model, forecast_series, test_forecast

        except Exception as e:
            st.error(f"❌ Erreur lors de l'entraînement: {e}")
            import traceback
            st.code(traceback.format_exc())
            return None, None, None

    def _create_time_series_features(self, series, max_lag):
        """Crée des features pour Random Forest sur séries temporelles"""
        df = pd.DataFrame({'value': series.values})

        # Créer des lags
        for lag in range(1, max_lag + 1):
            df[f'lag_{lag}'] = df['value'].shift(lag)

        # Features statistiques glissantes
        df['rolling_mean_7'] = df['value'].rolling(window=7).mean()
        df['rolling_std_7'] = df['value'].rolling(window=7).std()

        # Features temporelles
        if hasattr(series.index, 'month'):
            df['month'] = series.index.month
            df['quarter'] = series.index.quarter
            df['dayofweek'] = series.index.dayofweek

        # Target (décalée de 1)
        df['target'] = df['value'].shift(-1)

        return df.dropna()

    def _recursive_forecast_rf(self, history, model, max_lag, steps):
        """Prévisions récursives avec Random Forest"""
        forecast = []
        current_data = list(history.values[-max_lag:])

        for _ in range(steps):
            # Créer les features pour le point actuel
            features = {}
            for lag in range(1, max_lag + 1):
                features[f'lag_{lag}'] = current_data[-lag] if len(current_data) >= lag else current_data[-1]

            # Prédire le point suivant
            pred = model.predict(pd.DataFrame([features]))[0]
            forecast.append(pred)

            # Mettre à jour les données
            current_data.append(pred)

        return forecast

    def evaluate_time_series_model(self, actual, predicted):
        """Évalue un modèle de série temporelle"""
        if len(actual) != len(predicted):
            min_len = min(len(actual), len(predicted))
            actual = actual[:min_len]
            predicted = predicted[:min_len]

        metrics = {}

        # Métriques d'erreur
        metrics['mse'] = mean_squared_error(actual, predicted)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(actual, predicted)
        metrics['mape'] = mean_absolute_percentage_error(actual, predicted) * 100

        # Score R²
        metrics['r2'] = r2_score(actual, predicted)

        # Directional accuracy
        if len(actual) > 1:
            actual_changes = np.sign(np.diff(actual))
            predicted_changes = np.sign(np.diff(predicted))
            directional_accuracy = (actual_changes == predicted_changes).mean()
            metrics['directional_accuracy'] = directional_accuracy * 100

        return metrics

    def _display_time_series_metrics(self, metrics):
        """Affiche les métriques des séries temporelles"""
        st.subheader("📊 Métriques de performance")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("RMSE", f"{metrics.get('rmse', 0):.2f}")
        with col2:
            st.metric("MAE", f"{metrics.get('mae', 0):.2f}")
        with col3:
            st.metric("MAPE", f"{metrics.get('mape', 0):.2f}%")
        with col4:
            st.metric("R²", f"{metrics.get('r2', 0):.3f}")

        if 'directional_accuracy' in metrics:
            st.info(f"Précision directionnelle: {metrics['directional_accuracy']:.1f}%")

    def plot_time_series_forecast(self, ts_series, forecast_series, model_type, forecast_horizon=30):
        """Génère des graphiques pour les séries temporelles"""
        try:
            # Graphique principal
            fig = go.Figure()

            # Données historiques
            fig.add_trace(go.Scatter(
                x=ts_series.index,
                y=ts_series.values,
                mode='lines',
                name='Données historiques',
                line=dict(color='blue', width=2)
            ))

            # Prévisions
            if forecast_series is not None:
                fig.add_trace(go.Scatter(
                    x=forecast_series.index,
                    y=forecast_series.values,
                    mode='lines',
                    name='Prévisions',
                    line=dict(color='red', width=2, dash='dash')
                ))

            # Mettre en évidence la période de prévision
            if len(ts_series) > 0 and forecast_series is not None:
                last_historical = ts_series.index[-1]
                fig.add_vline(
                    x=last_historical,
                    line_dash="dash",
                    line_color="gray",
                    annotation_text="Début prévisions",
                    annotation_position="top right"
                )

            fig.update_layout(
                title=f'Prévisions {model_type} - Horizon: {forecast_horizon} périodes',
                xaxis_title='Date',
                yaxis_title='Valeur',
                height=500,
                hovermode='x unified'
            )

            return fig

        except Exception as e:
            st.error(f"❌ Erreur lors de la création du graphique: {e}")
            return None

    def generate_time_series_insights(self, ts_series, forecast_results):
        """Génère des insights à partir des séries temporelles"""
        try:
            insights = []

            # Tendance générale
            if len(ts_series) > 10:
                slope, _ = np.polyfit(range(len(ts_series)), ts_series.values, 1)
                if slope > 0:
                    insights.append("📈 Tendance à la hausse détectée")
                elif slope < 0:
                    insights.append("📉 Tendance à la baisse détectée")
                else:
                    insights.append("➡️ Tendance stable détectée")

            # Volatilité
            volatility = ts_series.std() / ts_series.mean() if ts_series.mean() != 0 else 0
            if volatility > 0.5:
                insights.append("⚡ Haute volatilité détectée")
            elif volatility > 0.2:
                insights.append("🌊 Volatilité modérée")
            else:
                insights.append("🌊 Faible volatilité")

            # Saisonnalité
            if STATSMODELS_AVAILABLE and len(ts_series) > 50:
                try:
                    if hasattr(ts_series.index, 'month'):
                        monthly_means = ts_series.groupby(ts_series.index.month).mean()
                        if monthly_means.std() > monthly_means.mean() * 0.1:
                            insights.append("🔄 Saisonnalité mensuelle détectée")
                except:
                    pass

            # Insights des prévisions
            if forecast_results and 'full_forecast' in forecast_results:
                forecast = forecast_results['full_forecast']
                if len(forecast) > 0:
                    forecast_change = ((forecast.iloc[-1] - forecast.iloc[0]) / forecast.iloc[0] * 100
                                       if forecast.iloc[0] != 0 else 0)

                    if forecast_change > 10:
                        insights.append(f"🚀 Prévision: augmentation de {forecast_change:.1f}%")
                    elif forecast_change < -10:
                        insights.append(f"⚠️ Prévision: diminution de {abs(forecast_change):.1f}%")
                    else:
                        insights.append(f"📊 Prévision: variation de {forecast_change:.1f}%")

            return insights

        except Exception as e:
            st.warning(f"Insights limités: {e}")
            return ["📊 Analyse de base disponible"]

    # =================================================
    # MÉTHODES EXISTANTES POUR L'ÉVALUATION
    # =================================================

    def _optimize_threshold(self, X_train, y_train):
        """Optimise le seuil de classification pour maximiser le F1-score"""
        try:
            y_proba = self.classification_model.predict_proba(X_train)[:, 1]

            # Calcul des métriques pour différents seuils
            thresholds = np.arange(0.1, 0.9, 0.05)
            best_f1 = 0
            best_threshold = 0.5
            best_metrics = {}

            for threshold in thresholds:
                y_pred = (y_proba >= threshold).astype(int)
                f1 = f1_score(y_train, y_pred, zero_division=0)

                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
                    best_metrics = {
                        'f1': f1,
                        'precision': precision_score(y_train, y_pred, zero_division=0),
                        'recall': recall_score(y_train, y_pred, zero_division=0),
                        'mcc': matthews_corrcoef(y_train, y_pred)
                    }

            self.threshold_optimized = best_threshold
            st.info(f"🔧 Seuil optimal pour classification: {best_threshold:.2f}")

        except Exception as e:
            st.warning(f"⚠️ Impossible d'optimiser le seuil: {e}")
            self.threshold_optimized = 0.5

    def evaluate(self, X_test, y_test):
        """
        Évalue les performances du modèle avec des métriques complètes
        """
        if self.classification_model is None:
            raise ValueError("Le modèle n'a pas été entraîné.")

        predictions, probabilities = self._predict_with_model(X_test)

        # Métriques de base
        metrics = {
            'accuracy': accuracy_score(y_test, predictions),
            'precision': precision_score(y_test, predictions, zero_division=0,
                                         average='weighted' if not self.target_binary else 'binary'),
            'recall': recall_score(y_test, predictions, zero_division=0,
                                   average='weighted' if not self.target_binary else 'binary'),
            'f1_score': f1_score(y_test, predictions, zero_division=0,
                                 average='weighted' if not self.target_binary else 'binary'),
            'balanced_accuracy': balanced_accuracy_score(y_test, predictions),
            'mcc': matthews_corrcoef(y_test, predictions),
            'kappa': cohen_kappa_score(y_test, predictions),
            'model_type': self.model_type,
            'n_features': X_test.shape[1],
            'test_samples': len(y_test),
            'target_binary': self.target_binary,
            'optimal_threshold': self.threshold_optimized if self.target_binary else None
        }

        # ROC-AUC seulement pour classification binaire avec probabilités
        if self.target_binary and probabilities is not None and y_test.nunique() == 2:
            try:
                metrics.update({
                    'roc_auc': roc_auc_score(y_test, probabilities),
                    'average_precision': average_precision_score(y_test, probabilities),
                    'log_loss': log_loss(y_test, probabilities),
                    'brier_score': brier_score_loss(y_test, probabilities)
                })
            except:
                pass

        # Matrice de confusion
        metrics['confusion_matrix'] = confusion_matrix(y_test, predictions).tolist()

        # Rapport de classification
        metrics['classification_report'] = classification_report(
            y_test, predictions,
            output_dict=True,
            zero_division=0
        )

        # Calcul des métriques par classe pour classification binaire
        if self.target_binary and y_test.nunique() == 2:
            try:
                tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
                metrics.update({
                    'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
                    'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
                    'false_negative_rate': fn / (fn + tp) if (fn + tp) > 0 else 0,
                    'true_positive': int(tp),
                    'true_negative': int(tn),
                    'false_positive': int(fp),
                    'false_negative': int(fn),
                    'precision_positive': tp / (tp + fp) if (tp + fp) > 0 else 0,
                    'precision_negative': tn / (tn + fn) if (tn + fn) > 0 else 0,
                    'f1_positive': 2 * metrics['precision_positive'] * metrics['recall'] /
                                   (metrics['precision_positive'] + metrics['recall'])
                    if (metrics['precision_positive'] + metrics['recall']) > 0 else 0
                })
            except:
                pass

        self.test_metrics = metrics
        return metrics

    def _predict_with_model(self, X):
        """Prédiction avec le modèle approprié"""
        if self.classification_model is not None:
            if self.target_binary and hasattr(self.classification_model, 'predict_proba'):
                probabilities = self.classification_model.predict_proba(X)[:, 1]
                predictions = (probabilities >= self.threshold_optimized).astype(int)
                return predictions, probabilities
            else:
                predictions = self.classification_model.predict(X)
                probabilities = (self.classification_model.predict_proba(X)
                                 if hasattr(self.classification_model, 'predict_proba') else None)
                return predictions, probabilities
        elif self.regression_model is not None:
            predictions = self.regression_model.predict(X)
            return predictions, None
        else:
            raise ValueError("Aucun modèle entraîné")

    # =================================================
    # MÉTHODES EXISTANTES POUR LA VISUALISATION
    # =================================================

    def get_feature_importance_plot(self):
        """Génère un graphique d'importance des features"""
        if self.feature_importance is None:
            return None

        top_features = self.feature_importance.head(15)

        fig = go.Figure(data=[
            go.Bar(
                x=top_features['importance'],
                y=top_features['feature'],
                orientation='h',
                marker_color='crimson',
                text=[f'{imp:.3f}' for imp in top_features['importance']],
                textposition='outside'
            )
        ])

        fig.update_layout(
            title=f'Top 15 des Features les plus importantes ({self.model_type})',
            xaxis_title='Importance',
            yaxis_title='Feature',
            height=500,
            template='plotly_white',
            showlegend=False
        )

        return fig

    def get_roc_curve_plot(self, X_test, y_test):
        """Génère la courbe ROC (seulement pour classification binaire)"""
        if not self.target_binary or not hasattr(self.classification_model, 'predict_proba'):
            return None

        try:
            probabilities = self.classification_model.predict_proba(X_test)[:, 1]
            if y_test.nunique() != 2:
                return None

            fpr, tpr, thresholds = roc_curve(y_test, probabilities)
            auc_score = roc_auc_score(y_test, probabilities)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'ROC curve (AUC = {auc_score:.3f})',
                line=dict(color='darkorange', width=3),
                fill='tozeroy',
                fillcolor='rgba(255, 140, 0, 0.2)'
            ))
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                name='Random (AUC = 0.5)',
                line=dict(color='navy', width=2, dash='dash')
            ))

            fig.update_layout(
                title=f'Courbe ROC - {self.model_type}',
                xaxis_title='Taux de Faux Positifs',
                yaxis_title='Taux de Vrais Positifs',
                yaxis=dict(scaleanchor="x", scaleratio=1),
                xaxis=dict(constrain='domain'),
                height=500,
                template='plotly_white',
                hovermode='x unified'
            )

            return fig
        except:
            return None

    # =================================================
    # MÉTHODES UTILITAIRES
    # =================================================

    def save_model(self, path='models/risk_model.pkl'):
        """Sauvegarde le modèle entraîné"""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)

        model_data = {
            'classification_model': self.classification_model,
            'regression_model': self.regression_model,
            'time_series_model': self.time_series_model,
            'scaler': self.scaler,
            'imputer': self.imputer,
            'label_encoders': self.label_encoders,
            'feature_importance': self.feature_importance,
            'threshold': self.threshold_optimized,
            'model_type': self.model_type,
            'target_binary': self.target_binary,
            'cat_features': self.cat_features,
            'train_metrics': self.train_metrics,
            'test_metrics': self.test_metrics,
            'forecast_results': self.forecast_results
        }

        joblib.dump(model_data, path)
        st.success(f"✅ Modèle sauvegardé: {path}")

    def load_model(self, path='models/risk_model.pkl'):
        """Charge un modèle sauvegardé"""
        model_data = joblib.load(path)

        self.classification_model = model_data.get('classification_model')
        self.regression_model = model_data.get('regression_model')
        self.time_series_model = model_data.get('time_series_model')
        self.scaler = model_data.get('scaler')
        self.imputer = model_data.get('imputer')
        self.label_encoders = model_data.get('label_encoders')
        self.feature_importance = model_data.get('feature_importance')
        self.threshold_optimized = model_data.get('threshold')
        self.model_type = model_data.get('model_type')
        self.target_binary = model_data.get('target_binary')
        self.cat_features = model_data.get('cat_features', [])
        self.train_metrics = model_data.get('train_metrics')
        self.test_metrics = model_data.get('test_metrics')
        self.forecast_results = model_data.get('forecast_results')

        st.success(f"✅ Modèle chargé: {path}")

    def get_model_info(self):
        """Retourne des informations sur le modèle"""
        info = {
            'type': self.model_type,
            'target_type': 'Binaire' if self.target_binary else 'Multi-classes',
            'optimal_threshold': self.threshold_optimized if self.target_binary else 'N/A',
            'features_count': self.feature_importance.shape[0] if self.feature_importance is not None else 'N/A',
            'train_metrics_available': self.train_metrics is not None,
            'test_metrics_available': self.test_metrics is not None,
            'time_series_model_available': self.time_series_model is not None
        }

        return info


# Fonction utilitaire pour créer des subplots
def make_subplots(rows=1, cols=1, **kwargs):
    """Crée des subplots avec Plotly"""
    return go.Figure().set_subplots(rows=rows, cols=cols, **kwargs)