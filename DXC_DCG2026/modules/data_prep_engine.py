# modules/data_prep_engine.py

import pandas as pd
import numpy as np
from datetime import datetime


class DataPrepEngine:
    """
    Moteur de préparation et de qualité des données
    Assurance Automobile – Hackathon
    """

    def __init__(self):
        self.quality_report = {}

    # -------------------------------------------------
    # Chargement des données
    # -------------------------------------------------
    def load_data(self, file) -> pd.DataFrame:
        if hasattr(file, "name"):  # Streamlit UploadedFile
            filename = file.name.lower()
            if filename.endswith(".csv"):
                df = pd.read_csv(file, decimal=",")
            elif filename.endswith(".xlsx"):
                df = pd.read_excel(file)
            else:
                raise ValueError("Format non supporté")
        elif isinstance(file, str):  # chemin local
            if file.endswith(".csv"):
                df = pd.read_csv(file, decimal=",")
            elif file.endswith(".xlsx"):
                df = pd.read_excel(file)
            else:
                raise ValueError("Format non supporté")
        else:
            raise ValueError("Type de fichier invalide")

        self.quality_report["initial_rows"] = len(df)
        self.quality_report["initial_cols"] = len(df.columns)

        return df

    # -------------------------------------------------
    # 🔹 NETTOYAGE (MANQUANT AVANT)
    # -------------------------------------------------
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # -------------------------------------------------
        # 1. Conversion explicite des dates
        # -------------------------------------------------
        date_cols = ["dtemi", "datedeb", "datefin", "datcpt"]
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], dayfirst=True, errors="coerce")

        # -------------------------------------------------
        # 2. Conversion explicite des variables numériques
        # -------------------------------------------------
        numeric_candidates = [
            "Prime",
            "nb_jour_couv",
            "age_conducteur",
            "anciennete_permis_en_annees",
            "nb_sinistres_passe",
            "cout_sinistres_passe",
            "nb_impayes",
            "retard_paiement_moyen_jours",
            "anciennete_client_en_jours",
            "nb_changements_assureur"
        ]

        for col in numeric_candidates:
            if col in df.columns:
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.replace(",", ".", regex=False)
                    .replace("nan", np.nan)
                    .astype(float)
                )

        # -------------------------------------------------
        # 3. Suppression doublons
        # -------------------------------------------------
        df = df.drop_duplicates()

        # -------------------------------------------------
        # 4. Imputation NUMÉRIQUE SÛRE (colonne par colonne)
        # -------------------------------------------------
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())

        # -------------------------------------------------
        # 5. Rapport qualité
        # -------------------------------------------------
        self.quality_report["final_rows"] = len(df)
        self.quality_report["missing_values"] = int(df.isna().sum().sum())

        return df

    # -------------------------------------------------
    # 🔹 FEATURE ENGINEERING
    # -------------------------------------------------
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # -------------------------------------------------
        # Durée de couverture
        # -------------------------------------------------
        if "datedeb" in df.columns and "datefin" in df.columns:
            df["nb_jour_couv"] = (
                    df["datefin"] - df["datedeb"]
            ).dt.days

            df["nb_jour_couv"] = df["nb_jour_couv"].clip(lower=1)

        # -------------------------------------------------
        # Ancienneté contrat
        # -------------------------------------------------
        if "datedeb" in df.columns:
            df["anciennete_contrat_jours"] = (
                    pd.Timestamp.today() - df["datedeb"]
            ).dt.days

        # -------------------------------------------------
        # Prime normalisée
        # -------------------------------------------------
        if "Prime" in df.columns and "nb_jour_couv" in df.columns:
            df["prime_par_jour"] = df["Prime"] / df["nb_jour_couv"]
            df["prime_annualisee"] = df["prime_par_jour"] * 365

        # -------------------------------------------------
        # Saison de souscription
        # -------------------------------------------------
        if "datedeb" in df.columns:
            df["start_month"] = df["datedeb"].dt.month
            df["start_quarter"] = df["datedeb"].dt.quarter

        return df

    # -------------------------------------------------
    # Rapport qualité
    # -------------------------------------------------
    def get_quality_report(self) -> dict:
        return self.quality_report