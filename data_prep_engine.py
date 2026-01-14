import pandas as pd
import numpy as np
from datetime import datetime

class DataPrepEngine:
    """
    Moteur de préparation et nettoyage des données.
    """
    def __init__(self):
        self.quality_report = {}
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Charge les données depuis un fichier CSV."""
        df = pd.read_csv(file_path)
        self.quality_report['initial_rows'] = len(df)
        self.quality_report['initial_cols'] = len(df.columns)
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Nettoie les données et calcule les métriques de qualité."""
        # Suppression des doublons
        df = df.drop_duplicates()
        
        # Gestion des valeurs manquantes
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        # Conversion des dates
        date_cols = [col for col in df.columns if 'date' in col.lower()]
        for col in date_cols:
            df[col] = pd.to_datetime(df[col], errors='coerce')
        
        self.quality_report['final_rows'] = len(df)
        self.quality_report['missing_values'] = df.isnull().sum().sum()
        self.quality_report['duplicates_removed'] = self.quality_report['initial_rows'] - len(df)
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crée de nouvelles features pour le modèle."""
        if 'start_date' in df.columns and 'end_date' in df.columns:
            df['policy_age_days'] = (df['end_date'] - df['start_date']).dt.days
            df['days_until_expiry'] = (df['end_date'] - datetime.now()).dt.days
            df['premium_change_pct'] = df['current_premium'] / df['initial_premium'] - 1
        
        # Saisonnalité
        if 'start_date' in df.columns:
            df['start_month'] = df['start_date'].dt.month
            df['start_quarter'] = df['start_date'].dt.quarter
        
        return df
    
    def get_quality_report(self) -> dict:
        """Retourne le rapport de qualité des données."""
        return self.quality_report