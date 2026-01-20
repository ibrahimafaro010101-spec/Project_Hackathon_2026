# modules/custom_llm_client.py - Version Finale Optimisée
import os
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from openai import OpenAI
import traceback


class OpenAIAnalyzer:
    """
    Client OpenAI optimisé pour l'analyse de données d'assurance
    Version avec gestion d'erreurs robuste et méthodes étendues
    """

    def __init__(self, api_key: str, model: str = "gpt-4"):
        """
        Initialise le client OpenAI

        Args:
            api_key: Clé API OpenAI
            model: Modèle à utiliser (gpt-4, gpt-3.5-turbo, etc.)
        """
        if not api_key or api_key.strip() == "":
            raise ValueError("Clé API OpenAI requise et non vide")

        try:
            self.api_key = api_key.strip()
            self.client = OpenAI(api_key=self.api_key)
            self.model = model
            print(f"✅ Client OpenAI initialisé avec le modèle: {model}")
        except Exception as e:
            raise ValueError(f"Erreur d'initialisation OpenAI: {e}")

    def analyze_query(self, query: str, dataframe: pd.DataFrame,
                      max_context_rows: int = 100) -> Dict[str, Any]:
        """
        Analyse une requête utilisateur avec le contexte des données

        Args:
            query: Question de l'utilisateur
            dataframe: DataFrame contenant les données
            max_context_rows: Nombre maximum de lignes pour le contexte

        Returns:
            Dict contenant l'analyse complète
        """
        try:
            # Validation des entrées
            if not query or query.strip() == "":
                return {"erreur": "Requête vide"}

            if dataframe is None or dataframe.empty:
                return {"erreur": "Aucune donnée disponible"}

            # Générer un résumé des données
            data_summary = self._generate_data_summary(dataframe, max_context_rows)

            # Construire le prompt
            prompt = self._build_analysis_prompt(query, data_summary, dataframe)

            # Appeler l'API OpenAI
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": self._get_system_prompt()
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=1500,
                timeout=60
            )

            result_text = response.choices[0].message.content

            # Parser la réponse
            return self._parse_analysis_result(result_text, query)

        except Exception as e:
            error_msg = f"Erreur lors de l'analyse: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            return {
                "erreur": error_msg,
                "comprehension": "Erreur lors du traitement",
                "methodologie": "Non disponible",
                "insights": [],
                "reponse_detaillee": error_msg
            }

    def _generate_data_summary(self, df: pd.DataFrame, max_rows: int = 100) -> str:
        """Génère un résumé concis des données"""

        # Échantillonnage si nécessaire
        df_sample = df.head(max_rows) if len(df) > max_rows else df

        # Informations de base
        summary = f"""
📊 DONNÉES DISPONIBLES:
- Lignes: {len(df):,}
- Colonnes: {len(df.columns)}
- Mémoire: {df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB

📋 COLONNES ({len(df.columns)}):
"""

        # Détails des colonnes (limiter à 20)
        for col in list(df.columns)[:20]:
            dtype = str(df[col].dtype)
            non_null = df[col].notna().sum()
            unique = df[col].nunique()

            col_info = f"- {col} ({dtype}): {non_null:,} non-null, {unique:,} valeurs uniques"

            # Ajouter des statistiques selon le type
            if pd.api.types.is_numeric_dtype(df[col]):
                if df[col].notna().any():
                    col_info += f" | Min: {df[col].min():.2f}, Max: {df[col].max():.2f}, Moy: {df[col].mean():.2f}"
            elif pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
                if df[col].notna().any() and unique <= 10:
                    top_val = df[col].value_counts().index[0]
                    col_info += f" | Top: {top_val}"

            summary += col_info + "\n"

        if len(df.columns) > 20:
            summary += f"... et {len(df.columns) - 20} autres colonnes\n"

        # Statistiques globales
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        cat_cols = df.select_dtypes(include=['object', 'category']).columns

        summary += f"""
📈 TYPES DE VARIABLES:
- Numériques: {len(numeric_cols)}
- Catégorielles: {len(cat_cols)}
- Valeurs manquantes: {df.isna().sum().sum():,} ({(df.isna().sum().sum() / (len(df) * len(df.columns)) * 100):.1f}%)
"""

        return summary

    def _build_analysis_prompt(self, query: str, data_summary: str, df: pd.DataFrame) -> str:
        """Construit le prompt pour l'analyse"""

        prompt = f"""
# ANALYSE DE DONNÉES - ASSURANCE

## QUESTION DE L'UTILISATEUR:
{query}

## CONTEXTE DES DONNÉES:
{data_summary}

## VOTRE MISSION:
Analysez cette question dans le contexte des données disponibles et fournissez:

1. **COMPRÉHENSION**: Reformulez la question et identifiez l'objectif métier
2. **MÉTHODOLOGIE**: Décrivez l'approche analytique recommandée
3. **INSIGHTS**: 3-5 insights clés basés sur les données disponibles
4. **RECOMMANDATIONS**: Actions concrètes et prochaines étapes

## FORMAT DE RÉPONSE:
Structurez votre réponse en sections claires avec des bullet points.
Soyez concret, actionnable et adapté au domaine de l'assurance.
"""
        return prompt

    def _parse_analysis_result(self, result_text: str, query: str) -> Dict[str, Any]:
        """Parse le résultat de l'analyse"""

        # Structure de base
        result = {
            "comprehension": "",
            "methodologie": "",
            "insights": [],
            "recommandations": [],
            "reponse_detaillee": result_text
        }

        try:
            # Extraction des sections par mots-clés
            lines = result_text.split('\n')
            current_section = None

            for line in lines:
                line_lower = line.lower().strip()

                # Détection des sections
                if any(kw in line_lower for kw in ['compréhension', 'comprendre', 'objectif']):
                    current_section = 'comprehension'
                elif any(kw in line_lower for kw in ['méthodologie', 'méthode', 'approche']):
                    current_section = 'methodologie'
                elif any(kw in line_lower for kw in ['insight', 'constat', 'observation']):
                    current_section = 'insights'
                elif any(kw in line_lower for kw in ['recommandation', 'action', 'prochaine']):
                    current_section = 'recommandations'

                # Ajout du contenu
                if line.strip() and current_section:
                    if line.startswith(('-', '•', '*', '1.', '2.', '3.')):
                        content = line.lstrip('-•*123456789. ').strip()
                        if content:
                            if current_section in ['insights', 'recommandations']:
                                result[current_section].append(content)
                            else:
                                result[current_section] += content + " "

            # Nettoyage
            result['comprehension'] = result['comprehension'].strip() or f"Analyse de: {query}"
            result['methodologie'] = result['methodologie'].strip() or "Analyse descriptive des données"

            # Si pas d'insights extraits, créer des defaults
            if not result['insights']:
                result['insights'] = [
                    "Analyse des données disponibles",
                    "Identification des patterns clés",
                    "Recommandations basées sur les tendances"
                ]

            if not result['recommandations']:
                result['recommandations'] = [
                    "Approfondir l'analyse statistique",
                    "Valider les hypothèses avec des experts métier"
                ]

        except Exception as e:
            print(f"Erreur parsing: {e}")
            # Garder au moins la réponse détaillée

        return result

    def _get_system_prompt(self) -> str:
        """Prompt système pour l'expertise assurance"""
        return """Vous êtes un expert senior en analyse de données pour le secteur de l'assurance automobile.

Votre expertise couvre:
- Tarification et segmentation des risques
- Analyse de sinistres et prévisions
- Optimisation des primes et garanties
- Conformité réglementaire (RGPD, Solvabilité II)
- Data science appliquée à l'assurance

Principes de réponse:
1. Soyez concret et actionnable
2. Utilisez la terminologie métier appropriée
3. Proposez des analyses statistiques rigoureuses
4. Tenez compte des contraintes réglementaires
5. Priorisez la création de valeur business

Répondez toujours en français professionnel avec une structure claire."""

    def process_query(self, prompt: str, max_tokens: int = 800) -> str:
        """
        Traite une requête simple sans contexte de données

        Args:
            prompt: Question ou instruction
            max_tokens: Nombre maximum de tokens dans la réponse

        Returns:
            str: Réponse du modèle
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "Vous êtes un expert en assurance et analyse de données."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=max_tokens,
                timeout=30
            )

            return response.choices[0].message.content

        except Exception as e:
            error_msg = f"Erreur lors du traitement: {str(e)}"
            print(error_msg)
            return error_msg

    def generate_code(self, description: str, language: str = "python") -> str:
        """
        Génère du code basé sur une description

        Args:
            description: Description de ce que le code doit faire
            language: Langage de programmation

        Returns:
            str: Code généré
        """
        try:
            prompt = f"""Générez du code {language} pour: {description}

Exigences:
- Code propre et commenté
- Gestion d'erreurs
- Bonnes pratiques
- Utilisation de pandas/numpy si pertinent

Fournissez uniquement le code, sans explications supplémentaires."""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": f"Vous êtes un expert en programmation {language}."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=1000
            )

            return response.choices[0].message.content

        except Exception as e:
            return f"# Erreur lors de la génération: {str(e)}"


# Alias pour compatibilité avec l'ancien code
AdvancedOpenAIClient = OpenAIAnalyzer