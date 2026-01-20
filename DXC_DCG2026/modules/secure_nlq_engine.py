# modules/secure_nlq_engine.py - Version Finale avec Génération de Graphiques
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from openai import OpenAI
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import traceback
import warnings

warnings.filterwarnings('ignore')


class SecureNLQEngine:
    """
    Moteur NLQ (Natural Language Query) sécurisé avec génération de graphiques
    Analyse les requêtes utilisateur et génère des visualisations pertinentes
    """

    def __init__(self, api_key: str, model: str = "gpt-4"):
        """
        Initialise le moteur NLQ

        Args:
            api_key: Clé API OpenAI
            model: Modèle à utiliser (gpt-4, gpt-3.5-turbo, etc.)
        """
        if not api_key or api_key.strip() == "":
            raise ValueError("Clé API OpenAI requise et non vide")

        try:
            self.client = OpenAI(api_key=api_key.strip())
            self.model = model
            self.metadata_cache = {}
            print(f"✅ Moteur NLQ initialisé avec le modèle: {model}")
        except Exception as e:
            raise ValueError(f"Erreur d'initialisation OpenAI: {e}")

    def analyze_query_with_metadata(self, user_query: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyse une requête utilisateur avec les métadonnées uniquement (sécurisé)

        Args:
            user_query: Question de l'utilisateur
            metadata: Métadonnées des données (structure, types, etc.)

        Returns:
            Dict avec les résultats d'analyse
        """
        if not user_query or user_query.strip() == "":
            return {"error": "Requête vide", "status": "error"}

        try:
            print(f"🔍 Analyse sécurisée de: {user_query[:100]}...")

            # Construire le prompt avec métadonnées
            prompt = self._build_metadata_prompt(user_query, metadata)

            # Analyser avec le LLM
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000,
                timeout=60
            )

            result_text = response.choices[0].message.content

            # Parser la réponse
            try:
                analysis = json.loads(result_text)
            except json.JSONDecodeError:
                analysis = self._parse_text_response(result_text)

            return {
                "status": "success",
                "query": user_query,
                "analysis": analysis,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            error_msg = f"Erreur lors de l'analyse: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            return {
                "status": "error",
                "error": error_msg,
                "query": user_query,
                "timestamp": datetime.now().isoformat()
            }

    def analyze_query_with_data(self, user_query: str, dataframe: pd.DataFrame,
                               max_samples: int = 1000) -> Dict[str, Any]:
        """
        Analyse une requête avec les données réelles et génère des graphiques

        Args:
            user_query: Question de l'utilisateur
            dataframe: DataFrame contenant les données
            max_samples: Nombre maximum d'échantillons pour l'analyse

        Returns:
            Dict avec analyse et graphiques générés
        """
        if not user_query or user_query.strip() == "":
            return {"error": "Requête vide", "status": "error"}

        if dataframe is None or dataframe.empty:
            return {"error": "Aucune donnée disponible", "status": "error"}

        try:
            print(f"🔍 Analyse avec données: {user_query[:100]}...")

            # Échantillonner les données
            df_sample = self._sample_dataframe(dataframe, max_samples)

            # Générer métadonnées
            metadata = self._generate_metadata_from_data(df_sample)

            # Analyser la requête
            analysis_result = self._analyze_with_llm(user_query, metadata, df_sample)

            # Générer les graphiques
            graphs = self._generate_requested_graphs(analysis_result, df_sample)

            # Créer le rapport complet
            full_report = self._create_comprehensive_report(analysis_result, graphs, df_sample)

            return {
                "status": "success",
                "query": user_query,
                "analysis": analysis_result,
                "graphs": graphs,
                "report": full_report,
                "sample_info": {
                    "original_rows": len(dataframe),
                    "sampled_rows": len(df_sample),
                    "columns": len(dataframe.columns),
                    "timestamp": datetime.now().isoformat()
                }
            }

        except Exception as e:
            error_msg = f"Erreur lors de l'analyse: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            return {
                "status": "error",
                "error": error_msg,
                "query": user_query,
                "timestamp": datetime.now().isoformat()
            }

    def _sample_dataframe(self, df: pd.DataFrame, max_samples: int = 1000) -> pd.DataFrame:
        """Échantillonne le DataFrame intelligemment"""
        if len(df) <= max_samples:
            return df.copy()

        try:
            # Échantillonnage stratifié si possible
            cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

            if cat_cols:
                main_cat_col = max(cat_cols, key=lambda col: df[col].nunique())
                df_sampled = df.groupby(main_cat_col, group_keys=False).apply(
                    lambda x: x.sample(min(len(x), max(1, int(max_samples * len(x) / len(df)))))
                ).reset_index(drop=True)

                if len(df_sampled) > max_samples:
                    df_sampled = df_sampled.sample(n=max_samples, random_state=42)

                return df_sampled
        except:
            pass

        # Fallback: échantillonnage aléatoire
        return df.sample(n=min(max_samples, len(df)), random_state=42)

    def _generate_metadata_from_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Génère des métadonnées complètes à partir des données"""
        structure_columns = []

        for col in df.columns:
            col_info = {
                'nom': col,
                'type_donnee': str(df[col].dtype),
                'valeurs_uniques': int(df[col].nunique()),
                'valeurs_manquantes': int(df[col].isna().sum()),
                'pourcentage_manquants': float((df[col].isna().sum() / len(df)) * 100)
            }

            # Statistiques selon le type
            if pd.api.types.is_numeric_dtype(df[col]):
                col_info['est_numerique'] = True
                if df[col].notna().any():
                    col_info.update({
                        'min': float(df[col].min()),
                        'max': float(df[col].max()),
                        'moyenne': float(df[col].mean()),
                        'ecart_type': float(df[col].std())
                    })

            elif pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
                col_info['est_categorielle'] = True
                if df[col].notna().any():
                    top_values = df[col].value_counts().head(5).to_dict()
                    col_info['top_valeurs'] = {str(k): int(v) for k, v in top_values.items()}

            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                col_info['est_temporelle'] = True

            structure_columns.append(col_info)

        metadata = {
            'structure_columns': structure_columns,
            'general_info': {
                'nombre_colonnes': len(df.columns),
                'nombre_lignes': len(df),
                'pourcentage_manquants_global': float((df.isna().sum().sum() / (len(df) * len(df.columns))) * 100)
            },
            'business_context_hints': {
                'domaine': self._detect_domain(df.columns),
                'variables_cles': self._identify_key_variables(df.columns)
            },
            'statistical_summary': {
                'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
                'categorical_columns': df.select_dtypes(include=['object', 'category']).columns.tolist(),
                'date_columns': df.select_dtypes(include=['datetime64']).columns.tolist()
            }
        }

        return metadata

    def _detect_domain(self, columns) -> str:
        """Détecte le domaine métier"""
        col_names = ' '.join([str(col).lower() for col in columns])

        if any(kw in col_names for kw in ['prime', 'assur', 'sinistre', 'risque', 'police']):
            return 'assurance'
        elif any(kw in col_names for kw in ['compte', 'transaction', 'credit', 'debit']):
            return 'finance'
        elif any(kw in col_names for kw in ['produit', 'prix', 'commande', 'vente']):
            return 'retail'

        return 'general'

    def _identify_key_variables(self, columns) -> List[str]:
        """Identifie les variables clés"""
        key_vars = []
        priority_keywords = ['id', 'nom', 'date', 'montant', 'prix', 'score', 'risque', 'prime', 'client']

        for col in columns:
            if any(kw in str(col).lower() for kw in priority_keywords):
                key_vars.append(str(col))

        return key_vars[:10]

    def _build_metadata_prompt(self, user_query: str, metadata: Dict[str, Any]) -> str:
        """Construit le prompt avec métadonnées uniquement"""
        columns_info = "## Colonnes disponibles:\n"
        for col_info in metadata.get('structure_columns', [])[:20]:
            col_line = f"- {col_info['nom']} ({col_info['type_donnee']})"
            if col_info.get('est_numerique'):
                col_line += f" [Numérique]"
            elif col_info.get('est_categorielle'):
                col_line += f" [Catégorielle, {col_info.get('valeurs_uniques', 0)} valeurs]"
            columns_info += col_line + "\n"

        prompt = f"""
# ANALYSE NLQ - MÉTADONNÉES UNIQUEMENT

## QUESTION:
{user_query}

## MÉTADONNÉES:
{columns_info}

Domaine: {metadata.get('business_context_hints', {}).get('domaine', 'general')}
Nombre de lignes: {metadata.get('general_info', {}).get('nombre_lignes', 0):,}
Nombre de colonnes: {metadata.get('general_info', {}).get('nombre_colonnes', 0)}

## VOTRE MISSION:
Analysez cette question et proposez une stratégie d'analyse basée UNIQUEMENT sur les métadonnées.

Répondez au format JSON:
{{
    "intention": "Objectif de la question",
    "strategie_analyse": "Méthodologie recommandée",
    "visualisations_suggestees": [
        {{"type": "histogram|scatter|bar", "variables": ["col1"], "description": "..."}}
    ],
    "insights_cles": ["Insight 1", "Insight 2"],
    "reponse_detaillee": "Réponse complète en français"
}}
"""
        return prompt

    def _analyze_with_llm(self, user_query: str, metadata: Dict[str, Any], df_sample: pd.DataFrame) -> Dict[str, Any]:
        """Analyse avec le LLM"""
        prompt = self._build_analysis_prompt(user_query, metadata, df_sample)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000,
                timeout=60
            )

            result_text = response.choices[0].message.content

            try:
                return json.loads(result_text)
            except json.JSONDecodeError:
                return self._parse_text_response(result_text)

        except Exception as e:
            print(f"Erreur LLM: {e}")
            return self._create_fallback_analysis(df_sample)

    def _build_analysis_prompt(self, user_query: str, metadata: Dict[str, Any], df_sample: pd.DataFrame) -> str:
        """Construit le prompt complet pour l'analyse"""
        data_summary = f"""
Données: {len(df_sample):,} lignes, {len(df_sample.columns)} colonnes
Domaine: {metadata['business_context_hints']['domaine']}
"""

        columns_info = "Colonnes:\n"
        for col_info in metadata['structure_columns'][:15]:
            columns_info += f"- {col_info['nom']} ({col_info['type_donnee']})\n"

        prompt = f"""
# ANALYSE DE DONNÉES

## QUESTION:
{user_query}

## CONTEXTE:
{data_summary}

{columns_info}

Proposez une analyse complète avec graphiques réalisables.

Format JSON:
{{
    "intention": "...",
    "strategie_analyse": "...",
    "graphiques_recommandes": [
        {{"type": "histogram", "variables": ["col1"], "description": "...", 
          "parametres": {{"x": "col1"}}}}
    ],
    "insights_cles": ["..."],
    "reponse_detaillee": "..."
}}
"""
        return prompt

    def _parse_text_response(self, text_response: str) -> Dict[str, Any]:
        """Parse une réponse texte"""
        return {
            "intention": "Analyse de données",
            "strategie_analyse": text_response[:500],
            "graphiques_recommandes": [],
            "insights_cles": ["Analyse effectuée"],
            "reponse_detaillee": text_response
        }

    def _create_fallback_analysis(self, df_sample: pd.DataFrame) -> Dict[str, Any]:
        """Crée une analyse par défaut"""
        numeric_cols = df_sample.select_dtypes(include=[np.number]).columns.tolist()

        return {
            "intention": "Analyse exploratoire",
            "strategie_analyse": "Analyse descriptive des données disponibles",
            "graphiques_recommandes": [
                {
                    "type": "histogram",
                    "variables": numeric_cols[:1] if numeric_cols else [],
                    "description": "Distribution de la variable principale"
                }
            ],
            "insights_cles": ["Données chargées avec succès"],
            "reponse_detaillee": f"Analyse sur {len(df_sample)} lignes."
        }

    def _generate_requested_graphs(self, analysis_result: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
        """Génère les graphiques demandés"""
        graphs = {"generated": [], "errors": []}

        for i, graph_info in enumerate(analysis_result.get('graphiques_recommandes', [])[:4]):
            try:
                graph_type = graph_info.get('type', '').lower()
                variables = graph_info.get('variables', [])
                params = graph_info.get('parametres', {})

                # Valider les variables
                variables = [v for v in variables if v in df.columns]
                if not variables:
                    continue

                # Créer le graphique
                fig = self._create_plotly_figure(df, graph_type, variables, params)

                if fig:
                    graphs["generated"].append({
                        "id": f"graph_{i+1}",
                        "type": graph_type,
                        "variables": variables,
                        "html": fig.to_html(full_html=False, include_plotlyjs='cdn'),
                        "description": graph_info.get('description', '')
                    })

            except Exception as e:
                graphs["errors"].append({"graph_info": graph_info, "error": str(e)})

        return graphs

    def _create_plotly_figure(self, df: pd.DataFrame, graph_type: str,
                             variables: List[str], params: Dict[str, Any]) -> Optional[go.Figure]:
        """Crée un graphique Plotly"""
        try:
            if graph_type == 'histogram' and len(variables) >= 1:
                return px.histogram(df, x=variables[0], title=f"Distribution de {variables[0]}")

            elif graph_type == 'scatter' and len(variables) >= 2:
                return px.scatter(df, x=variables[0], y=variables[1],
                                title=f"{variables[0]} vs {variables[1]}")

            elif graph_type == 'bar' and len(variables) >= 1:
                if df[variables[0]].dtype in ['object', 'category']:
                    value_counts = df[variables[0]].value_counts().head(10)
                    return px.bar(x=value_counts.index, y=value_counts.values,
                                title=f"Top 10 - {variables[0]}")

            elif graph_type == 'box' and len(variables) >= 1:
                return px.box(df, y=variables[0], title=f"Boîte à moustaches - {variables[0]}")

        except Exception as e:
            print(f"Erreur graphique: {e}")
            return None

    def _create_comprehensive_report(self, analysis_result: Dict[str, Any],
                                    graphs: Dict[str, Any], df_sample: pd.DataFrame) -> Dict[str, Any]:
        """Crée le rapport complet"""
        return {
            "summary": {
                "graphs_generated": len(graphs.get("generated", [])),
                "sample_size": len(df_sample),
                "variables_analyzed": len(df_sample.columns)
            },
            "analysis_details": {
                "strategie": analysis_result.get("strategie_analyse", ""),
                "insights": analysis_result.get("insights_cles", [])
            }
        }

    def _get_system_prompt(self) -> str:
        """Prompt système"""
        return """Vous êtes un expert en analyse de données et visualisation pour l'assurance.

Mission: Analyser les questions et proposer des visualisations pertinentes.

Règles:
- Utilisez les noms de colonnes EXACTEMENT comme fournis
- Proposez des graphiques réalisables
- Soyez concret et actionnable
- Répondez en JSON strict ou en français structuré"""