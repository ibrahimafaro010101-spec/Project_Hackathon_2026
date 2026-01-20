from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
import sys
import os
import json

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

try:
    from modules.nlp_engine import NlpEngine
except ImportError:
    NlpEngine = None

router = APIRouter()


class NLQRequest(BaseModel):
    query: str
    context: Optional[Dict[str, Any]] = None
    data_source: Optional[str] = None


class NLQResponse(BaseModel):
    query: str
    interpretation: str
    sql_query: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    visualization_suggestion: Optional[str] = None


@router.post("/process-query", response_model=NLQResponse)
async def process_nlq(request: NLQRequest):
    """Traiter une requête en langage naturel"""
    try:
        if NlpEngine is None:
            # Mode simulation pour le développement
            return simulate_nlq_response(request.query)

        nlp_engine = NlpEngine()

        # Traiter la requête
        result = nlp_engine.process_query(
            request.query,
            context=request.context,
            data_source=request.data_source
        )

        return NLQResponse(
            query=request.query,
            interpretation=result.get('interpretation', ''),
            sql_query=result.get('sql_query'),
            result=result.get('result'),
            visualization_suggestion=result.get('visualization_suggestion')
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de traitement NLQ: {str(e)}")


def simulate_nlq_response(query: str) -> NLQResponse:
    """Simuler une réponse NLQ pour le développement"""
    query_lower = query.lower()

    # Détection du type de requête
    if "vente" in query_lower or "revenue" in query_lower:
        interpretation = "Analyse des performances de vente"
        sql_query = "SELECT SUM(revenue) as total_ventes, AVG(revenue) as moyenne_ventes FROM sales WHERE date >= '2024-01-01'"
        result = {"total_ventes": 150000, "moyenne_ventes": 5000}
        viz_suggestion = "line_chart"

    elif "client" in query_lower or "customer" in query_lower:
        interpretation = "Analyse du comportement client"
        sql_query = "SELECT COUNT(DISTINCT customer_id) as nb_clients, AVG(purchase_amount) as panier_moyen FROM purchases"
        result = {"nb_clients": 1250, "panier_moyen": 145.50}
        viz_suggestion = "bar_chart"

    elif "tendance" in query_lower or "trend" in query_lower:
        interpretation = "Identification des tendances"
        sql_query = "SELECT MONTH(date) as mois, SUM(revenue) as ventes_mensuelles FROM sales GROUP BY MONTH(date)"
        result = {"tendance": "croissante", "taux_croissance": "15%"}
        viz_suggestion = "line_chart"

    else:
        interpretation = "Requête générale d'analyse"
        sql_query = "SELECT COUNT(*) as nb_lignes FROM dataset"
        result = {"nb_lignes": 10000}
        viz_suggestion = "table"

    return NLQResponse(
        query=query,
        interpretation=interpretation,
        sql_query=sql_query,
        result=result,
        visualization_suggestion=viz_suggestion
    )


@router.get("/query-examples")
async def get_query_examples():
    """Récupérer des exemples de requêtes NLQ"""
    return {
        "examples": [
            {
                "query": "Quelles sont les ventes totales du dernier trimestre ?",
                "category": "sales",
                "difficulty": "beginner"
            },
            {
                "query": "Montre-moi l'évolution des clients actifs par mois",
                "category": "customers",
                "difficulty": "intermediate"
            },
            {
                "query": "Identifie les produits avec la marge bénéficiaire la plus faible",
                "category": "products",
                "difficulty": "advanced"
            },
            {
                "query": "Compare les performances entre la région Nord et Sud",
                "category": "regional",
                "difficulty": "intermediate"
            }
        ]
    }