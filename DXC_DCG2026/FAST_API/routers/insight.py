from fastapi import APIRouter, HTTPException, UploadFile, File
from typing import Dict, Any, List
import pandas as pd
import sys
import os
import json

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

try:
    from modules.insight_engine import InsightEngine
except ImportError:
    InsightEngine = None

router = APIRouter()


@router.post("/analyze-dataset")
async def analyze_dataset(file: UploadFile = File(...), analysis_type: str = "full"):
    """Analyser un dataset pour générer des insights"""
    try:
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file.file)
        elif file.filename.endswith('.xlsx'):
            df = pd.read_excel(file.file)
        else:
            raise HTTPException(status_code=400, detail="Format de fichier non supporté")

        insight_engine = InsightEngine()

        # Générer les insights selon le type demandé
        if analysis_type == "statistical":
            insights = insight_engine.generate_statistical_insights(df)
        elif analysis_type == "trend":
            insights = insight_engine.identify_trends(df)
        elif analysis_type == "anomaly":
            insights = insight_engine.detect_anomalies(df)
        else:  # full analysis
            insights = insight_engine.generate_comprehensive_insights(df)

        return {
            "dataset_info": {
                "rows": len(df),
                "columns": list(df.columns),
                "column_types": df.dtypes.astype(str).to_dict()
            },
            "insights": insights
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur d'analyse: {str(e)}")


@router.post("/generate-report")
async def generate_report(report_config: Dict[str, Any]):
    """Générer un rapport d'analyse détaillé"""
    try:
        insight_engine = InsightEngine()

        # Charger les données si fournies
        if 'data' in report_config:
            df = pd.DataFrame(report_config['data'])
        elif 'file_path' in report_config:
            df = pd.read_csv(report_config['file_path'])
        else:
            raise HTTPException(status_code=400, detail="Aucune donnée fournie")

        # Générer le rapport
        report = insight_engine.generate_report(
            df,
            report_type=report_config.get('report_type', 'standard'),
            sections=report_config.get('sections', ['summary', 'trends', 'recommendations'])
        )

        return {
            "report_id": f"report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}",
            "generated_at": pd.Timestamp.now().isoformat(),
            "sections": list(report.keys()),
            "content": report
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de génération: {str(e)}")


@router.get("/insight-templates")
async def get_insight_templates():
    """Récupérer les templates d'insights disponibles"""
    return {
        "templates": [
            {
                "id": "sales_performance",
                "name": "Performance des ventes",
                "description": "Analyse de la performance des ventes",
                "required_columns": ["date", "revenue", "product"]
            },
            {
                "id": "customer_behavior",
                "name": "Comportement client",
                "description": "Analyse du comportement des clients",
                "required_columns": ["customer_id", "purchase_date", "amount"]
            },
            {
                "id": "financial_health",
                "name": "Santé financière",
                "description": "Analyse de la santé financière",
                "required_columns": ["period", "revenue", "expenses", "profit"]
            }
        ]
    }