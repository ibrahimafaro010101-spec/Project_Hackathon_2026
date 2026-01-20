from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import json
import sys
import os

# Ajouter le chemin des modules
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

try:
    from modules.predictive_engine import PredictiveEngine
    from modules.data_prep_engine import DataPrepEngine
except ImportError:
    # Fallback pour le développement
    PredictiveEngine = None
    DataPrepEngine = None

router = APIRouter()


@router.post("/upload-dataset")
async def upload_dataset(file: UploadFile = File(...)):
    """Télécharger un dataset pour analyse prédictive"""
    try:
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file.file)
        elif file.filename.endswith('.xlsx'):
            df = pd.read_excel(file.file)
        else:
            raise HTTPException(status_code=400, detail="Format de fichier non supporté")

        # Sauvegarder temporairement
        temp_path = f"data/temp_{file.filename}"
        os.makedirs("data", exist_ok=True)
        df.to_csv(temp_path, index=False)

        return {
            "filename": file.filename,
            "rows": len(df),
            "columns": list(df.columns),
            "temp_path": temp_path
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors du traitement: {str(e)}")


@router.post("/train-model")
async def train_model(config: Dict[str, Any]):
    """Entraîner un modèle prédictif"""
    try:
        # Charger les données
        data_path = config.get("data_path")
        if not data_path or not os.path.exists(data_path):
            raise HTTPException(status_code=404, detail="Fichier de données non trouvé")

        df = pd.read_csv(data_path)

        # Préparer les données
        prep_engine = DataPrepEngine()
        prepared_data = prep_engine.prepare_data(df, config)

        # Entraîner le modèle
        predictive_engine = PredictiveEngine()
        model_result = predictive_engine.train_model(
            prepared_data['X_train'],
            prepared_data['y_train'],
            prepared_data['X_test'],
            prepared_data['y_test'],
            config.get("model_type", "random_forest")
        )

        return {
            "accuracy": float(model_result['accuracy']),
            "precision": float(model_result['precision']),
            "recall": float(model_result['recall']),
            "model_info": model_result['model_info']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur d'entraînement: {str(e)}")


@router.post("/predict")
async def make_prediction(prediction_data: Dict[str, Any]):
    """Faire une prédiction avec le modèle entraîné"""
    try:
        predictive_engine = PredictiveEngine()

        # Charger le modèle si nécessaire
        if 'model_path' in prediction_data:
            predictive_engine.load_model(prediction_data['model_path'])

        # Préparer les features
        features = np.array(prediction_data['features']).reshape(1, -1)

        # Faire la prédiction
        prediction = predictive_engine.predict(features)

        return {
            "prediction": prediction.tolist(),
            "confidence": predictive_engine.get_confidence(features).tolist() if hasattr(predictive_engine,
                                                                                         'get_confidence') else None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction: {str(e)}")


@router.get("/model-metrics/{model_id}")
async def get_model_metrics(model_id: str):
    """Récupérer les métriques d'un modèle"""
    # Implémentation fictive - à adapter
    return {
        "model_id": model_id,
        "accuracy": 0.85,
        "precision": 0.83,
        "recall": 0.82,
        "f1_score": 0.825
    }