# modules/__init__.py
# Importations organisées des modules du projet

# Importations principales
from .custom_llm_client import OpenAIAnalyzer, AdvancedOpenAIClient
from .data_prep_engine import DataPrepEngine
from .visualization_generator import VisualizationGenerator
from .predictive_engine import AdvancedPredictiveEngine
from .insight_engine import InsightEngine
from .secure_nlq_engine import NLPEngine, QueryResult
from .report_engine import ReportEngine
from .data_analyzer import DataAnalyzer
from .metadata_extractor import MetadataExtractor
from .business_context import BusinessContextProvider
from .data_processing_engine import DataProcessingEngine


# Importation des configurations
try:
    from .indicators_config import INSURANCE_INDICATORS
except ImportError:
    # Fallback si indicators_config n'existe pas
    INSURANCE_INDICATORS = {
        'demographic': ['age', 'sexe', 'profession'],
        'financial': ['prime', 'sinistre_montant'],
        'risk': ['score_risque', 'nombre_sinistres'],
        'temporal': ['date_effet', 'date_echeance']
    }

# Énumération des modules disponibles
AVAILABLE_MODULES = {
    'llm': ['OpenAIAnalyzer', 'AdvancedOpenAIClient'],
    'data_prep': ['DataPrepEngine'],
    'processing': ['DataProcessingEngine'],
    'visualization': ['VisualizationGenerator'],
    'predictive': ['AdvancedPredictiveEngine'],
    'insight': ['InsightEngine'],
    'nlq': ['NLPEngine', 'QueryResult'],
    'report': ['ReportEngine'],
    'analysis': ['DataAnalyzer'],
    'metadata': ['MetadataExtractor'],
    'context': ['BusinessContextProvider'],
}

__all__ = [
    # Clients LLM
    'OpenAIAnalyzer',
    'AdvancedOpenAIClient',

    # Moteurs de données
    'DataPrepEngine',
    'DataProcessingEngine',

    # Analyse et visualisation
    'DataAnalyzer',
    'VisualizationGenerator',

    # Modèles prédictifs
    'AdvancedPredictiveEngine',

    # Insights et métadonnées
    'InsightEngine',
    'MetadataExtractor',
    'BusinessContextProvider',

    # NLP et rapports
    'NLPEngine',
    'QueryResult',
    'ReportEngine',

    # Configurations
    'INSURANCE_INDICATORS',

    # Fonctions utilitaires
    'DataProcessingEngine',
]


# Fonction pour vérifier la disponibilité des modules
def check_module_availability():
    """Vérifie la disponibilité de tous les modules"""
    available = {}
    missing = {}

    for module_name in __all__:
        try:
            # Essayer d'importer le module
            exec(f"import modules.{module_name.lower()} as test_module", globals())
            available[module_name] = True
        except ImportError:
            available[module_name] = False
            missing[module_name] = f"Module {module_name} non disponible"
        except Exception as e:
            available[module_name] = False
            missing[module_name] = str(e)

    return {
        'available': available,
        'missing': missing,
        'total_modules': len(__all__),
        'available_count': sum(available.values()),
        'missing_count': len(missing)
    }


# Fonction pour initialiser tous les modules
def initialize_modules(config=None):
    """Initialise tous les modules avec une configuration optionnelle"""
    modules = {}

    # Configuration par défaut
    default_config = {
        'openai_api_key': None,
        'data_processing_engine': {},
        'visualization': {'theme': 'plotly_white'},
        'predictive': {'random_state': 42}
    }

    config = config or default_config

    try:
        # Initialiser DataPrepEngine
        modules['data_prep'] = DataPrepEngine()

        # Initialiser DataProcessingEngine
        modules['data_processing_engine'] = DataProcessingEngine(config.get('data_processing_engine'))

        # Initialiser DataAnalyzer
        modules['analyzer'] = DataAnalyzer()

        # Initialiser VisualizationGenerator
        modules['visualization'] = VisualizationGenerator()

        # Initialiser PredictiveEngine (si clé API disponible)
        if config.get('openai_api_key'):
            modules['predictive'] = AdvancedPredictiveEngine()

        # Initialiser InsightEngine
        modules['insight'] = InsightEngine()

        # Initialiser ReportEngine
        modules['report'] = ReportEngine()

        print(f"✅ Modules initialisés: {list(modules.keys())}")

    except Exception as e:
        print(f"⚠️ Erreur lors de l'initialisation des modules: {e}")

    return modules


# Version du package
__version__ = '1.0.0'
__author__ = 'LIK Insurance Analyst Team (UM6P)'
__description__ = 'Modules de traitement et d\'analyse de données pour l\'assurance'

# Message d'importation
print(f"✅ modules/__init__.py chargé - Version {__version__}")
print(f"📦 {len(__all__)} modules disponibles")