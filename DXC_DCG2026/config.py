# config.py - Configuration API invisible
import os
from dotenv import load_dotenv

# Charger le fichier .env
load_dotenv()

# Configuration API
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Vérification silencieuse
API_CONFIGURED = bool(OPENAI_API_KEY and OPENAI_API_KEY.strip())

# Autres configurations
MODEL = os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1000"))
TIMEOUT = int(os.getenv("TIMEOUT", "30"))

# Configuration de l'application
APP_NAME = "LIK Insurance Analyst" # On a préféré l'appeler ainsi
APP_VERSION = "2.0"
DEBUG_MODE = os.getenv("DEBUG", "False").lower() == "true"