# Project_Hackathon_2026
*DXC DCG HACKATHON PROJECT 2026*


### RenewAI – Intelligent Insurance Analytics Platform
#### Contexte du projet

Ce projet a été développé dans le cadre du Hackathon DXC – Intelligent Analytics.
Il vise à démontrer comment l’intelligence artificielle, combinée à l’analyse de données métier, peut améliorer la gestion des risques, la prise de décision et la performance économique dans le secteur de l’assurance automobile.

### Objectif du projet

###### Problématique centrale :

*Comment anticiper et gérer les clients à risque afin de réduire les pertes et améliorer la décision métier grâce à l’IA ?*

###### Objectifs opérationnels

* Identifier les clients à risque de résiliation ou de sinistralité élevée

* Automatiser l’analyse des données d’assurance

* Fournir des insights métier exploitables

* Offrir un assistant intelligent en langage naturel pour les décideurs

* Aider à la prise de décision stratégique (tarification, fidélisation, prévention fraude)

### Approche générale

###### Le projet repose sur 4 piliers complémentaires :

*  Préparation & qualité des données

*  Moteur NLP intelligent (IA / ChatGPT)

*  Modèles prédictifs & scoring client

*  Dashboard décisionnel interactif

### Architecture du projet

DXC_DCG2026/
################################
# Architecture détectée de base#
################################

hackathon_dashboard/
│
├── app.py                          # Application principale
├── requirements.txt                # Dépendances
│
├── modules/
│   ├── nlp_engine.py              # Moteur NLQ
│   ├── data_prep_engine.py        # Préparation des données
│   ├── predictive_engine.py       # Modèles de prédiction
│   └── insight_engine.py          # Génération d'insights
│
├── data/
│   └── sample_policies.csv        # Données exemple
│
└── assets/
    └── style.css                  (optionnel)

################################
# Architecture jour 1          #
################################

DXC_DCG2026/
│
├── app.py                       # Entrée Streamlit (navigation + orchestration)
├── requirements.txt
├── README.md                    # Comment lancer + démo + données attendues
├── .env.example                 # Exemple variables d’environnement (PAS de clé dedans)
│
│
├── Archive/                     # Pour la documentation des traveaux
│    ├── architecture_project.txt/
│
│
├── data/
│   ├── raw/                     # Données brutes uploadées (optionnel)
│   ├── processed/               # Données préparées (export)
│   ├── Data_set_Hackathon_FINAL_ASSURANCE_AUTO.xlsx
│   └── Data_set_Hackathon_FINAL_ASSURANCE_AUTO_SIMULE.xlsx
│
├── assets/
│   ├── style.css                # Thème (Arial Black, bleu, etc.)
│   ├── logo.png                 # Ton logo (local)
│   └── icon.png                 # Petit favicon/icone (optionnel)
│
├── modules/
│   ├── __init__.py
│   ├── indicators_config.py     # Référentiel indicateurs + variables (métadonnées)
│   ├── llm_client.py            # User OpenAI (API key via env) (on va un peu revoir cette partie)
│   ├── nlp_engine.py            # NLP: LLM + fallback regex + JSON strict (on a choisit CHATGPT comme moteur de recherche)
│   ├── data_prep_engine.py      # Préparation, cleaning et exportation vers autres fénêtres
│   ├── predictive_engine.py     # RF/Logistic (safe), train/evaluate/predict (Ici, on aura besoin de votre advice sur le choix des modèles)
│   ├── insight_engine.py        # Insights + graphes Plotly (si possible, on vera powerBI)
│   ├── report.py                # Sur la base de l'IA, On veut donner la possibilité à l'user de rédiger directement un rapport (PDF, Docx) ou de façon automatique
│   ├── validators.py            # (optionnel) contrôle schéma colonnes/types
│   └── ui_components.py         # (optionnel) composants UI réutilisables
│
│
├── .env                         # Contenant le API key
│
├── Dockerfile/                  # Permettant de deployer l'Appli
│
└── notebooks/                   # (optionnel) explorations EDA (hors app)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
NB : Nous comptons mettre des sous parties dans le modules insight_engine.py (en analyse (+ tests) univariées, multivariée, multidimentionnelle) %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

### Données utilisées

Le jeu de données représente des contrats d’assurance automobile avec :

informations contractuelles

dates de couverture

primes

opérations (avenant, affaire nouvelle, terme)

clients (anonymisés)

Enrichissement des données

Le projet génère automatiquement :

indicateurs de risque

indicateurs comportementaux

indicateurs financiers

variables explicatives pour les modèles IA

### Indicateurs métier intégrés
🔴 Risque & sinistralité

Fréquence de sinistre

Coût moyen de sinistre

Loss Ratio

Taux de gravité

Score de risque client

👤 Comportement client

Retard de paiement (jours)

Nombre d’impayés

Ancienneté du contrat

Nombre de renouvellements

Taux de résiliation

💰 Finance

Prime annuelle / prime nette

Rentabilité client

Marge technique

Valeur vie client (CLV)

🚨 Fraude (détection)

Sinistres rapprochés dans le temps

Déclaration rapide après souscription

Montant anormalement élevé

Répétition de dommages similaires

🤖 Moteur NLP intelligent (IA)

Le projet intègre un assistant en langage naturel capable de :

Comprendre des questions métier en français

Identifier l’intention (risque, renouvellement, sinistre, fraude…)

Extraire les entités (contrat, montant, période)

Associer les indicateurs pertinents

Générer une réponse métier explicable

Exemple de requêtes
• Quel est le risque de résiliation du contrat 16122 ?
• Quels sont les clients à forte sinistralité ?
• Quelle est la rentabilité de ce client ?
• Détecte-t-on un risque de fraude ?


### Le moteur fonctionne :

avec ChatGPT (mode principal)

avec un fallback regex (mode sécurisé hackathon)

### Modélisation & scoring

Modèles utilisés :

Régression logistique

Random Forest

###### Variables cibles possibles :

renouvellement

risque de résiliation

###### Sorties :

Probabilité de risque

Score client normalisé

Interprétation métier

### Dashboard décisionnel

Le dashboard Streamlit propose :

Vue exécutive (KPI clés)

Analyse financière

Analyse des contrats

Insights IA automatisés

Qualité des données

Assistant IA interactif

 Robustesse & sécurité

Fonctionnement possible sans API OpenAI

Données clients anonymisées

Architecture modulaire et extensible

Séparation claire Data / IA / UI

### Installation & exécution
### Installation des dépendances
pip install -r requirements.txt

### Lancer l’application
streamlit run app.py

### (Optionnel) Activer ChatGPT
export OPENAI_API_KEY="votre_cle_api"

### Équipe projet (Hackathon)

Data & Feature Engineering

IA & NLP

Modélisation prédictive

Dashboard & Business

Organisation inspirée d’une équipe produit data professionnelle

### Conclusion

RenewAI démontre comment l’IA peut transformer les données d’assurance en décisions intelligentes, rapides et explicables, au service :

de la rentabilité

de la gestion du risque

de l’expérience client
