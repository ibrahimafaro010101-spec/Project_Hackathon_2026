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
│
├── app.py                       # Application Streamlit principale
│
├── modules/
│   ├── data_prep_engine.py      # Nettoyage & feature engineering
│   ├── nlp_engine.py            # Moteur NLP (ChatGPT + fallback)
│   ├── llm_client.py            # Client OpenAI / ChatGPT
│   ├── predictive_engine.py     # Modèles prédictifs & scoring
│   ├── insight_engine.py        # Génération d’insights & rapports
│   └── indicators_config.py     # Référentiel des indicateurs métier
│
├── data/
│   ├── Data_set_Hackathon.xlsx
│   └── Data_set_Hackathon_FINAL_ASSURANCE_AUTO.xlsx
│
├── assets/
│   ├── style.css                # Design & thème
│   └── logo.png                 # Logo du projet
│
├── README.md                    # Documentation du projet
└── requirements.txt             # Dépendances Python

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
