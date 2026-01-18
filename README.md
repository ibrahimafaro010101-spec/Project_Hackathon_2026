---
output:
  pdf_document: default
  html_document: default
---
# INTELLIGENT ANALYTICS HACKATHON - 2026  
**Team UM6P**

**Auteurs :**  
- Ibrahima FARO[^1]  
- Aya ALAMI[^2]  
- Mariam DIAKITE[^3]  
- Babacar SANDING[^4]  

**Référents de la Team :** El Gargouh Younes & Nadif Firdaouss  

**Titre :** *Profilage et gestion des clients à risque dans une assurance Automobile*  

**Date :** \today  

---

## Table des matières

1.  [Contexte](#contexte)
2.  [Justification du choix du secteur de l'Automobile](#justification-du-choix-du-secteur-de-lautomobile)
3.  [Objectif global](#objectif-global)
4.  [Problématique centrale](#problématique-centrale)
5.  [Ambition](#ambition)
6.  [Données d'entrainement](#données-dentrainement)
7.  [Approche générale et architecture](#approche-générale-et-architecture)
    1.  [Préparation & qualité des données](#préparation--qualité-des-données)
    2.  [Moteur NLQ intelligent (OpenAI)](#moteur-nlq-intelligent-openai)
    3.  [Insight AI](#insight-ai)
    4.  [Modèles prédictifs & scoring client](#modèles-prédictifs--scoring-client)
8.  [Timeline du projet](#timeline-du-projet)
9.  [Choix des outils, technologies et packages](#choix-des-outils-technologies-et-packages)
10. [Références](#références)

---

## Contexte

Ce projet est mis en place par la Team UM6P dans le cadre du Hackathon DXC – Intelligent Analytics. Il vise à démontrer comment l’intelligence artificielle, combinée à l’analyse de données, peut améliorer la gestion des risques, la prise de décision et la performance économique dans le secteur de l’assurance automobile.

## Justification du choix du secteur de l'Automobile

Dans le rapport de sécurité routière Maroc de l'observatoire national de la sécurité routière (@NARSA2022), le choix du secteur de l’assurance automobile est particulièrement solide pour 3 raisons : impact, volume de données, valeur business immédiate.

Le rapport souligne que le Maroc a enregistré 113 625 accidents corporels en 2022, avec 3 499 décès. Ce niveau de mortalité dépassait l’objectif intermédiaire de la stratégie nationale (objectif < 2 643 tués en 2022). Ce résultat permet de remarquer que le secteur de l’automobile présente un risque fréquent et coûteux, parfaitement aligné avec une solution IA « Intelligent Analytics » centrée sur prévention - tarification - pilotage, et surtout dans la proposition des recommandations.

Le rapport insiste sur le fait que des données fiables sont un levier clé pour : comprendre les causes, cibler les facteurs de risque, évaluer l’impact des interventions, et prendre des décisions fondées sur des preuves. C’est exactement ce qu'on compte faire dans ce projet : *transformer des historiques de contrats en indicateurs (prime/jour, loss ratio, risque client, fraude…) puis en décisions.*

Du point de vue du poids économique, le rapport rappelle que les accidents routiers représentent un fardeau économique estimé à environ 3% du PNB des pays et le parc des véhicules de tourisme est très important (44,7% du parc national, et 50,91% des véhicules impliqués dans les accidents corporels sont des véhicules de tourisme). Ainsi, le marché auto offre beaucoup de contrats, d'expositions, donc de données et de gains potentiels (meilleure segmentation, prime, prévention, gestion sinistres).

En Afrique, ces enjeux sont encore plus critiques. Le continent affiche le taux de mortalité routière le plus élevé au monde, avec environ 19,6 décès pour 100 000 habitants, contre une moyenne mondiale d’environ 15 pour 100 000 (OMS, 2023), alors même qu’il ne représente qu’environ 3% du parc automobile mondial. Paradoxalement, l’Afrique connaît une croissance rapide de la motorisation, portée par l’urbanisation, l’essor des classes moyennes et le développement des activités de transport (Banque mondiale, Africa Transport Outlook). Cette dynamique accroît mécaniquement l’exposition au risque automobile, dans des contextes où les infrastructures, les systèmes de contrôle et les mécanismes assurantiels restent souvent insuffisants.

> Nous avons choisi l’assurance automobile car c’est un risque universel, à forte fréquence et à coûts très variables, ce qui en fait un cas d’usage idéal pour l’IA. À l’échelle mondiale, les assureurs cherchent à améliorer la rentabilité technique (loss ratio), accélérer la gestion des sinistres, et mieux piloter la rétention client dans un contexte de concurrence et d’évolution des coûts. En Afrique, la croissance urbaine et l’intensification des usages (flottes, mobilité) rendent l’exposition au risque plus complexe, tandis que la qualité des données peut être hétérogène : une solution d’analytics intelligent qui nettoie, structure et transforme les données en indicateurs actionnables (risque, fraude, paiement, résiliation) apporte un gain immédiat pour la décision métier.

## Objectif global

L’assurance automobile constitue aujourd’hui l’un des segments les plus stratégiques et les plus exposés au risque pour les compagnies d’assurance. La multiplication des sinistres, l’évolution des comportements des assurés et l’intensification de la concurrence rendent la gestion du portefeuille clients de plus en plus complexe. Dans ce contexte, les méthodes traditionnelles d’analyse, souvent statiques et réactives, montrent leurs limites.

Le présent projet vise à exploiter le potentiel de l’intelligence artificielle et de l’analyse avancée des données afin de transformer la manière dont les assureurs identifient, évaluent et gèrent les risques liés à leurs clients. L’objectif est de passer d’une logique de gestion a posteriori à une approche prédictive, proactive et orientée décision métier.

## Problématique centrale

Nous formulons la problématique centrale comme suit :

> Comment anticiper et gérer efficacement les clients à risque dans l’assurance automobile afin de réduire les pertes financières et d’améliorer la qualité de la décision métier grâce à l’intelligence artificielle ?

Cette problématique est au cœur des enjeux actuels du secteur de l’assurance, où la performance dépend de plus en plus de la capacité à anticiper les comportements futurs des assurés plutôt que de simplement constater les événements passés. Elle s’inscrit dans un contexte marqué par :

*   **une sinistralité croissante**, liée à l’augmentation du parc automobile, à la densification urbaine et à l’évolution des comportements de conduite, ce qui exerce une pression directe sur la rentabilité des assureurs.
*   **une résiliation élevée des contrats**, notamment dans l’assurance automobile, où les clients sont de plus en plus volatils et sensibles aux prix, rendant la fidélisation complexe et coûteuse.
*   **une forte asymétrie d’information** entre l’assureur et l’assuré, notamment sur les comportements réels de conduite, les risques latents ou les intentions de résiliation, ce qui complique l’évaluation fine du risque.
*   **des décisions encore largement réactives** plutôt que prédictives fondées sur des règles fixes ou des analyses descriptives, qui interviennent souvent après la survenance du sinistre ou la perte du client, au lieu d’agir en amont.

## Ambition

L’ambition qui nous anime est de renforcer la capacité décisionnelle des acteurs de l’assurance automobile, en leur fournissant des outils capables de détecter les signaux faibles, d’anticiper les risques et d’orienter les stratégies de tarification, de fidélisation et de prévention.

En mettant en place cette application, nous pourrons présenter le produit à des assureurs tout en mettant en avant les ajouts que nous avons réalisés par rapport à la méthode classique.

Nous aspirons à mettre à la disposition des utilisateurs une interface facile, simple, automatisée et basée sur de l'intelligence spécialement entraînée dans le domaine.

## Données d'entrainement

Le projet s’appuie sur un jeu de données issu d’un portefeuille d’assurance automobile, contenant 26 383 lignes correspondant à des enregistrements contractuels.

À l’origine, le jeu de données comprenait **12 variables principales**, décrivant essentiellement :

*   l’identification des contrats et des factures (ex. *num\_contrat*, *Num\_facture*) ;
*   les informations temporelles (*datedeb*, *datefin*, *datcpt*, *exe*) ;
*   les montants de prime (*Prime*) ;
*   la nature de l’opération (*libop*) ;
*   le statut de renouvellement du contrat (*renewed*).

Afin de répondre aux objectifs analytiques et prédictifs du projet, la base a été enrichie par **ingénierie de variables**. Plusieurs variables dérivées ont été construites.

### Variables de tarification et de durée
*   *nb\_jour\_couv*
*   *prime\_par\_jour*
*   *prime\_annualisee*
*   *log\_prime*
*   *anciennete\_contrat\_jours*
*   *anciennete\_client\_en\_jours*

### Variables contractuelles et comportementales
*   *is\_avenant*
*   *is\_affaire\_nouvelle*
*   *is\_terme*
*   *nb\_impayes*
*   *retard\_paiement\_moyen\_jours*

### Variables liées au risque et à la sinistralité
*   *nb\_sinistres\_passe*
*   *cout\_sinistres\_passe*
*   *claim\_frequency*
*   *average\_claim\_cost*
*   *loss\_ratio*
*   *severity\_rate*

### Scores et indicateurs avancés
*   *client\_risk\_score*
*   *client\_profitability*
*   *technical\_margin*

À l’issue de cette phase d’enrichissement, la base finale comprend 44 variables, qui constitue le socle utilisé pour l’analyse descriptive, le moteur NLQ, la génération d’insights et les modèles prédictifs de scoring client et de risque.

## Approche générale et architecture

Le projet repose sur 4 piliers obligatoires :

1.  Préparation & qualité des données
2.  Moteur NLQ intelligent (nous avons choisi OPENIA)
3.  AI insight
4.  Modèles prédictifs & scoring client

et d’autres complémentaires pour le traitement de données intelligent, etc.

### Préparation & qualité des données

La préparation et la qualité des données constituent le premier pilier du projet, car elles conditionnent directement la fiabilité des analyses, des insights et des modèles prédictifs. Dans le secteur de l’assurance automobile, les données sont souvent hétérogènes, issues de plusieurs processus métier (facturation, contrats, renouvellement) et peuvent contenir des incohérences ou des valeurs manquantes.

Cette étape vise d’abord à structurer et fiabiliser les données brutes, en assurant :

*   le nettoyage des valeurs manquantes ou aberrantes ;
*   la mise en cohérence des formats, notamment pour les dates et les montants ;
*   la vérification de la complétude et de la validité des informations contractuelles.

Ensuite, un travail d’ingénierie des variables est réalisé afin de transformer les données brutes en indicateurs exploitables pour l’analyse et la décision. Cela inclut notamment la construction de variables telles que la durée de couverture, la prime par jour, la prime annualisée, l’ancienneté du contrat ou du client, ainsi que des indicateurs liés aux événements contractuels.

L’objectif de ce pilier est de passer d’une base de données purement descriptive à une base orientée décision, capable d’alimenter efficacement le moteur NLQ, le module d’Insight AI et les modèles prédictifs. Une donnée bien préparée permet non seulement d’améliorer la performance des modèles, mais aussi de garantir la cohérence, la traçabilité et la crédibilité des résultats présentés aux décideurs.

### Moteur NLQ intelligent (OpenAI)

#### Révolutionner l'Accès aux Données pour un Avantage Compétitif
Le NLQ (Natural Language Query) Engine est bien plus qu'un simple moteur de recherche ; c'est un catalyseur stratégique. En transformant l'accès complexe aux données en une interaction conversationnelle intuitive, il permet à chaque utilisateur, quel que soit son niveau technique, d'extraire des informations importantes. Cette capacité à interroger vos données et métadonnées en langage naturel se traduit par des gains d'efficacité opérationnelle, une prise de décision accélérée et un avantage concurrentiel indéniable, en transformant les requêtes en insights actionnables.

*   **Accès élargi aux données** : mise à disposition d’outils permettant aux équipes d’interroger les données sans compétences techniques avancées, afin de réduire les dépendances aux équipes spécialisées et d’accélérer l’exploration des informations.
*   **Exploitation des bases de données** : utilisation des données existantes pour produire des analyses statistiques, des indicateurs et des modèles prédictifs, dans le but d’exploiter les informations disponibles de manière plus complète.
*   **Aide à la décision basée sur les données** : production de réponses structurées et contextualisées à partir des données disponibles, afin de soutenir les processus de décision avec des éléments mesurables et vérifiables.

#### Catalyseur de Performance et d'Avantage Compétitif
Le NLQ Engine est stratégiquement conçu pour transformer la manière dont votre organisation interagit avec ses données, en offrant des fonctionnalités clés qui se traduisent directement par un retour sur investissement (ROI) tangible. Il ne se contente pas de répondre aux questions ; il génère des explications contextuelles et des formats de réponse optimisés, essentiels pour une compréhension approfondie et une prise de décision éclairée, propulsant ainsi l'efficacité opérationnelle et la compétitivité.

**Optimisation des Insights et de la Réactivité :**
*   **Synthèses Exécutives Impactantes** : Des réponses courtes et concises pour une identification rapide des tendances critiques et une prise de décision stratégique sans délai.
*   **Analyses Détaillées Approfondies** : Des rapports exhaustifs et nuancés, permettant des diagnostics précis et l'élaboration de stratégies basées sur des données vérifiées.
*   **Démonstrations Interactives** : Visualisez et manipulez les données en temps réel pour une meilleure compréhension des corrélations et de l'identification d'opportunités cachées, accélérant l'adoption des solutions.

Cette capacité unique à intégrer directement vos données et métadonnées assure une flexibilité opérationnelle maximale et une personnalisation sans précédent des requêtes, vous permettant d'adapter précisément l'analyse à vos objectifs business et de découvrir des insights qui garantissent un avantage compétitif durable.

![Architecture de l'application](NLP.png)

#### Sécurité et Confidentialité : Un Avantage Compétitif Stratégique avec le NLQ Engine
La sécurité et la confidentialité des données ne sont plus de simples exigences, mais des piliers fondamentaux pour la croissance et la réputation de votre entreprise. Au cœur de la conception du NLQ Engine réside un engagement ferme à protéger vos actifs informationnels à chaque étape de leur exploitation.

C’est dans cette logique que le NLQ Engine repose sur un principe fondamental : la séparation stricte entre les données sensibles et les données exploitables. Lors de l’utilisation de l’application, deux types d’entrées sont systématiquement définis :

1.  La donnée originale, brute et sensible est fournie initialement au système afin de lui permettre de comprendre le contexte, la structure et la logique métier associée. Cette donnée est utilisée uniquement comme référence de compréhension. Une fois cette phase effectuée, elle est immédiatement isolée, puis supprimée des espaces de traitement actif, afin de garantir la confidentialité et de limiter toute exposition inutile.

    Le système ne conserve pas la donnée brute elle-même, mais uniquement son historique logique et structurel : relations, schémas, dépendances, règles implicites et métadonnées associées. C’est sur cette base que le moteur construit son raisonnement.

    Autrement dit, le chat ne raisonne jamais à partir de la donnée sensible elle-même, mais à partir de l’empreinte informationnelle qu’elle a laissée : sa structure, son organisation, ses relations, ses schémas d’usage.

    Cette approche permet de garantir que :
    *   les informations confidentielles ne sont jamais directement exposées.
    *   la logique métier reste exploitable.
    *   la continuité du raisonnement est assurée.
    *   l’historique conversationnel reste cohérent et pertinent.
    *   la confidentialité est préservée à long terme.

    Ainsi, même après suppression des données sensibles, le système continue à fonctionner efficacement en se basant uniquement sur l’historique logique sécurisé, garantissant à la fois performance, traçabilité et protection des informations critiques.

2.  Une couche de métadonnées, sur laquelle l’ensemble des traitements, analyses et interactions seront effectués.

Cette architecture permet de garantir que les opérations ne s’effectuent jamais directement sur les données critiques, mais uniquement sur leur représentation sécurisée. Ce mécanisme réduit drastiquement les risques de fuite, d’exposition ou de mauvaise manipulation, tout en maintenant une capacité d’analyse complète et performante.

Nous déployons des protocoles de sécurité avancés et des architectures robustes afin de garantir une protection maximale de toutes les informations sensibles. Cette approche assure non seulement une conformité rigoureuse et une réduction significative des risques, mais renforce également la confiance de vos clients et partenaires, transformant la sécurité en un véritable levier de valeur et un avantage compétitif durable.

*   **Protection des Actifs Critiques** : mise en œuvre de mécanismes de sécurité avancés fondés sur l’isolation des données sensibles et leur exploitation indirecte via des métadonnées sécurisées.
*   **Conformité Réglementaire Optimisée** : intégration native des exigences réglementaires (RGPD, HIPAA, etc.) grâce à une architecture qui limite l’exposition directe des données personnelles et sensibles.
*   **Gestion des Accès Sécurisée** : authentification renforcée et gestion fine des privilèges pour chaque interaction.

### Insight AI

#### Débloquez une Valeur Stratégique avec la Visualisation des Données
Insight AI n'est pas seulement une solution d'analyse descriptive, c'est votre avantage concurrentiel. En transformant des volumes massifs de données brutes en visualisations claires et intelligentes, nous permettons à votre organisation d'accélérer la prise de décision, d'optimiser l'efficacité opérationnelle et de maximiser le retour sur investissement. Identifiez les tendances émergentes, anticipez les défis et saisissez des opportunités stratégiques avec une précision sans précédent.

#### De l'Analyse Explicative à la Décision Stratégique
Au-delà de la simple visualisation, Insight AI transforme vos données brutes en une intelligence explicative et actionnable. Notre solution va au-delà des chiffres pour révéler les facteurs clés influençant vos performances, vous offrant une compréhension profonde et prédictive. Cela permet à votre organisation de prendre des décisions éclairées qui maximisent le retour sur investissement, optimisent l'efficacité opérationnelle et forgent un avantage concurrentiel durable.

*   **Intelligence Actionnable** : des explications claires et approfondies des résultats, transformant les données en leviers de performance mesurables.
*   **Optimisation Stratégique** : des insights concrets qui guident vos stratégies clients, garantissant des décisions précises pour une croissance exponentielle.

### Modèles Prédictifs & scoring client

Notre suite intègre des modèles prédictifs de pointe, méticuleusement élaborés pour décrypter les tendances émergentes et anticiper les comportements futurs. En exploitant l'intégralité de vos données historiques, ces modèles délivrent des projections d'une précision inégalée, vous conférant un avantage stratégique déterminant pour une prise de décision proactive, une optimisation des ressources et une croissance durable. C'est l'outil essentiel pour transformer l'incertitude en opportunité, assurer un retour sur investissement tangible et consolider votre position de leader sur le marché.

> « Anticiper l'avenir n'est plus une conjecture, mais une maîtrise stratégique des données pour sculpter votre succès et distancer la concurrence. »

#### Débloquez la Croissance Future : La Mécanique de Nos Modèles Prédictifs
Nos modèles prédictifs de pointe transforment vos données historiques en une feuille de route stratégique. En s'appuyant sur des algorithmes sophistiqués, ils détectent les schémas, corrélations et causalités latentes, offrant des projections précises pour anticiper les ventes, optimiser les stratégies client et mitiger les risques opérationnels.

*   **Analyse des Données Historiques** : collecte rigoureuse et ingénierie des informations passées, constituant la fondation de prévisions fiables et pertinentes.
*   **Application d'Algorithmes ML** : exploitation d'algorithmes d'apprentissage automatique avancés pour déceler les tendances et patterns cachés dans vos données.
*   **Projection Précise des Scénarios** : génération de prévisions hautement précises et de scénarios probabilistes, transformant l'incertitude en opportunité stratégique.
*   **Optimisation de la Planification Stratégique** : traduction des prédictions en actions concrètes, permettant une prise de décision proactive et une performance optimisée.

#### Maximiser Performance et Rentabilité avec l'IA Prédictive
L'intégration stratégique des modèles prédictifs n'est plus un avantage, c'est une nécessité impérative pour toute entreprise visant l'excellence opérationnelle et une croissance soutenue. Notre suite d'IA transforme vos données brutes en informations exploitables, vous permettant d'optimiser radicalement l'allocation de vos ressources, d'affiner vos campagnes marketing pour des résultats sans précédent et d'anticiper les risques pour une meilleure atténuation.

## Timeline du projet

![Agile Project Plan](plan_agile.png)

Notre plan de travail agile, structuré sur un sprint de six jours (du 13 au 19 janvier 2026), est conçu pour garantir la livraison incrémentale d'une solution fonctionnelle à forte valeur métier.

*   **Jour 1 (Cadrage)** : Définition de la vision, architecture technique et interfaces communes. Préparation des données et sélection des algorithmes initiaux.
*   **Jours 2 & 3 (Développement & Intégration)** : Développement parallèle des composants (NLQ, insights, modèles, UI). Intégration pour former un produit minimal viable (MVP) stable.
*   **Jours 4 & 5 (Améliorations & Bonus)** : Ajout de fonctionnalités "Nice to Have" et "Bonus" (contexte conversationnel, alertes, explicabilité, UX avancée).
*   **Jour 6 (Préparation & Présentation)** : Consolidation et préparation d'une présentation percutante mettant en avant la valeur métier.

## Choix des outils, technologies et packages

Nous avons opté pour une stack technologique moderne, légère et orientée data & IA, permettant un développement rapide, modulaire et facilement démontrable.

**Langage principal : Python (3.12-3.14)**
*   Standard en data science et IA.
*   Écosystème riche de bibliothèques.
*   Rapidité de prototypage et facilité d'intégration.

**Environnement de développement : PyCharm Community Edition 2025**

**Bibliothèques clés :**
*   **Traitement des données** : `numpy`, `pandas`
*   **Visualisation** : `plotly`, `seaborn`
*   **ML & Modélisation** : `scikit-learn`, `xgboost`, `joblib`
*   **Interface utilisateur** : `streamlit`
*   **NLQ / IA** : `openai` (API pour ChatGPT-4)

**Sécurité** : L'utilisation de l'API OpenAI est configurée pour fonctionner en local avec une clé API sécurisée, garantissant que les traitements sensibles restent contrôlés.

![Quelques packages utilisés](package.png)

![Génération de la clé API OpenAI](openai.png)

> **Nous vous mettons en copie une démonstration du fonctionnement de l'application (même si cette dernière n'est pas finalisée encore).**

---

## Références

NARSA. (2022). *Rapport de sécurité routière Maroc*. Observatoire National de la Sécurité Routière.

[^1]: <ibrahima.faro@um6p.ma>
[^2]: <aya.alami@um6p.ma>
[^3]: <mariam.diakite@um6p.ma>
[^4]: <babacar.sanding@um6p.ma>
