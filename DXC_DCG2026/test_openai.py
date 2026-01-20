# test_openai.py
import sys
import os

# Ajouter le chemin des modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "modules"))


def test_openai_client():
    """Teste le client OpenAI"""
    try:
        from custom_llm_client import OpenAIAnalyzer

        # Demander la clé API
        api_key = input("Entrez votre clé API OpenAI: ").strip()

        if not api_key:
            print("❌ Aucune clé API fournie")
            return

        # Créer un client
        client = OpenAIAnalyzer(api_key=api_key)
        print("✅ Client OpenAI créé")

        # Créer des données de test
        import pandas as pd
        import numpy as np

        df = pd.DataFrame({
            'age_conducteur': np.random.randint(18, 70, 100),
            'Prime': np.random.uniform(500, 2000, 100),
            'nb_sinistres': np.random.randint(0, 5, 100),
            'type_vehicule': np.random.choice(['Citadine', 'Berline', 'SUV', 'Utilitaire'], 100),
            'anciennete_permis': np.random.randint(1, 40, 100)
        })

        print(f"📊 Données de test: {len(df)} lignes, {len(df.columns)} colonnes")

        # Tester une requête
        query = "Quels sont les facteurs qui influencent le plus les primes d'assurance?"
        print(f"🧪 Test de la requête: {query}")

        result = client.analyze_query(query, df)

        print("\n" + "=" * 50)
        print("📋 RÉSULTAT DU TEST")
        print("=" * 50)

        if "erreur" in result:
            print(f"❌ ERREUR: {result['erreur']}")
        else:
            print(f"✅ Compréhension: {result.get('comprehension', 'N/A')[:100]}...")
            print(f"✅ Méthodologie: {result.get('methodologie', 'N/A')[:100]}...")
            print(f"✅ Insights: {len(result.get('insights', []))} insights générés")
            print(f"✅ Recommandations: {len(result.get('recommandations', []))} recommandations")
            print(f"✅ Réponse détaillée: {'OUI' if result.get('reponse_detaillee') else 'NON'}")

            # Afficher un extrait de la réponse
            reponse = result.get('reponse_detaillee', '')
            if reponse:
                print(f"\n📝 Extrait de la réponse:\n{reponse[:300]}...")

    except ImportError as e:
        print(f"❌ Erreur d'importation: {e}")
        print("Assurez-vous que custom_llm_client.py existe dans le dossier modules/")
    except Exception as e:
        print(f"❌ Erreur générale: {e}")


if __name__ == "__main__":
    test_openai_client()