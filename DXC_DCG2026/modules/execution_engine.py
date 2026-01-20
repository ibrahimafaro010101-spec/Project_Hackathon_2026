# modules/execution_engine.py - Version corrigée et simplifiée
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, List
import os


class LocalExecutionEngine:
    """
    Moteur d'exécution local simplifié
    """

    def __init__(self, working_dir: str = "."):
        self.working_dir = Path(working_dir)
        self.results = {}

    def execute_analysis_package(self, package_dir: str, data_path: str = None) -> Dict[str, Any]:
        """
        Exécute un package d'analyse
        """
        try:
            package_path = Path(package_dir)

            # Vérifier les prérequis
            if not self._check_prerequisites(package_path):
                return {
                    "success": False,
                    "error": "Fichiers requis manquants"
                }

            # Mettre à jour le chemin des données si fourni
            if data_path:
                self._update_data_path(package_path / "main.py", data_path)

            # Exécuter l'analyse
            execution_result = self._run_analysis(package_path)

            return execution_result

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def _check_prerequisites(self, package_path: Path) -> bool:
        """Vérifie les prérequis"""
        required_files = ["main.py"]

        for file in required_files:
            if not (package_path / file).exists():
                print(f" Fichier manquant: {file}")
                return False

        # Vérifier requirements.txt (optionnel)
        if not (package_path / "requirements.txt").exists():
            print("requirements.txt non trouvé (optionnel)")

        return True

    def _update_data_path(self, main_script: Path, data_path: str):
        """Met à jour le chemin des données"""
        if not main_script.exists():
            return

        try:
            with open(main_script, 'r', encoding='utf-8') as f:
                content = f.read()

            # Remplacer plusieurs patterns possibles
            replacements = [
                ('DATA_PATH = "', f'DATA_PATH = "{data_path}"'),
                ('data_file = "', f'data_file = "{data_path}"'),
                ('"votre_fichier.csv"', f'"{data_path}"'),
                ('"donnees.csv"', f'"{data_path}"')
            ]

            new_content = content
            for old, new in replacements:
                if old in new_content:
                    new_content = new_content.replace(old, new, 1)
                    break

            with open(main_script, 'w', encoding='utf-8') as f:
                f.write(new_content)

        except Exception as e:
            print(f" Erreur lors de la mise à jour du chemin: {e}")

    def _run_analysis(self, package_path: Path) -> Dict[str, Any]:
        """Exécute l'analyse principale"""
        main_script = package_path / "main.py"

        try:
            print(f" Exécution de: {main_script}")

            # Installer les dépendances si requirements.txt existe
            requirements_file = package_path / "requirements.txt"
            if requirements_file.exists():
                print(" Installation des dépendances...")
                try:
                    subprocess.run(
                        [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)],
                        check=True,
                        capture_output=True,
                        text=True
                    )
                    print(" Dépendances installées")
                except subprocess.CalledProcessError as e:
                    print(f" Erreur installation: {e.stderr}")

            # Exécuter le script
            result = subprocess.run(
                [sys.executable, str(main_script)],
                capture_output=True,
                text=True,
                cwd=str(package_path)
            )

            # Collecter les résultats
            results = self._collect_results(package_path)
            output_files = self._find_output_files(package_path)

            return {
                "success": result.returncode == 0,
                "returncode": result.returncode,
                "stdout": result.stdout[:1000] + "..." if len(result.stdout) > 1000 else result.stdout,
                "stderr": result.stderr,
                "results": results,
                "output_files": output_files
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def _collect_results(self, package_path: Path) -> Dict[str, Any]:
        """Collecte les résultats de l'analyse"""
        results = {}

        # Chercher les fichiers JSON
        for json_file in package_path.glob("*.json"):
            if json_file.name not in ["config.json"]:
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        results[json_file.name] = json.load(f)
                except:
                    results[json_file.name] = f"Fichier: {json_file.name}"

        # Chercher les fichiers CSV
        for csv_file in package_path.glob("*.csv"):
            results[csv_file.name] = f"Fichier CSV: {csv_file.name}"

        return results

    def _find_output_files(self, package_path: Path) -> List[str]:
        """Trouve tous les fichiers de sortie"""
        output_files = []

        for file in package_path.glob("*"):
            if file.is_file():
                ext = file.suffix.lower()
                if ext in ['.json', '.csv', '.html', '.txt', '.log']:
                    output_files.append(file.name)

        return output_files

    def generate_execution_report(self, execution_result: Dict[str, Any]) -> str:
        """Génère un rapport d'exécution"""

        report = ["#  Rapport d'Exécution", ""]

        # Statut
        if execution_result.get("success"):
            report.append("##  SUCCÈS")
            report.append("L'exécution s'est terminée avec succès.")
        else:
            report.append("##  ÉCHEC")
            error = execution_result.get("error", "Erreur inconnue")
            report.append(f"Erreur: {error}")

        # Code de retour
        returncode = execution_result.get("returncode")
        if returncode is not None:
            report.append(f"\n**Code de retour:** {returncode}")

        # Sortie
        stdout = execution_result.get("stdout", "")
        if stdout:
            report.append("\n##  Sortie standard")
            report.append(f"```\n{stdout}\n```")

        # Erreurs
        stderr = execution_result.get("stderr", "")
        if stderr:
            report.append("\n##  Erreurs")
            report.append(f"```\n{stderr}\n```")

        # Fichiers générés
        output_files = execution_result.get("output_files", [])
        if output_files:
            report.append("\n##  Fichiers générés")
            for file in output_files:
                report.append(f"- {file}")
            report.append(f"\n**Total:** {len(output_files)} fichiers")

        # Résultats
        results = execution_result.get("results", {})
        if results:
            report.append("\n##  Résultats")
            for filename, content in results.items():
                report.append(f"\n### {filename}")
                if isinstance(content, dict):
                    # Afficher quelques clés du JSON
                    for key, value in list(content.items())[:3]:
                        report.append(f"- **{key}:** {str(value)[:100]}...")
                else:
                    report.append(str(content))

        return "\n".join(report)

    def create_simple_analysis_package(self, output_dir: str = "analysis_package") -> str:
        """
        Crée un package d'analyse simple
        """
        package_path = Path(output_dir)
        package_path.mkdir(exist_ok=True)

        # Créer main.py
        main_content = '''"""
 Analyse de données - Package généré
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime

def main():
    print("=" * 50)
    print(" ANALYSE DE DONNÉES")
    print("=" * 50)

    # Chemin des données
    data_file = "donnees.csv"  # À remplacer par votre fichier

    try:
        # Charger les données
        print(f" Chargement: {data_file}")
        df = pd.read_csv(data_file)
        print(f" Données chargées: {len(df)} lignes × {len(df.columns)} colonnes")

        # Analyse simple
        print("\\n ANALYSE DESCRIPTIVE")
        print("-" * 40)

        # Statistiques de base
        results = {
            "metadata": {
                "date_analyse": datetime.now().isoformat(),
                "lignes": len(df),
                "colonnes": len(df.columns),
                "fichier_source": data_file
            },
            "statistiques": {}
        }

        # Colonnes numériques
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            print("Colonnes numériques:")
            for col in numeric_cols[:5]:  # Limiter aux 5 premières
                stats = {
                    "moyenne": float(df[col].mean()),
                    "mediane": float(df[col].median()),
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                    "ecart_type": float(df[col].std())
                }
                results["statistiques"][col] = stats
                print(f"  • {col}: moyenne={stats['moyenne']:.2f}")

        # Colonnes catégorielles
        cat_cols = df.select_dtypes(include=['object']).columns
        if len(cat_cols) > 0:
            print("\\nColonnes catégorielles:")
            for col in cat_cols[:3]:
                unique_count = df[col].nunique()
                print(f"  • {col}: {unique_count} valeurs uniques")
                results["statistiques"][f"{col}_unique"] = unique_count

        # Sauvegarder les résultats
        with open("resultats_analyse.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print("\\n Résultats sauvegardés dans resultats_analyse.json")

        # Générer un CSV de synthèse
        if len(numeric_cols) > 0:
            df[numeric_cols].describe().to_csv("statistiques_synthese.csv")
            print(" Statistiques sauvegardées dans statistiques_synthese.csv")

        print("\\n" + "=" * 50)
        print(" ANALYSE TERMINÉE")
        print("=" * 50)

    except FileNotFoundError:
        print(f" Fichier {data_file} non trouvé")
        print("Veuillez placer votre fichier de données dans le même dossier")
    except Exception as e:
        print(f" Erreur: {e}")

if __name__ == "__main__":
    main()
'''

        with open(package_path / "main.py", "w", encoding="utf-8") as f:
            f.write(main_content)

        # Créer requirements.txt
        requirements = """# Dépendances pour l'analyse
pandas>=1.5.0
numpy>=1.21.0
"""

        with open(package_path / "requirements.txt", "w", encoding="utf-8") as f:
            f.write(requirements)

        # Créer README.md
        readme = """#  Package d'Analyse de Données

##  Utilisation

1. Placez votre fichier de données (CSV) dans ce dossier
2. Renommez-le en `donnees.csv` ou modifiez `main.py`
3. Exécutez: `python main.py`

##  Structure

- `main.py` - Script d'analyse principal
- `requirements.txt` - Dépendances Python
- `resultats_analyse.json` - Résultats au format JSON (généré)
- `statistiques_synthese.csv` - Statistiques descriptives (généré)

## 🔧 Personnalisation

Modifiez `main.py` pour adapter l'analyse à vos besoins.
"""

        with open(package_path / "README.md", "w", encoding="utf-8") as f:
            f.write(readme)

        return str(package_path)


# Fonction utilitaire simple
def run_simple_analysis(data_file: str, output_dir: str = "results") -> Dict[str, Any]:
    """
    Exécute une analyse simple sur un fichier de données
    """
    engine = LocalExecutionEngine()

    # Créer un package temporaire
    import tempfile
    import shutil

    temp_dir = tempfile.mkdtemp()
    package_dir = engine.create_simple_analysis_package(temp_dir)

    try:
        # Copier le fichier de données
        import shutil
        target_file = Path(package_dir) / "donnees.csv"
        shutil.copy(data_file, target_file)

        # Exécuter l'analyse
        result = engine.execute_analysis_package(package_dir, str(target_file))

        # Copier les résultats vers output_dir
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        for file in Path(package_dir).glob("*"):
            if file.is_file() and file.suffix in ['.json', '.csv', '.txt']:
                shutil.copy(file, output_path / file.name)

        return result

    finally:
        # Nettoyer le répertoire temporaire
        shutil.rmtree(temp_dir, ignore_errors=True)