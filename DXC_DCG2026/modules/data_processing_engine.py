import pandas as pd
import numpy as np
from datetime import datetime
import re
from typing import Dict, List, Any, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')

# Import des bibliothèques scientifiques disponibles
from scipy import stats
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


class DataProcessingEngine:
    """
    Processeur de données avec méthodes scientifiques rigoureuses
    tout en conservant les formats et modalités originaux
    """

    def __init__(self, significance_level: float = 0.05):
        """
        Initialise le processeur scientifique

        Args:
            significance_level: Niveau de signification pour les tests statistiques
        """
        self.df_raw = None
        self.df_processed = None
        self.column_types = {}
        self.original_dtypes = {}
        self.original_categories = {}  # Pour stocker les catégories originales
        self.significance_level = significance_level
        self.scientific_report = {}
        self.imputation_models = {}

    # =================================================
    # 1. DÉTECTION SCIENTIFIQUE DES TYPES
    # =================================================

    def detect_column_types(self, df: pd.DataFrame) -> Dict:
        """
        Détection scientifique des types de variables avec tests formels
        """
        column_types = {}

        for col in df.columns:
            col_series = df[col]
            col_dtype = str(col_series.dtype)

            # Analyse scientifique du type
            scientific_type = self._scientific_type_detection(col_series, col)

            # Tests statistiques selon le type détecté
            statistical_tests = self._perform_statistical_tests(col_series, scientific_type)

            # Analyse de distribution
            distribution_analysis = self._analyze_distribution(col_series, scientific_type)

            # Détection des anomalies
            anomaly_detection = self._detect_anomalies(col_series, scientific_type)

            column_types[col] = {
                "type": scientific_type,
                "original_dtype": col_dtype,
                "unique_values": col_series.nunique(),
                "missing_percentage": (col_series.isnull().sum() / len(col_series)) * 100,
                "statistical_tests": statistical_tests,
                "distribution": distribution_analysis,
                "anomalies": anomaly_detection,
                "sample_values": self._get_sample_values(col_series),
                "cardinality": self._calculate_cardinality(col_series, scientific_type)
            }

            # Sauvegarde des catégories originales pour les variables catégorielles
            if scientific_type == "categorical" and col_series.dtype == 'object':
                unique_cats = col_series.dropna().unique().tolist()
                self.original_categories[col] = unique_cats

        self.column_types = column_types
        self.original_dtypes = {col: str(df[col].dtype) for col in df.columns}

        return column_types

    def _scientific_type_detection(self, series: pd.Series, col_name: str) -> str:
        """
        Détection scientifique du type de variable avec tests statistiques
        """
        col_lower = col_name.lower()
        n_unique = series.nunique()
        n_total = len(series.dropna())

        if n_total == 0:
            return "unknown"

        # 1. Test pour identifiants uniques
        if self._is_identifier(series, col_name, n_unique, n_total):
            return "identifier"

        # 2. Test pour dates avec patterns statistiques
        if self._is_date_column_scientific(series, col_name):
            return "date"

        # 3. Variables numériques - tests formels
        if pd.api.types.is_numeric_dtype(series):
            return self._determine_numeric_type(series, n_unique, n_total)

        # 4. Variables catégorielles - analyse de cardinalité
        if series.dtype == 'object':
            return self._determine_categorical_type(series, n_unique, n_total)

        # 5. Variables booléennes - test binaire
        if self._is_binary_variable(series, n_unique):
            return "binary"

        return "unknown"

    def _is_identifier(self, series: pd.Series, col_name: str,
                       n_unique: int, n_total: int) -> bool:
        """Test statistique pour identifiants"""
        id_keywords = ['id', 'code', 'num', 'ref', 'numero', 'matricule',
                       'pk', 'key', 'cle']
        col_lower = col_name.lower()

        # Critère 1: Nom de colonne suggérant un identifiant
        name_suggests_id = any(keyword in col_lower for keyword in id_keywords)

        # Critère 2: Taux d'unicité très élevé (>95%)
        high_uniqueness = (n_unique / n_total) > 0.95 if n_total > 0 else False

        # Critère 3: Tous les non-nuls sont uniques
        all_unique = n_unique == n_total

        return name_suggests_id and (high_uniqueness or all_unique)

    def _is_date_column_scientific(self, series: pd.Series, col_name: str) -> bool:
        """
        Détection scientifique des colonnes de date
        """
        # Exclusion basée sur des patterns connus
        non_date_patterns = ['prime', 'montant', 'amount', 'price', 'cost',
                             'value', 'ratio', 'rate', 'pourcentage', 'percentage',
                             'score', 'index', 'quantile']

        if any(pattern in col_name.lower() for pattern in non_date_patterns):
            return False

        sample = series.dropna().head(100)
        if len(sample) < 10:
            return False

        # Conversion et analyse des patterns de date
        date_likelihood = self._calculate_date_likelihood(sample)

        # Seuil scientifique: > 60% de vraisemblance de date
        return date_likelihood > 0.6

    def _calculate_date_likelihood(self, sample: pd.Series) -> float:
        """Calcule la vraisemblance que la série soit une date"""
        sample_str = sample.astype(str)
        patterns = [
            r'^\d{4}-\d{2}-\d{2}$',  # YYYY-MM-DD
            r'^\d{2}/\d{2}/\d{4}$',  # DD/MM/YYYY
            r'^\d{2}-\d{2}-\d{4}$',  # DD-MM-YYYY
            r'^\d{4}/\d{2}/\d{2}$',  # YYYY/MM/DD
            r'^\d{8}$',  # YYYYMMDD
            r'^\d{1,2}/\d{1,2}/\d{2,4}$',  # Formats variés
        ]

        pattern_matches = 0
        total_tested = 0

        for pattern in patterns:
            matches = sample_str.str.match(pattern).sum()
            pattern_matches += matches
            total_tested += len(sample_str)

        return pattern_matches / total_tested if total_tested > 0 else 0

    def _determine_numeric_type(self, series: pd.Series,
                                n_unique: int, n_total: int) -> str:
        """Détermine le type numérique précis"""
        series_numeric = pd.to_numeric(series, errors='coerce').dropna()

        if len(series_numeric) == 0:
            return "unknown"

        # Test de discrétion
        is_discrete = all(series_numeric.apply(lambda x: float(x).is_integer()))

        # Analyse de cardinalité relative
        cardinality_ratio = n_unique / n_total if n_total > 0 else 0

        if cardinality_ratio < 0.1 and n_unique <= 20:
            return "categorical_numeric" if is_discrete else "ordinal"
        elif cardinality_ratio < 0.3:
            return "discrete_numerical" if is_discrete else "continuous_numerical"
        else:
            return "continuous_numerical"

    def _determine_categorical_type(self, series: pd.Series,
                                    n_unique: int, n_total: int) -> str:
        """Détermine le type catégoriel précis"""
        cardinality_ratio = n_unique / n_total if n_total > 0 else 0

        if n_unique == 2:
            # Test pour variable binaire
            unique_vals = set(series.dropna().astype(str).str.lower())
            binary_sets = [{'0', '1'}, {'true', 'false'}, {'yes', 'no'},
                           {'oui', 'non'}, {'vrai', 'faux'}]
            if any(unique_vals.issubset(binary_set) for binary_set in binary_sets):
                return "binary"

        if cardinality_ratio < 0.1 or n_unique <= 20:
            return "categorical"
        elif cardinality_ratio < 0.3:
            return "high_cardinality_categorical"
        else:
            return "text"

    def _is_binary_variable(self, series: pd.Series, n_unique: int) -> bool:
        """Test formel pour variable binaire"""
        if n_unique != 2:
            return False

        unique_vals = set(series.dropna().astype(str).str.lower())
        binary_patterns = [
            {'0', '1'},
            {'true', 'false'},
            {'yes', 'no'},
            {'oui', 'non'},
            {'vrai', 'faux'},
            {'t', 'f'},
            {'y', 'n'}
        ]

        return any(unique_vals.issubset(pattern) for pattern in binary_patterns)

    # =================================================
    # 2. TESTS STATISTIQUES SCIENTIFIQUES
    # =================================================

    def _perform_statistical_tests(self, series: pd.Series, var_type: str) -> Dict:
        """Exécute des tests statistiques appropriés selon le type de variable"""
        tests = {}

        if var_type in ["continuous_numerical", "discrete_numerical"]:
            numeric_series = pd.to_numeric(series, errors='coerce').dropna()

            if len(numeric_series) >= 3:
                # Test de normalité
                tests["normality"] = self._test_normality(numeric_series)

                # Test d'homogénéité de variance (si applicable)
                if len(numeric_series) >= 10:
                    tests["variance_stability"] = self._test_variance_stability(numeric_series)

                # Tests de distribution
                tests["distribution_fit"] = self._test_distribution_fit(numeric_series)

        elif var_type == "categorical":
            # Test d'indépendance du chi² (pour variables catégorielles)
            if len(series.dropna()) >= 20:
                tests["chi_square_uniformity"] = self._test_categorical_uniformity(series)

        elif var_type == "binary":
            # Test binomial pour variables binaires
            tests["binomial_test"] = self._test_binomial_proportion(series)

        return tests

    def _test_normality(self, series: pd.Series) -> Dict:
        """Tests de normalité multiples"""
        if len(series) < 3:
            return {"error": "sample_size_too_small"}

        try:
            # Test de Shapiro-Wilk (recommandé pour n < 5000)
            shapiro_stat, shapiro_p = stats.shapiro(series)

            # Test d'Anderson-Darling
            anderson_result = stats.anderson(series, dist='norm')

            # Calcul de skewness et kurtosis
            skewness = stats.skew(series)
            kurtosis = stats.kurtosis(series)

            return {
                "shapiro_wilk": {
                    "statistic": float(shapiro_stat),
                    "p_value": float(shapiro_p),
                    "is_normal": shapiro_p > self.significance_level
                },
                "anderson_darling": {
                    "statistic": float(anderson_result.statistic),
                    "critical_values": anderson_result.critical_values.tolist(),
                    "significance_levels": anderson_result.significance_level.tolist()
                },
                "skewness": float(skewness),
                "kurtosis": float(kurtosis),
                "is_normal": shapiro_p > self.significance_level and abs(skewness) < 2
            }
        except Exception as e:
            return {"error": str(e)}

    def _test_variance_stability(self, series: pd.Series) -> Dict:
        """Test de stabilité de la variance"""
        # Diviser en sous-groupes
        n = len(series)
        if n < 20:
            return {"error": "sample_size_too_small"}

        k = min(4, n // 5)
        groups = np.array_split(series, k)
        variances = [np.var(group) for group in groups if len(group) > 1]

        # Test de Levene pour homogénéité des variances
        if len(variances) >= 2:
            try:
                levene_stat, levene_p = stats.levene(*[group for group in groups if len(group) > 1])
                return {
                    "levene_test": {
                        "statistic": float(levene_stat),
                        "p_value": float(levene_p),
                        "variance_homogeneous": levene_p > self.significance_level
                    },
                    "variance_ratio": float(max(variances) / min(variances)) if min(variances) > 0 else None
                }
            except:
                pass

        return {"error": "test_not_applicable"}

    def _test_distribution_fit(self, series: pd.Series) -> Dict:
        """Test d'ajustement à différentes distributions"""
        distributions = {
            "normal": stats.norm,
            "lognormal": stats.lognorm,
            "exponential": stats.expon,
            "gamma": stats.gamma,
            "beta": stats.beta
        }

        results = {}
        for dist_name, dist_func in distributions.items():
            try:
                # Fit distribution
                params = dist_func.fit(series)

                # Kolmogorov-Smirnov test
                ks_stat, ks_p = stats.kstest(series, dist_name, args=params)

                results[dist_name] = {
                    "parameters": [float(p) for p in params],
                    "ks_statistic": float(ks_stat),
                    "ks_p_value": float(ks_p),
                    "good_fit": ks_p > self.significance_level
                }
            except:
                results[dist_name] = {"error": "fitting_failed"}

        return results

    def _test_categorical_uniformity(self, series: pd.Series) -> Dict:
        """Test du chi² pour uniformité des catégories"""
        value_counts = series.value_counts()
        n_categories = len(value_counts)

        if n_categories < 2:
            return {"error": "insufficient_categories"}

        # Test du chi² pour uniformité
        chi2_stat, chi2_p = stats.chisquare(value_counts)

        return {
            "chi2_statistic": float(chi2_stat),
            "p_value": float(chi2_p),
            "is_uniform": chi2_p > self.significance_level,
            "entropy": float(stats.entropy(value_counts)),
            "gini_impurity": float(1 - sum((value_counts / len(series)) ** 2))
        }

    def _test_binomial_proportion(self, series: pd.Series) -> Dict:
        """Test binomial pour proportion"""
        binary_series = series.dropna()
        if len(binary_series) == 0:
            return {"error": "no_data"}

        # Convertir en 0/1
        unique_vals = binary_series.unique()
        if len(unique_vals) != 2:
            return {"error": "not_binary"}

        # Encodage simple
        mapping = {unique_vals[0]: 0, unique_vals[1]: 1}
        numeric_series = binary_series.map(mapping)

        proportion = numeric_series.mean()
        n = len(numeric_series)

        # Test binomial exact
        k = int(proportion * n)
        binom_p = stats.binom_test(k, n, p=0.5)

        return {
            "proportion": float(proportion),
            "sample_size": int(n),
            "success_count": int(k),
            "binomial_p_value": float(binom_p),
            "is_balanced": abs(proportion - 0.5) < 0.1
        }

    # =================================================
    # 3. ANALYSE DE DISTRIBUTION ET ANOMALIES
    # =================================================

    def _analyze_distribution(self, series: pd.Series, var_type: str) -> Dict:
        """Analyse scientifique de la distribution"""
        analysis = {}

        if var_type in ["continuous_numerical", "discrete_numerical", "categorical_numeric"]:
            numeric_series = pd.to_numeric(series, errors='coerce').dropna()

            if len(numeric_series) > 0:
                analysis.update({
                    "mean": float(numeric_series.mean()),
                    "median": float(numeric_series.median()),
                    "std": float(numeric_series.std()),
                    "variance": float(numeric_series.var()),
                    "range": float(numeric_series.max() - numeric_series.min()),
                    "iqr": float(numeric_series.quantile(0.75) - numeric_series.quantile(0.25)),
                    "cv": float(numeric_series.std() / numeric_series.mean()) if numeric_series.mean() != 0 else None,
                    "skewness": float(stats.skew(numeric_series)),
                    "kurtosis": float(stats.kurtosis(numeric_series)),
                    "percentiles": {f"p{p}": float(numeric_series.quantile(p / 100))
                                    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]}
                })

        elif var_type in ["categorical", "binary", "high_cardinality_categorical"]:
            value_counts = series.value_counts(normalize=True)
            analysis.update({
                "mode": value_counts.index[0] if len(value_counts) > 0 else None,
                "mode_frequency": float(value_counts.iloc[0]) if len(value_counts) > 0 else None,
                "shannon_entropy": float(stats.entropy(value_counts)),
                "gini_index": float(1 - sum(value_counts ** 2)),
                "simpson_diversity": float(1 / sum(value_counts ** 2)) if sum(value_counts ** 2) > 0 else None,
                "category_counts": value_counts.to_dict()
            })

        return analysis

    def _detect_anomalies(self, series: pd.Series, var_type: str) -> Dict:
        """Détection scientifique d'anomalies"""
        anomalies = {}

        if var_type in ["continuous_numerical", "discrete_numerical"]:
            numeric_series = pd.to_numeric(series, errors='coerce')
            valid_values = numeric_series.dropna()

            if len(valid_values) >= 10:
                # Méthode IQR
                q1 = valid_values.quantile(0.25)
                q3 = valid_values.quantile(0.75)
                iqr = q3 - q1

                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr

                iqr_outliers = ((numeric_series < lower_bound) | (numeric_series > upper_bound)).sum()

                # Méthode Z-score
                z_scores = np.abs((valid_values - valid_values.mean()) / valid_values.std())
                z_outliers = (z_scores > 3).sum()

                # Isolation Forest (si assez de données)
                if len(valid_values) >= 100:
                    try:
                        iso_forest = IsolationForest(contamination=0.1, random_state=42)
                        iso_predictions = iso_forest.fit_predict(valid_values.values.reshape(-1, 1))
                        iso_outliers = (iso_predictions == -1).sum()
                    except:
                        iso_outliers = None
                else:
                    iso_outliers = None

                anomalies.update({
                    "iqr_outliers": int(iqr_outliers),
                    "z_score_outliers": int(z_outliers),
                    "isolation_forest_outliers": int(iso_outliers) if iso_outliers else None,
                    "outlier_percentage_iqr": float(iqr_outliers / len(numeric_series) * 100),
                    "outlier_bounds": {"lower": float(lower_bound), "upper": float(upper_bound)}
                })

        elif var_type == "categorical":
            # Détection de catégories rares
            value_counts = series.value_counts(normalize=True)
            rare_threshold = 0.01  # 1%
            rare_categories = value_counts[value_counts < rare_threshold]

            anomalies["rare_categories"] = {
                "count": len(rare_categories),
                "categories": rare_categories.index.tolist(),
                "frequencies": rare_categories.tolist()
            }

        return anomalies

    def _calculate_cardinality(self, series: pd.Series, var_type: str) -> Dict:
        """Calcule les métriques de cardinalité"""
        n_unique = series.nunique()
        n_total = len(series.dropna())

        return {
            "unique_count": n_unique,
            "cardinality_ratio": n_unique / n_total if n_total > 0 else 0,
            "entropy": float(stats.entropy(series.value_counts(normalize=True))) if n_total > 0 else 0
        }

    # =================================================
    # 4. PRÉTRAITEMENT SCIENTIFIQUE
    # =================================================

    def scientific_preprocess(self, df: pd.DataFrame, target_column: Optional[str] = None,
                              strategy: str = "conservative") -> pd.DataFrame:
        """
        Prétraitement scientifique rigoureux avec conservation des formats

        Args:
            df: DataFrame à traiter
            target_column: Colonne cible optionnelle
            strategy: "conservative", "balanced", or "aggressive"
        """
        df_processed = df.copy()
        self.scientific_report = {
            "preprocessing_strategy": strategy,
            "steps_applied": [],
            "quality_metrics": {},
            "type_conservation": {}
        }

        # 1. Sauvegarde des métadonnées originales
        self._preserve_original_metadata(df_processed)

        # 2. Détection scientifique des types
        if not self.column_types:
            self.detect_column_types(df_processed)

        # 3. Suppression des doublons avec analyse statistique
        df_processed = self._statistical_deduplication(df_processed)

        # 4. Traitement scientifique des valeurs manquantes
        df_processed = self._scientific_missing_value_handling(df_processed, target_column, strategy)

        # 5. Traitement scientifique des anomalies
        df_processed = self._scientific_anomaly_handling(df_processed, strategy)

        # 6. Normalisation scientifique (si nécessaire)
        df_processed = self._scientific_normalization(df_processed, strategy)

        # 7. Conservation et validation des types
        df_processed = self._validate_and_conserve_types(df_processed)

        # 8. Rapport de qualité
        self._generate_quality_report(df_processed)

        self.df_processed = df_processed
        return df_processed

    def _preserve_original_metadata(self, df: pd.DataFrame) -> None:
        """Sauvegarde les métadonnées originales"""
        self.original_dtypes = {col: str(df[col].dtype) for col in df.columns}

        # Sauvegarde des catégories pour les variables catégorielles
        for col in df.columns:
            if df[col].dtype.name == 'category':
                self.original_categories[col] = df[col].cat.categories.tolist()
            elif df[col].dtype == 'object' and df[col].nunique() <= 100:
                self.original_categories[col] = df[col].dropna().unique().tolist()

    def _statistical_deduplication(self, df: pd.DataFrame) -> pd.DataFrame:
        """Déduplication avec analyse statistique"""
        initial_rows = len(df)
        df_dedup = df.drop_duplicates()
        duplicates_removed = initial_rows - len(df_dedup)

        if duplicates_removed > 0:
            duplicate_percentage = (duplicates_removed / initial_rows) * 100
            self.scientific_report["steps_applied"].append(
                f"Déduplication statistique: {duplicates_removed} doublons supprimés "
                f"({duplicate_percentage:.2f}% des données)"
            )

        return df_dedup

    def _scientific_missing_value_handling(self, df: pd.DataFrame, target_column: Optional[str],
                                           strategy: str) -> pd.DataFrame:
        """Traitement scientifique des valeurs manquantes"""
        df_imputed = df.copy()

        for col in df_imputed.columns:
            if col == target_column:
                continue

            missing_count = df_imputed[col].isnull().sum()
            if missing_count == 0:
                continue

            missing_percentage = (missing_count / len(df_imputed)) * 100
            var_type = self.column_types.get(col, {}).get("type", "unknown")

            # Stratégie basée sur le pourcentage de valeurs manquantes
            if missing_percentage > 30:
                # Trop de valeurs manquantes - création d'indicateur
                indicator_name = f"{col}_is_missing"
                df_imputed[indicator_name] = df_imputed[col].isnull().astype(int)
                self.scientific_report["steps_applied"].append(
                    f"{col}: Indicateur de manquant créé ({missing_percentage:.1f}% manquants)"
                )

            # Imputation selon le type de variable et la stratégie
            imputation_method = self._select_imputation_method(
                df_imputed[col], var_type, missing_percentage, strategy
            )

            df_imputed[col] = imputation_method
            self.imputation_models[col] = imputation_method.__class__.__name__

            self.scientific_report["steps_applied"].append(
                f"{col}: {missing_count} valeurs manquantes traitées "
                f"(méthode: {self.imputation_models[col]})"
            )

        # Traitement spécial pour la variable cible
        if target_column and target_column in df_imputed.columns:
            initial_len = len(df_imputed)
            df_imputed = df_imputed.dropna(subset=[target_column])
            rows_removed = initial_len - len(df_imputed)

            if rows_removed > 0:
                self.scientific_report["steps_applied"].append(
                    f"Variable cible {target_column}: {rows_removed} observations supprimées"
                )

        return df_imputed

    def _select_imputation_method(self, series: pd.Series, var_type: str,
                                  missing_percentage: float, strategy: str):
        """Sélectionne la méthode d'imputation scientifique appropriée"""

        if var_type in ["continuous_numerical", "discrete_numerical", "categorical_numeric"]:
            if strategy == "aggressive" and missing_percentage < 20:
                # MICE - Multiple Imputation by Chained Equations
                try:
                    imputer = IterativeImputer(max_iter=10, random_state=42)
                    imputed_values = imputer.fit_transform(series.values.reshape(-1, 1))
                    return pd.Series(imputed_values.flatten(), index=series.index)
                except:
                    pass

            elif strategy == "balanced" and missing_percentage < 30:
                # KNN Imputation
                try:
                    imputer = KNNImputer(n_neighbors=5)
                    imputed_values = imputer.fit_transform(series.values.reshape(-1, 1))
                    return pd.Series(imputed_values.flatten(), index=series.index)
                except:
                    pass

            # Fallback: médiane robuste
            median_val = series.median()
            if pd.api.types.is_integer_dtype(series):
                median_val = int(median_val)
            return series.fillna(median_val)

        elif var_type in ["categorical", "binary"]:
            # Imputation par mode avec conservation des types
            mode_val = series.mode()
            if not mode_val.empty:
                return series.fillna(mode_val[0])
            else:
                # Création d'une catégorie spéciale
                special_value = "MISSING_CAT" if var_type == "categorical" else 0
                return series.fillna(special_value)

        else:
            # Imputation par la valeur la plus fréquente
            most_frequent = series.value_counts().index[0] if len(series.dropna()) > 0 else None
            return series.fillna(most_frequent) if most_frequent else series

    def _scientific_anomaly_handling(self, df: pd.DataFrame, strategy: str) -> pd.DataFrame:
        """Traitement scientifique des anomalies"""
        df_clean = df.copy()

        for col in df_clean.columns:
            var_type = self.column_types.get(col, {}).get("type", "unknown")

            if var_type in ["continuous_numerical", "discrete_numerical"]:
                series = df_clean[col].dropna()

                if len(series) < 10:
                    continue

                # Détection d'anomalies
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1

                if iqr == 0:
                    continue

                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr

                outliers_mask = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
                n_outliers = outliers_mask.sum()

                if n_outliers > 0:
                    if strategy == "conservative":
                        # Winsorization: remplacement par les percentiles
                        df_clean[col] = np.where(df_clean[col] < lower_bound, lower_bound,
                                                 np.where(df_clean[col] > upper_bound, upper_bound,
                                                          df_clean[col]))
                    elif strategy == "balanced":
                        # Imputation par la médiane
                        median_val = series.median()
                        df_clean.loc[outliers_mask, col] = median_val
                    # Pour "aggressive", on garde les outliers

                    self.scientific_report["steps_applied"].append(
                        f"{col}: {n_outliers} anomalies traitées (stratégie: {strategy})"
                    )

        return df_clean

    def _scientific_normalization(self, df: pd.DataFrame, strategy: str) -> pd.DataFrame:
        """Normalisation scientifique adaptative"""
        if strategy != "aggressive":
            return df  # Pas de normalisation pour les stratégies conservatrices

        df_normalized = df.copy()

        for col in df_normalized.columns:
            var_type = self.column_types.get(col, {}).get("type", "unknown")

            if var_type == "continuous_numerical":
                series = df_normalized[col].dropna()

                if len(series) < 10:
                    continue

                # Test de normalité
                normality = self.column_types[col].get("statistical_tests", {}).get("normality", {})

                if not normality.get("is_normal", False):
                    # Transformation pour normaliser
                    skewness = self.column_types[col].get("distribution", {}).get("skewness", 0)

                    try:
                        if skewness > 1:
                            # Forte asymétrie positive: log transformation
                            min_val = series.min()
                            if min_val <= 0:
                                df_normalized[col] = np.log1p(df_normalized[col] - min_val + 1e-10)
                            else:
                                df_normalized[col] = np.log1p(df_normalized[col])
                            self.scientific_report["steps_applied"].append(
                                f"{col}: Transformation log appliquée (skewness={skewness:.2f})"
                            )
                        elif skewness < -1:
                            # Forte asymétrie négative: transformation carrée
                            df_normalized[col] = np.power(df_normalized[col], 2)
                            self.scientific_report["steps_applied"].append(
                                f"{col}: Transformation carrée appliquée (skewness={skewness:.2f})"
                            )
                    except:
                        pass

        return df_normalized

    def _validate_and_conserve_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validation et conservation rigoureuse des types"""
        df_validated = df.copy()
        type_conservation = {}

        for col in df_validated.columns:
            original_dtype = self.original_dtypes.get(col)
            current_dtype = str(df_validated[col].dtype)

            # Conservation du type d'origine si possible
            if original_dtype and original_dtype != current_dtype:
                try:
                    if 'int' in original_dtype:
                        df_validated[col] = pd.to_numeric(df_validated[col], errors='coerce').astype(original_dtype)
                    elif original_dtype == 'category' and col in self.original_categories:
                        df_validated[col] = pd.Categorical(df_validated[col],
                                                           categories=self.original_categories[col])
                    elif 'datetime' in original_dtype:
                        df_validated[col] = pd.to_datetime(df_validated[col], errors='coerce')

                    type_conservation[col] = {
                        "original": original_dtype,
                        "intermediate": current_dtype,
                        "final": str(df_validated[col].dtype),
                        "conserved": str(df_validated[col].dtype) == original_dtype
                    }
                except:
                    type_conservation[col] = {
                        "original": original_dtype,
                        "final": current_dtype,
                        "conserved": False,
                        "error": "conversion_failed"
                    }
            else:
                type_conservation[col] = {
                    "original": original_dtype,
                    "final": current_dtype,
                    "conserved": True
                }

        self.scientific_report["type_conservation"] = type_conservation
        return df_validated

    def _generate_quality_report(self, df: pd.DataFrame) -> None:
        """Génère un rapport de qualité scientifique"""
        quality_metrics = {
            "completeness": float(df.notna().mean().mean()),
            "duplicate_percentage": float((len(df) - len(df.drop_duplicates())) / len(df) if len(df) > 0 else 0),
            "type_conservation_rate": float(sum(1 for v in self.scientific_report["type_conservation"].values()
                                                if v.get("conserved", False)) / len(
                self.scientific_report["type_conservation"])
                                            if self.scientific_report["type_conservation"] else 0)
        }

        # Métriques pour variables numériques
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            numeric_means = df[numeric_cols].mean()
            numeric_stds = df[numeric_cols].std()

            if numeric_means.mean() != 0:
                quality_metrics["numeric_variability"] = float(numeric_stds.mean() / numeric_means.mean())
            else:
                quality_metrics["numeric_variability"] = 0.0

            # Calcul de la skewness moyenne
            skewness_values = []
            for col in numeric_cols:
                try:
                    skew = stats.skew(df[col].dropna())
                    skewness_values.append(abs(skew))
                except:
                    pass

            if skewness_values:
                quality_metrics["numeric_skewness"] = float(np.mean(skewness_values))
            else:
                quality_metrics["numeric_skewness"] = 0.0

        self.scientific_report["quality_metrics"] = quality_metrics

    # =================================================
    # 5. MÉTHODES UTILITAIRES ET RAPPORTS
    # =================================================

    def get_scientific_report(self) -> Dict:
        """Retourne le rapport scientifique complet"""
        return self.scientific_report

    def get_statistical_summary(self) -> Dict:
        """Résumé statistique scientifique"""
        if not self.column_types:
            return {}

        summary = {
            "variable_types": {},
            "data_quality": {
                "total_variables": len(self.column_types),
                "complete_variables": sum(1 for v in self.column_types.values()
                                          if v.get("missing_percentage", 100) == 0),
                "normal_variables": 0,
                "high_quality_variables": 0
            },
            "anomalies_summary": {}
        }

        # Analyse par type
        type_counts = {}
        normal_count = 0
        high_quality_count = 0

        for col, info in self.column_types.items():
            var_type = info.get("type", "unknown")
            type_counts[var_type] = type_counts.get(var_type, 0) + 1

            # Compte des variables normales
            if info.get("statistical_tests", {}).get("normality", {}).get("is_normal", False):
                normal_count += 1

            # Variables de haute qualité (peu de manquants, pas d'anomalies)
            missing_pct = info.get("missing_percentage", 100)
            anomaly_pct = info.get("anomalies", {}).get("outlier_percentage_iqr", 0)

            if missing_pct < 5 and anomaly_pct < 1:
                high_quality_count += 1

        summary["variable_types"] = type_counts
        summary["data_quality"]["normal_variables"] = normal_count
        summary["data_quality"]["high_quality_variables"] = high_quality_count

        return summary

    def get_sample_values(self, series: pd.Series) -> List:
        """Retourne un échantillon représentatif des valeurs (JSON-sérialisable)"""
        non_null = series.dropna()
        if len(non_null) == 0:
            return []

        # Échantillon stratifié pour les variables catégorielles
        if series.dtype == 'object' or series.nunique() <= 10:
            # Prendre des échantillons de chaque catégorie
            samples = []
            for cat in series.unique()[:5]:  # Limite à 5 catégories
                cat_samples = series[series == cat].head(2)
                samples.extend(cat_samples.tolist())
            samples = samples[:5]  # Retourne max 5 échantillons
        else:
            # Pour les variables continues, retourner des percentiles
            if len(non_null) >= 5:
                percentiles = [0, 25, 50, 75, 100]
                samples = [float(np.percentile(non_null, p)) for p in percentiles]
            else:
                samples = non_null.head(5).tolist()

        # Convertir les Timestamp et autres types non JSON-serializable en string
        serializable_samples = []
        for value in samples:
            if pd.isna(value):
                serializable_samples.append(None)
            elif isinstance(value, (pd.Timestamp, pd.DatetimeIndex)):
                # Convertir les dates en string ISO
                serializable_samples.append(value.isoformat())
            elif isinstance(value, pd.Period):
                # Convertir les périodes
                serializable_samples.append(str(value))
            elif isinstance(value, (np.integer, np.int64, np.int32)):
                # Convertir les numpy integers
                serializable_samples.append(int(value))
            elif isinstance(value, (np.floating, np.float64, np.float32)):
                # Convertir les numpy floats
                serializable_samples.append(float(value))
            elif isinstance(value, np.ndarray):
                # Convertir les arrays numpy
                serializable_samples.append(value.tolist())
            else:
                # Pour les autres types, essayer de convertir en Python natif
                try:
                    serializable_samples.append(value.item() if hasattr(value, 'item') else value)
                except:
                    serializable_samples.append(str(value))

        return serializable_samples

    def _get_sample_values(self, series: pd.Series) -> List:
        """Méthode interne pour échantillon de valeurs"""
        return self.get_sample_values(series)

    def detect_potential_targets(self, df: pd.DataFrame) -> List[str]:
        """
        Détection scientifique des variables cibles potentielles
        """
        potential_targets = []

        for col in df.columns:
            col_lower = col.lower()

            # 1. Critères sémantiques
            target_keywords = [
                "target", "label", "class", "result", "outcome",
                "risque", "risk", "churn", "attrition", "lapse", "renewed",
                "defaut", "default", "fraude", "fraud", "resilie", "renouvelle",
                "statut", "status", "etat", "state", "categorie", "category",
                "flag", "binary", "binaire", "yesno", "oui_non"
            ]

            if any(keyword in col_lower for keyword in target_keywords):
                potential_targets.append(col)
                continue

            # 2. Critères statistiques
            n_unique = df[col].nunique()
            n_total = len(df[col].dropna())

            if n_total > 0:
                unique_ratio = n_unique / n_total

                # Variables binaires ou à faible cardinalité
                if n_unique == 2 or (n_unique <= 10 and unique_ratio < 0.3):
                    # Vérifier la distribution
                    value_counts = df[col].value_counts(normalize=True)
                    if len(value_counts) > 0:
                        min_proportion = value_counts.min()

                        # Éviter les variables trop déséquilibrées
                        if min_proportion > 0.05:  # Au moins 5% dans la catégorie minoritaire
                            potential_targets.append(col)

        return list(set(potential_targets))

    def get_column_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Statistiques détaillées par colonne
        """
        from scipy import stats as scipy_stats  # Renommer pour éviter le conflit

        column_stats = {}

        for col in df.columns:
            col_info = {
                "dtype": str(df[col].dtype),
                "non_null_count": int(df[col].notna().sum()),
                "null_count": int(df[col].isna().sum()),
                "null_percentage": float((df[col].isna().sum() / len(df)) * 100),
                "unique_count": int(df[col].nunique()),
                "sample_values": self.get_sample_values(df[col])
            }

            if pd.api.types.is_numeric_dtype(df[col]):
                non_null = df[col].dropna()
                if len(non_null) > 0:
                    col_info.update({
                        "mean": float(non_null.mean()),
                        "std": float(non_null.std()),
                        "min": float(non_null.min()),
                        "max": float(non_null.max()),
                        "median": float(non_null.median()),
                        "q1": float(non_null.quantile(0.25)),
                        "q3": float(non_null.quantile(0.75))
                    })

                    # Calculer skewness et kurtosis seulement si assez de données
                    if len(non_null) >= 3:
                        try:
                            col_info["skewness"] = float(scipy_stats.skew(non_null))
                            col_info["kurtosis"] = float(scipy_stats.kurtosis(non_null))
                        except:
                            col_info["skewness"] = None
                            col_info["kurtosis"] = None
                    else:
                        col_info["skewness"] = None
                        col_info["kurtosis"] = None

            column_stats[col] = col_info

        return column_stats

    def get_conversion_report(self) -> List[str]:
        """Retourne le rapport des conversions appliquées"""
        if hasattr(self, 'scientific_report'):
            return self.scientific_report.get("steps_applied", [])
        return []

    # Méthodes compatibilité
    def preprocess_data(self, df: pd.DataFrame, target_column: Optional[str] = None,
                        preserve_types: bool = True) -> pd.DataFrame:
        """
        Méthode de compatibilité avec l'ancienne interface

        Args:
            df: DataFrame à traiter
            target_column: Colonne cible optionnelle
            preserve_types: Si True, conserve les types d'origine
        """
        if preserve_types:
            strategy = "conservative"
        else:
            strategy = "aggressive"

        return self.scientific_preprocess(df, target_column, strategy)