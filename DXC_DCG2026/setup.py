# setup.py
from setuptools import setup, find_packages

setup(
    name="lik-insurance-analyst",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "streamlit>=1.28.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "plotly>=5.17.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "python-dotenv>=1.0.0",
        "openpyxl>=3.1.0",
    ],
    extras_require={
        "full": [
            "xgboost>=1.7.0",
            "lightgbm>=4.0.0",
            "catboost>=1.2.0",
            "statsmodels>=0.14.0",
            "prophet>=1.1.0",
            "shap>=0.42.0",
            "imbalanced-learn>=0.11.0",
            "reportlab>=4.0.0",
            "python-docx>=0.8.11",
        ]
    },
    python_requires=">=3.8",
)