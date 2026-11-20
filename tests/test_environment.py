import sys
import numpy as np
import pandas as pd
import sklearn
import matplotlib
import seaborn
import plotly
import xgboost

def test_environment():
    print("=" * 50)
    print("EcoRoute AI - Environment Verification")
    print("=" * 50)

    print("Python:", sys.version.split()[0])
    print("NumPy:", np.__version__)
    print("Pandas:", pd.__version__)
    print("Scikit-learn:", sklearn.__version__)
    print("Matplotlib:", matplotlib.__version__)
    print("Seaborn:", seaborn.__version__)
    print("Plotly:", plotly.__version__)
    print("XGBoost:", xgboost.__version__)

    print("=" * 50)
    print("Environment is ready!")
    print("=" * 50)

if __name__ == "__main__":
    test_environment()