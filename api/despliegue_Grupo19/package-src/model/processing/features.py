import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CategoryToString(BaseEstimator, TransformerMixin):
    """Convierte códigos categóricos (1, 2, ...) a texto ('1', '2', ...) conservando los vacíos.

    Es la misma representación usada al entrenar; así un código llega igual
    sin importar si viene como entero, decimal (3.0) o texto ('3').
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for col in X.columns:
            num = pd.to_numeric(X[col], errors="coerce")
            texto = num.round().astype("Int64").astype("string").astype(object)
            texto[num.isna().to_numpy()] = np.nan
            X[col] = texto
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)
