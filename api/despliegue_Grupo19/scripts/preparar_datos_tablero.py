"""Genera los CSV que consume el tablero (población + resultados de los modelos)."""
import json
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "package-src"))
from model.processing.data_manager import load_dataset, prepare_features, split_partitions

SALIDA = RAIZ / "dashboard" / "app" / "data"
SALIDA.mkdir(parents=True, exist_ok=True)
csv_fuente = Path(sys.argv[1])

d = pd.read_csv(csv_fuente, encoding="utf-8-sig")
cols = ["DIRECTORIO", "objetivo_atraso", "P2097", "P2098", "P2106", "P2099S2", "P2104", "P2105S1", "P2107S1",
        "n_personas_que_ahorran", "n_personas_con_cuenta", "n_personas_trabajo_semana"]
d = d[cols].copy()

# partición
mod = prepare_features(load_dataset(file_name="elco_creditos_modelo.csv"))
tr, va, te = split_partitions(mod)
part = pd.Series("entrenamiento", index=mod.index)
part.loc[va.index] = "validacion"; part.loc[te.index] = "prueba"
assert (mod["DIRECTORIO"].to_numpy() == d["DIRECTORIO"].to_numpy()).all()
d["particion"] = part.to_numpy()
d.to_csv(SALIDA / "creditos_tablero.csv", index=False)
print(d["particion"].value_counts().to_dict())

# lote de ejemplo
ej = d[d.particion == "prueba"][["DIRECTORIO", "P2097", "P2098", "P2106", "P2099S2", "P2104", "P2105S1", "P2107S1", "objetivo_atraso"]]
ej.to_csv(SALIDA / "lote_ejemplo_prueba.csv", index=False)

# comparación de modelos (prueba, conjunto A)
cmp_ = pd.DataFrame([
    ["Regresión logística", "A", .3709, .7273, .3146, .5895, .4103, 0],
    ["Árbol de decisión", "A", .3420, .6893, .2895, .4632, .3563, 0],
    ["Gradient Boosting", "A", .4576, .7360, .5111, .4842, .4973, 1],
], columns=["modelo", "configuracion", "pr_auc", "roc_auc", "precision", "recall", "f1", "final"])
cmp_.to_csv(SALIDA / "comparacion_modelos.csv", index=False)

ab = pd.DataFrame([
    ["A: variables del crédito (7)", .4576, .7360, .5111, .4842, .4973],
    ["B: crédito + hogar (14)", .4510, .7473, .3357, .4947, .4000],
], columns=["conjunto", "pr_auc", "roc_auc", "precision", "recall", "f1"])
ab.to_csv(SALIDA / "comparacion_A_vs_B.csv", index=False)

imp = pd.DataFrame([["P2099S2", "Año de inicio del crédito", .2228], ["P2104", "Valor inicial del crédito", .0291],
                    ["P2107S1", "Plazo (meses)", .0290], ["P2105S1", "Valor de la cuota", .0208],
                    ["P2106", "Periodicidad de pago", .0115], ["P2097", "Entidad acreedora", -.0090],
                    ["P2098", "Destino del crédito", -.0111]], columns=["variable", "descripcion", "importancia"])
imp.to_csv(SALIDA / "importancia_permutacion_A.csv", index=False)

card = json.loads(next((RAIZ / "package-src/model/trained").glob("*card*.json")).read_text(encoding="utf-8"))
pd.DataFrame(card["priorizacion_prueba"]).to_csv(SALIDA / "priorizacion_final.csv", index=False)
print("OK ->", SALIDA)
