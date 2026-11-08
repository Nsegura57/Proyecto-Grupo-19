"""Tablero: Cumplimiento crediticio y vulnerabilidad financiera de hogares (ELCO 2022).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Cumplimiento crediticio", layout="wide")

CSV = "ELCO_2022_creditos_ahorro_laboral_consolidado.csv"
AQUI = Path(__file__).resolve().parent
RUTAS = [AQUI.parent / "data" / CSV, AQUI / "data" / CSV, Path("data") / CSV, Path(CSV)]
MORADO, AZUL = "#7570E6", "#3D35B8"
API_URL = None

ENTIDAD = {1: "Bancos o entidades financieras en Colombia", 2: "Fondos de empleados o cooperativas",
           3: "Almacenes de cadena, hipermercados", 4: "Cajas de compensación", 5: "Gremios o asociaciones",
           6: "Familiares (de otros hogares)", 7: "Amigos(as)", 8: "Prestamistas", 9: "Empleador",
           10: "Bancos o entidades financieras en el exterior", 11: "Icetex", 12: "Tenderos",
           13: "Compras por catálogo", 14: "Casas de empeño o casas comerciales", 15: "Otro"}
DESTINO = {1: "Educación", 2: "Comprar casa o apartamento", 3: "Comprar carro", 4: "Comprar otros activos",
           5: "Cubrir daños de desastres naturales", 6: "Recreación, celebraciones, viajes y entretenimiento",
           7: "Montar o invertir en un negocio", 8: "Mejorar la vivienda", 9: "Pagar deudas",
           10: "Comprar ropa o comida", 11: "Comprar muebles o electrodomésticos", 12: "Pagar gastos de salud",
           13: "Compra de maquinaria y equipo", 14: "Comprar tierra", 15: "Inversiones agrícolas y/o pecuarias",
           16: "Otro"}
PERIODICIDAD = {1: "Diaria", 2: "Semanal", 3: "Quincenal", 4: "Mensual", 5: "Bimestral", 6: "Trimestral",
                7: "Semestral", 8: "Anual", 9: "Un solo pago", 10: "No hay periodicidad", 11: "Otra"}

METRICAS = {
    "Modelo 1 · línea base (C = 1)": {
        "Recall (con retrasos)": (0.6421, 0.6842), "Precisión": (0.3035, 0.2632),
        "F1": (0.4122, 0.3801), "PR-AUC": (0.3768, 0.3873)},
    "Modelo 1 · C = 0,1 (mejor en validación)": {
        "Recall (con retrasos)": (0.5368, 0.7053), "Precisión": (0.3228, 0.2627),
        "F1": (0.4032, 0.3829), "PR-AUC": (0.3772, 0.3833)},
}

REQUERIDAS = ["DIRECTORIO", "objetivo_atraso", "P2097", "P2098", "P2106", "P2099S2", "P2107S1",
              "n_personas_que_ahorran", "n_personas_con_cuenta", "n_personas_laboral"]


@st.cache_data
def cargar():
    ruta = next((r for r in RUTAS if r.exists()), None)
    if ruta is None:
        return None
    d = pd.read_csv(ruta, encoding="utf-8-sig")
    faltan = [c for c in REQUERIDAS if c not in d.columns]
    if faltan:
        return faltan
    d["P2107S1"] = d["P2107S1"].replace(9999, np.nan)
    d["Entidad"] = d["P2097"].astype("Int64").map(ENTIDAD).fillna("Sin dato")
    d["Destino"] = d["P2098"].astype("Int64").map(DESTINO).fillna("Sin dato")
    d["Periodicidad"] = d["P2106"].astype("Int64").map(PERIODICIDAD).fillna("Sin dato")
    d["Plazo"] = np.where(d["P2107S1"].isna(), "Sin plazo definido", "Con plazo definido")
    d["Estado"] = np.where(d["objetivo_atraso"] == 1, "Con retrasos", "Al día")
    d["Ahorro"] = np.where(d["n_personas_que_ahorran"].fillna(0) > 0, "Con ahorro", "Sin ahorro")
    d["Cuentas"] = np.where(d["n_personas_con_cuenta"].fillna(0) > 0, "Con cuentas", "Sin cuentas")
    d["Empleo"] = np.where(d["n_personas_laboral"].fillna(0) > 0, "Con empleo", "Sin empleo")
    return d


def consultar_api(payload):
    """Espacio reservado para la API del modelo. Aún no está conectada."""
    if API_URL is None:
        return None
    # import requests
    # r = requests.post(API_URL, json=payload, timeout=10); r.raise_for_status()
    # return r.json()
    return None


df = cargar()
if df is None:
    st.error("No encuentro el CSV. Búsqueda en: " + " | ".join(str(r) for r in RUTAS))
    st.stop()
if isinstance(df, list):
    st.error(f"Al CSV le faltan columnas: {df}")
    st.stop()

# ───────────── Encabezado y filtros ─────────────
st.markdown(f"<h2 style='color:{AZUL};margin-bottom:0'>Tablero: Cumplimiento crediticio y vulnerabilidad "
            "financiera de hogares</h2>", unsafe_allow_html=True)
st.caption("Apoyo para el analista de riesgos: explora la población, consulta un crédito y revisa cuáles "
           "conviene priorizar para revisión. No aprueba ni niega créditos.")

HOGAR_OPC = {"Todos": None, "Con ahorro": ("Ahorro", "Con ahorro"), "Sin ahorro": ("Ahorro", "Sin ahorro"),
             "Con cuentas": ("Cuentas", "Con cuentas"), "Sin cuentas": ("Cuentas", "Sin cuentas"),
             "Con empleo": ("Empleo", "Con empleo"), "Sin empleo": ("Empleo", "Sin empleo")}

amin, amax = int(df["P2099S2"].min()), int(df["P2099S2"].max())
c1, c2, c3, c4 = st.columns([1.3, 1.2, 1.2, 1])
with c1:
    rango = st.slider("Año de inicio del crédito", amin, amax, (amin, amax))
with c2:
    ent = st.multiselect("Entidad acreedora", sorted(df["Entidad"].unique()), placeholder="Todas")
with c3:
    dest = st.multiselect("Destino del crédito", sorted(df["Destino"].unique()), placeholder="Todos")
with c4:
    hogar = st.selectbox("Variables del hogar", list(HOGAR_OPC))

f = df[df["P2099S2"].between(*rango)]
if ent:
    f = f[f["Entidad"].isin(ent)]
if dest:
    f = f[f["Destino"].isin(dest)]
if HOGAR_OPC[hogar]:
    col, val = HOGAR_OPC[hogar]
    f = f[f[col] == val]
if f.empty:
    st.warning("No hay créditos con esos filtros.")
    st.stop()

# ───────────── Indicadores ─────────────
pct = f["objetivo_atraso"].mean() * 100
k = st.columns(4)
k[0].metric("Créditos", f"{len(f):,}".replace(",", "."))
k[1].metric("Hogares", f"{f['DIRECTORIO'].nunique():,}".replace(",", "."))
k[2].metric("Al día", f"{100 - pct:.1f}%")
k[3].metric("Con retrasos", f"{pct:.1f}%")


# ───────────── Gráficas ─────────────
def tasa(col):
    t = f.groupby(col)["objetivo_atraso"].agg(tasa="mean", n="size").reset_index()
    t["tasa"] *= 100
    return t


def barras(t, col, titulo, horizontal=False):
    if horizontal:
        t = t.sort_values("tasa")
        fig = px.bar(t, x="tasa", y=col, orientation="h", custom_data=["n"], text_auto=".1f")
        hover = "%{y}<br>Tasa de retraso: %{x:.1f}%<br>Créditos: %{customdata[0]}<extra></extra>"
        fig.update_layout(xaxis_title="Tasa de retraso (%)", yaxis_title=None, height=120 + 28 * len(t))
    else:
        fig = px.bar(t, x=col, y="tasa", custom_data=["n"], text_auto=".1f")
        hover = "%{x}<br>Tasa de retraso: %{y:.1f}%<br>Créditos: %{customdata[0]}<extra></extra>"
        fig.update_layout(yaxis_title="Tasa de retraso (%)", xaxis_title=None, height=300)
    fig.update_traces(marker_color=MORADO, hovertemplate=hover)
    fig.update_layout(title=titulo, margin=dict(l=10, r=10, t=50, b=10))
    return fig


izq, der = st.columns([1.15, 1])
with izq:
    a, b = st.columns(2)
    with a:
        e = f["Estado"].value_counts().reindex(["Al día", "Con retrasos"]).fillna(0).reset_index()
        e.columns = ["Estado", "Créditos"]
        fig = px.bar(e, x="Estado", y="Créditos", text_auto=True)
        fig.update_traces(marker_color=MORADO)
        fig.update_layout(title="Estado de pagos", xaxis_title=None, height=300, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig)
    with b:
        st.plotly_chart(barras(tasa("Ahorro"), "Ahorro", "Atrasos según ahorro"))
    a, b = st.columns(2)
    with a:
        st.plotly_chart(barras(tasa("Cuentas"), "Cuentas", "Atrasos según cuentas"))
    with b:
        st.plotly_chart(barras(tasa("Empleo"), "Empleo", "Atrasos según empleos"))

    opc = {"Entidad acreedora": "Entidad", "Destino del crédito": "Destino",
           "Periodicidad de pago": "Periodicidad", "Plazo": "Plazo"}
    sel = st.selectbox("Tasa de retraso según característica del crédito", list(opc))
    st.plotly_chart(barras(tasa(opc[sel]), opc[sel], f"Atrasos según {sel.lower()}", horizontal=sel != "Plazo"))
    st.caption("Los grupos con pocos créditos dan tasas inestables; el número de créditos aparece al pasar el cursor.")

# ───────────── Comparación de modelos y consulta por API ─────────────
with der:
    st.subheader("Comparación entre modelos")
    cfg = st.selectbox("Configuración", list(METRICAS))
    m = METRICAS[cfg]
    tabla = pd.DataFrame([{"Métrica": k_, "Modelo A": f"{a_:.3f}", "Modelo B": f"{b_:.3f}",
                           "Mejora B−A": f"{b_ - a_:+.3f}"} for k_, (a_, b_) in m.items()])
    st.dataframe(tabla, hide_index=True)
    dr = m["Recall (con retrasos)"][1] - m["Recall (con retrasos)"][0]
    dp = m["Precisión"][1] - m["Precisión"][0]
    cumple = dr >= 0.05 and dp >= -0.03
    msg = (f"Meta del equipo (Δ recall ≥ +0,05 y Δ precisión ≥ −0,03): Δ recall = {dr:+.3f}, "
           f"Δ precisión = {dp:+.3f}. " + ("Se cumple." if cumple else "No se cumple."))
    (st.success if cumple else st.warning)(msg)
    st.caption("Métricas en el conjunto de prueba, registradas en MLflow.")

    st.subheader("Consulta por API")
    with st.form("consulta"):
        x1, x2 = st.columns(2)
        valor = x1.number_input("Valor inicial (pesos)", min_value=0, value=5_000_000, step=100_000)
        cuota = x2.number_input("Cuota (pesos)", min_value=0, value=200_000, step=10_000)
        plazo = x1.number_input("Plazo (meses)", min_value=0, max_value=240, value=36)
        per_in = x2.selectbox("Periodicidad de pago", list(PERIODICIDAD.values()), index=3)
        ent_in = x1.selectbox("Entidad acreedora", list(ENTIDAD.values()))
        dest_in = x2.selectbox("Destino del crédito", list(DESTINO.values()))
        ahorran = x1.number_input("Personas que ahorran", min_value=0, max_value=20, value=1)
        cuentas = x2.number_input("Personas con cuenta", min_value=0, max_value=20, value=1)
        enviar = st.form_submit_button("Consultar modelo")

    st.markdown(f"<h4 style='color:{AZUL}'>Resultado orientativo</h4>", unsafe_allow_html=True)
    if enviar:
        payload = {"valor_inicial": valor, "cuota": cuota, "plazo_meses": plazo, "periodicidad": per_in,
                   "entidad": ent_in, "destino": dest_in, "n_personas_que_ahorran": ahorran,
                   "n_personas_con_cuenta": cuentas}
        r = consultar_api(payload)
        if r is None:
            st.write("**Probabilidad de retraso:** por calcular")
            st.write("**Prioridad de revisión:** por definir tras validación")
            st.info("La conexión con la API se habilitará en la siguiente etapa.")
            with st.expander("Datos que se enviarán a la API"):
                st.json(payload)
        else:
            st.write(r)
    else:
        st.write("Complete los datos del crédito y pulse **Consultar modelo**.")
    st.caption("Resultado orientativo: no equivale a aprobar ni rechazar un crédito.")

st.caption("Fuente: DANE – ELCO 2022. Ejercicio académico; no es un instrumento de aprobación de crédito. "
           "«Con retrasos» corresponde a P2110 = 2 (no están al día en el pago).")