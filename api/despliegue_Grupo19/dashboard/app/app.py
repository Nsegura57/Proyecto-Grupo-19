"""Tablero del Grupo 19: explorar créditos, consultar uno, priorizar un lote y ver los modelos."""
import base64
import io
import os
from pathlib import Path

import dash
import pandas as pd
import plotly.graph_objects as go
import requests
from dash import Input, Output, State, dash_table, dcc, html

MORADO, AZUL, GRIS, ROJO = "#7570E6", "#3D35B8", "#8A8FA3", "#D9534F"
DATOS = Path(__file__).resolve().parent / "data"
API_URL = os.getenv("API_URL", "localhost")
API_PORT = os.getenv("API_PORT", "8001")
API_BASE = API_URL.rstrip("/") if API_URL.startswith("http") else f"http://{API_URL}:{API_PORT}"
V1 = f"{API_BASE}/api/v1"

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
FEATURES = ["P2097", "P2098", "P2106", "P2099S2", "P2104", "P2105S1", "P2107S1"]

# ---------------------------------------------------------------- datos
df = pd.read_csv(DATOS / "creditos_tablero.csv")
df["Entidad"] = df["P2097"].map(ENTIDAD)
df["Destino"] = df["P2098"].map(DESTINO)
df["Periodicidad"] = df["P2106"].map(PERIODICIDAD)
df["Plazo"] = df["P2107S1"].apply(lambda v: "Con plazo definido" if pd.notna(v) and v != 9999 else "Sin plazo definido")
df["Estado"] = df["objetivo_atraso"].map({0: "Al día", 1: "Con retrasos"})
df["Ahorro"] = df["n_personas_que_ahorran"].gt(0).map({True: "Con ahorro", False: "Sin ahorro"})
df["Cuentas"] = df["n_personas_con_cuenta"].gt(0).map({True: "Con cuenta", False: "Sin cuenta"})
df["Empleo"] = df["n_personas_trabajo_semana"].gt(0).map({True: "Con empleo", False: "Sin empleo"})
cmp_modelos = pd.read_csv(DATOS / "comparacion_modelos.csv")
cmp_ab = pd.read_csv(DATOS / "comparacion_A_vs_B.csv")
imp = pd.read_csv(DATOS / "importancia_permutacion_A.csv")
prio_final = pd.read_csv(DATOS / "priorizacion_final.csv")
lote_ejemplo = pd.read_csv(DATOS / "lote_ejemplo_prueba.csv")
TASA_BASE = df["objetivo_atraso"].mean()
ANIO_MIN, ANIO_MAX = int(df["P2099S2"].min()), int(df["P2099S2"].max())
CARACT = {"Entidad": "Entidad", "Destino": "Destino", "Periodicidad": "Periodicidad", "Plazo": "Plazo"}

# ---------------------------------------------------------------- utilidades
def fmt_pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def tarjeta(titulo, valor, color=AZUL):
    return html.Div([html.Div(titulo, className="kpi-t"), html.Div(valor, className="kpi-v", style={"color": color})],
                    className="kpi")


def figura_vacia(msg="Sin datos para los filtros elegidos"):
    f = go.Figure()
    f.add_annotation(text=msg, showarrow=False, font=dict(size=14, color=GRIS))
    f.update_xaxes(visible=False); f.update_yaxes(visible=False)
    f.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="white", plot_bgcolor="white")
    return f


def llamar_api(ruta, cuerpo=None):
    """GET si no hay cuerpo; POST en caso contrario. Devuelve (json, mensaje_error)."""
    try:
        r = requests.post(f"{V1}{ruta}", json=cuerpo, timeout=60) if cuerpo is not None else requests.get(f"{V1}{ruta}", timeout=15)
    except requests.RequestException:
        return None, f"No se pudo conectar con la API en {API_BASE}. Verifique que esté en ejecución."
    if r.status_code == 200:
        return r.json(), None
    try:
        det = r.json().get("detail")
    except ValueError:
        det = r.text[:200]
    if isinstance(det, list):
        det = "; ".join(f"{d.get('loc', '')}: {d.get('msg', '')}" for d in det)
    return None, f"La API respondió {r.status_code}: {det}"


def opciones(dic):
    return [{"label": f"{k} · {v}", "value": k} for k, v in dic.items()]


def tabla(frame, **kw):
    return dash_table.DataTable(
        data=frame.to_dict("records"), columns=[{"name": c, "id": c} for c in frame.columns],
        style_header={"backgroundColor": AZUL, "color": "white", "fontWeight": "bold"},
        style_cell={"fontFamily": "inherit", "padding": "6px 10px", "textAlign": "left", "whiteSpace": "normal", "height": "auto"},
        style_table={"overflowX": "auto"}, **kw)


# ---------------------------------------------------------------- diseño
app = dash.Dash(__name__, suppress_callback_exceptions=True, title="Créditos y retrasos · Grupo 19")
server = app.server

app.index_string = """<!DOCTYPE html><html><head>{%metas%}<title>{%title%}</title>{%favicon%}{%css%}
<style>
body{font-family:'Segoe UI',Roboto,Arial,sans-serif;margin:0;background:#F6F6FC;color:#222}
.wrap{max-width:1150px;margin:0 auto;padding:18px 22px 40px}
h1{color:#3D35B8;margin:6px 0 4px}h2{color:#3D35B8;margin-top:26px}
.lead{background:white;border-left:5px solid #7570E6;padding:12px 16px;border-radius:6px;margin:10px 0 14px}
.card{background:white;border-radius:10px;padding:16px 18px;box-shadow:0 1px 4px rgba(0,0,0,.08);margin:12px 0}
.kpis{display:flex;gap:12px;flex-wrap:wrap}.kpi{flex:1;min-width:170px;background:white;border-radius:10px;padding:12px 16px;box-shadow:0 1px 4px rgba(0,0,0,.08)}
.kpi-t{font-size:13px;color:#666}.kpi-v{font-size:28px;font-weight:700}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px}
label{font-weight:600;font-size:14px;display:block;margin:10px 0 4px}
.btn{background:#3D35B8;color:white;border:0;border-radius:6px;padding:10px 20px;font-size:15px;cursor:pointer;margin-top:14px}
.btn:hover{background:#7570E6}.ok{color:#1b7f3b}.bad{color:#D9534F}.small{font-size:13px;color:#666}
.res{font-size:34px;font-weight:700}.pill{display:inline-block;border-radius:20px;padding:4px 14px;color:white;font-weight:600}
.aviso{background:#FFF8E1;border-left:5px solid #F0B429;padding:10px 14px;border-radius:6px;font-size:14px}
</style></head><body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer></body></html>"""

filtros = html.Div([
    html.Div([html.Label("Año de inicio del crédito"),
              dcc.RangeSlider(id="f-anio", min=ANIO_MIN, max=ANIO_MAX, step=1, value=[1997, ANIO_MAX],
                              marks={y: str(y) for y in range(1990 if ANIO_MIN < 1990 else 2000, ANIO_MAX + 1, 5)}
                              | {ANIO_MAX: str(ANIO_MAX)}, tooltip={"placement": "bottom"})]),
    html.Div([html.Div([html.Label("Entidad acreedora"), dcc.Dropdown(id="f-entidad", options=list(ENTIDAD.values()), multi=True, placeholder="Todas")]),
              html.Div([html.Label("Destino del crédito"), dcc.Dropdown(id="f-destino", options=list(DESTINO.values()), multi=True, placeholder="Todos")]),
              html.Div([html.Label("Hogar"), dcc.Dropdown(id="f-hogar", clearable=False, value="Todos",
                        options=["Todos", "Con ahorro", "Sin ahorro", "Con cuenta", "Sin cuenta", "Con empleo", "Sin empleo"])])],
             className="grid2"),
], className="card")

tab_explorar = html.Div([
    html.P("Explore cómo se distribuyen los créditos de los hogares y en qué casos aparecen más retrasos. "
           "Los filtros afectan todas las cifras y gráficos de esta pestaña.", className="small"),
    filtros,
    html.Div(id="kpis", className="kpis"),
    html.Div([html.Div(dcc.Graph(id="g-estado"), className="card"), html.Div(dcc.Graph(id="g-anio"), className="card")], className="grid2"),
    html.Div([html.Label("Características del crédito"),
              dcc.Dropdown(id="sel-caract", options=list(CARACT), value="Entidad", clearable=False, style={"maxWidth": "320px"}),
              dcc.Graph(id="g-caract")], className="card"),
    html.Div([html.Label("Hogar"), html.P("Porcentaje de créditos con retraso según condición del hogar.", className="small"),
              dcc.Graph(id="g-hogar")], className="card"),
])

tab_consultar = html.Div([
    html.P("Ingrese los datos de un crédito. El modelo estima la probabilidad de retraso, su posición dentro de la "
           "población de referencia y la prioridad de revisión sugerida.", className="small"),
    html.Div([
        html.Div([
            html.Div([html.Label("Entidad acreedora"), dcc.Dropdown(id="c-entidad", options=opciones(ENTIDAD), value=1, clearable=False)]),
            html.Div([html.Label("Destino del crédito"), dcc.Dropdown(id="c-destino", options=opciones(DESTINO), value=2, clearable=False)]),
            html.Div([html.Label("Periodicidad de pago"), dcc.Dropdown(id="c-period", options=opciones(PERIODICIDAD), value=4, clearable=False)]),
            html.Div([html.Label("Año de inicio del crédito"), dcc.Input(id="c-anio", type="number", min=1950, max=2030, step=1, value=2018, style={"width": "100%"})]),
            html.Div([html.Label("Valor inicial del crédito (pesos)"), dcc.Input(id="c-valor", type="number", min=0, value=25000000, style={"width": "100%"})]),
            html.Div([html.Label("Valor de la cuota (pesos, opcional)"), dcc.Input(id="c-cuota", type="number", min=0, value=450000, style={"width": "100%"})]),
            html.Div([html.Label("Plazo en meses (opcional)"), dcc.Input(id="c-plazo", type="number", min=0, value=60, style={"width": "100%"})]),
        ], className="grid2"),
        html.Button("Calcular", id="c-btn", n_clicks=0, className="btn"),
    ], className="card"),
    html.Div(id="c-resultado"),
])

tab_priorizar = html.Div([
    html.P("Ordene un lote de créditos de mayor a menor probabilidad de retraso para decidir cuáles revisar primero.", className="small"),
    html.Div([
        html.Label("Origen del lote"),
        dcc.RadioItems(id="p-origen", value="ejemplo", inline=True, inputStyle={"marginRight": "6px", "marginLeft": "14px"},
                       options=[{"label": "Lote de ejemplo (592 créditos de prueba)", "value": "ejemplo"},
                                {"label": "Cargar mi archivo CSV", "value": "csv"}]),
        html.Div(id="p-upload-box", children=dcc.Upload(id="p-upload", children=html.Div(["Arrastre o ", html.A("seleccione un CSV")]),
                 style={"border": "2px dashed #7570E6", "borderRadius": "8px", "padding": "18px", "textAlign": "center", "marginTop": "10px"})),
        html.P(["El CSV debe tener las columnas: ", html.Code(", ".join(FEATURES)),
                " (P2105S1 y P2107S1 pueden ir vacías). Si trae la columna objetivo_atraso se mostrará cuántos retrasos se encuentran."], className="small"),
        html.Button("Priorizar", id="p-btn", n_clicks=0, className="btn"),
        dcc.Store(id="p-store"),
    ], className="card"),
    html.Div(id="p-resultado"),
])

tab_modelos = html.Div([
    html.P("Comparación de los tres algoritmos con el conjunto A (variables del crédito) sobre la partición de prueba. "
           "Se resalta el modelo final desplegado.", className="small"),
    html.Div([html.H3("Comparación de algoritmos (prueba)"),
              tabla(cmp_modelos.drop(columns="final").rename(columns={"modelo": "Modelo", "configuracion": "Configuración", "pr_auc": "PR-AUC", "roc_auc": "ROC-AUC", "precision": "Precisión", "recall": "Recall", "f1": "F1"}).assign(Modelo=lambda t: [m + ("(Modelo Elegido)" if f else "") for m, f in zip(t["Modelo"], cmp_modelos["final"])]),
                    style_data_conditional=[{"if": {"filter_query": '{Modelo} contains "final"'}, "backgroundColor": "#E8E6FB", "fontWeight": "bold"}]),
              html.P("Umbral de clasificación: el del Gradient Boosting es 0,675 (máximo F1 en validación).", className="small")], className="card"),
    html.Div([html.H3("¿Aportan las variables del hogar? Gradient Boosting, A vs B"),
              tabla(cmp_ab.rename(columns={"conjunto": "Conjunto", "pr_auc": "PR-AUC", "roc_auc": "ROC-AUC", "precision": "Precisión", "recall": "Recall", "f1": "F1"})),
              html.P("Criterio de éxito: Δrecall ≥ +0,05 y Δprecisión ≥ −0,03. Resultado: Δrecall = +0,0105 y Δprecisión = −0,1754, "
                     "por lo que el criterio no se cumple y se conserva el conjunto A.", className="small")], className="card"),
    html.Div([html.H3("Qué variables pesan más (importancia por permutación, validación)"), dcc.Graph(figure=go.Figure(
        go.Bar(x=imp["importancia"][::-1], y=(imp["variable"] + " · " + imp["descripcion"])[::-1], orientation="h", marker_color=MORADO))
        .update_layout(height=340, margin=dict(l=10, r=10, t=10, b=30), paper_bgcolor="white", plot_bgcolor="white", xaxis_title="Caída de PR-AUC al permutar"))],
        className="card"),
    html.Div([html.H3("Priorización: revisar solo una parte de los créditos"),
              tabla(prio_final.assign(precision_top=lambda t: t.precision_top.map(fmt_pct), recall_top=lambda t: t.recall_top.map(fmt_pct), creditos_por_retraso=lambda t: t.creditos_por_retraso.round(1))
                    .rename(columns={"porcentaje_revisado": "% revisado", "creditos_revisados": "Créditos revisados", "retrasos_encontrados": "Retrasos encontrados (de 95)", "precision_top": "Precisión", "recall_top": "Retrasos capturados", "creditos_por_retraso": "Créditos por retraso"})),
              html.P("Revisando el 10 % de los créditos con mayor probabilidad se encuentran 30 de los 95 retrasos de la partición de prueba.", className="small")], className="card"),
])

app.layout = html.Div(className="wrap", children=[
    html.H1("Créditos de los hogares y retrasos en el pago"),
    html.Div("Entre los créditos reportados en ELCO 2022, ¿cuáles conviene revisar primero por su probabilidad de retraso? ", className="lead"),
    html.Div(id="estado-api", className="small"),
    dcc.Tabs(id="tabs", value="t1", colors={"primary": AZUL, "background": "#ECEBFA", "border": "#ddd"}, children=[
        dcc.Tab(label="1. Explorar", value="t1", children=tab_explorar),
        dcc.Tab(label="2. Consultar un crédito", value="t2", children=tab_consultar),
        dcc.Tab(label="3. Priorizar un lote", value="t3", children=tab_priorizar),
        dcc.Tab(label="4. Modelos", value="t4", children=tab_modelos),
    ]),
    html.P("Fuente: DANE – ELCO 2022. Ejercicio académico; no es un instrumento de aprobación de crédito.", className="small", style={"marginTop": "28px"}),
])


# ---------------------------------------------------------------- callbacks
@app.callback(Output("estado-api", "children"), Input("tabs", "value"))
def estado_api(_):
    h, err = llamar_api("/health")
    if err:
        return html.Span(f"● API no disponible ({API_BASE}). La exploración funciona; consultar y priorizar requieren la API.", className="bad")
    return html.Span(f"● API conectada · modelo v{h['model_version']}", className="ok")


def filtrar(anio, ent, dest, hogar):
    d = df[(df["P2099S2"] >= anio[0]) & (df["P2099S2"] <= anio[1])]
    if ent:
        d = d[d["Entidad"].isin(ent)]
    if dest:
        d = d[d["Destino"].isin(dest)]
    if hogar != "Todos":
        d = d[(d["Ahorro"] == hogar) | (d["Cuentas"] == hogar) | (d["Empleo"] == hogar)]
    return d


def barras_tasa(d, col, titulo):
    if d.empty:
        return figura_vacia()
    g = d.groupby(col)["objetivo_atraso"].agg(["mean", "size"]).sort_values("mean")
    f = go.Figure(go.Bar(x=g["mean"] * 100, y=g.index, orientation="h", marker_color=MORADO,
                         text=[f"{m * 100:.1f}% (n={n})" for m, n in zip(g["mean"], g["size"])], textposition="outside", cliponaxis=False))
    f.update_layout(title=titulo, height=max(280, 34 * len(g) + 90), margin=dict(l=10, r=70, t=50, b=30), xaxis_title="% con retrasos",
                    paper_bgcolor="white", plot_bgcolor="white")
    return f


@app.callback(Output("kpis", "children"), Output("g-estado", "figure"), Output("g-anio", "figure"),
              Output("g-caract", "figure"), Output("g-hogar", "figure"),
              Input("f-anio", "value"), Input("f-entidad", "value"), Input("f-destino", "value"),
              Input("f-hogar", "value"), Input("sel-caract", "value"))
def explorar(anio, ent, dest, hogar, caract):
    d = filtrar(anio, ent, dest, hogar)
    n = len(d)
    if n == 0:
        return [tarjeta("Créditos", "0")], figura_vacia(), figura_vacia(), figura_vacia(), figura_vacia()
    tasa = d["objetivo_atraso"].mean()
    kpis = [tarjeta("Créditos", f"{n:,}".replace(",", ".")), tarjeta("Hogares", f"{d['DIRECTORIO'].nunique():,}".replace(",", ".")),
            tarjeta("Al día", fmt_pct(1 - tasa)), tarjeta("Con retrasos", fmt_pct(tasa), ROJO)]
    cnt = d["Estado"].value_counts().reindex(["Al día", "Con retrasos"]).fillna(0)
    f1 = go.Figure(go.Pie(labels=cnt.index, values=cnt.values, hole=.55, marker_colors=[MORADO, ROJO], sort=False))
    f1.update_layout(title="Estado de pagos", height=320, margin=dict(l=10, r=10, t=50, b=10), paper_bgcolor="white")
    y = d.groupby("P2099S2")["objetivo_atraso"].agg(["mean", "size"])
    f2 = go.Figure(go.Scatter(x=y.index, y=y["mean"] * 100, mode="lines+markers", marker=dict(size=np_size(y["size"])), line_color=AZUL,
                              customdata=y["size"], hovertemplate="Año %{x}<br>%{y:.1f}% con retrasos<br>n=%{customdata}<extra></extra>"))
    f2.update_layout(title="% con retrasos según año de inicio (tamaño = n)", height=320, margin=dict(l=10, r=10, t=50, b=30),
                     yaxis_title="% con retrasos", paper_bgcolor="white", plot_bgcolor="white")
    hog = pd.concat([d.groupby(c)["objetivo_atraso"].agg(["mean", "size"]) for c in ("Ahorro", "Cuentas", "Empleo")])
    fh = go.Figure(go.Bar(x=hog.index, y=hog["mean"] * 100, marker_color=MORADO, text=[f"{m * 100:.1f}% (n={s})" for m, s in zip(hog["mean"], hog["size"])], textposition="outside"))
    fh.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=30), yaxis_title="% con retrasos", paper_bgcolor="white", plot_bgcolor="white")
    return kpis, f1, f2, barras_tasa(d, caract, f"% con retrasos según {caract.lower()}"), fh


def np_size(s):
    return [max(5, min(18, 4 + n ** 0.5)) for n in s]


COLOR_PRIO = {"Alta": ROJO, "Media": "#F0B429", "Baja": "#2E9F5B"}


@app.callback(Output("c-resultado", "children"), Input("c-btn", "n_clicks"),
              State("c-entidad", "value"), State("c-destino", "value"), State("c-period", "value"), State("c-anio", "value"),
              State("c-valor", "value"), State("c-cuota", "value"), State("c-plazo", "value"), prevent_initial_call=True)
def consultar(_, ent, dest, per, anio, valor, cuota, plazo):
    if anio is None or valor is None:
        return html.Div("Complete el año de inicio y el valor inicial del crédito.", className="aviso")
    credito = {"P2097": ent, "P2098": dest, "P2106": per, "P2099S2": int(anio), "P2104": valor, "P2105S1": cuota, "P2107S1": plazo}
    r, err = llamar_api("/predict", {"inputs": [credito]})
    if err:
        return html.Div(err, className="aviso")
    p, top, prio = r["predictions"][0], r["top_percent"][0], r["priority"][0]
    clase = "Se predice retraso" if r["labels"][0] else "No se predice retraso"
    mismos = df[(df["P2097"] == ent)]
    ctx = (f"Entre los {len(mismos)} créditos de esta entidad en los datos, {fmt_pct(mismos['objetivo_atraso'].mean())} tuvo retrasos "
           f"(en total, {fmt_pct(TASA_BASE)}).") if len(mismos) else ""
    return html.Div([
        html.H3("Resultado"),
        html.Div([html.Span(f"{p * 100:.1f}%", className="res", style={"color": AZUL}), html.Span(" probabilidad estimada de retraso", className="small")]),
        html.P([html.Span(f"Prioridad {prio}", className="pill", style={"background": COLOR_PRIO[prio]}), "  ", clase,
                f" (umbral {r['threshold']})"], style={"marginTop": "10px"}),
        html.P(f"El crédito queda entre el {top:.1f}% con mayor probabilidad de la población de referencia."),
        html.P(ctx, className="small"),
    ], className="card")


def leer_csv(contenido):
    _, b64 = contenido.split(",", 1)
    return pd.read_csv(io.BytesIO(base64.b64decode(b64)))


@app.callback(Output("p-upload-box", "style"), Input("p-origen", "value"))
def mostrar_upload(o):
    return {} if o == "csv" else {"display": "none"}


@app.callback(Output("p-resultado", "children"), Output("p-store", "data"), Input("p-btn", "n_clicks"),
              State("p-origen", "value"), State("p-upload", "contents"), prevent_initial_call=True)
def priorizar(_, origen, contenido):
    if origen == "csv":
        if not contenido:
            return html.Div("Cargue un archivo CSV.", className="aviso"), None
        try:
            lote = leer_csv(contenido)
        except Exception:
            return html.Div("No se pudo leer el archivo como CSV.", className="aviso"), None
        faltan = [c for c in ("P2097", "P2098", "P2106", "P2099S2", "P2104") if c not in lote.columns]
        if faltan:
            return html.Div(f"Faltan columnas obligatorias: {', '.join(faltan)}.", className="aviso"), None
    else:
        lote = lote_ejemplo.copy()
    entradas = lote[[c for c in FEATURES if c in lote.columns]].astype(object).where(lambda t: t.notna(), None).to_dict("records")
    r, err = llamar_api("/priorizar", {"inputs": entradas})
    if err:
        return html.Div(err, className="aviso"), None
    rk = pd.DataFrame(r["ranking"])
    rk = rk.join(lote.reset_index(drop=True), on="indice")
    n = len(rk)
    tiene_y = "objetivo_atraso" in rk.columns and rk["objetivo_atraso"].notna().all()
    filas = []
    for pct in (10, 20, 30):
        k = max(1, -(-n * pct // 100))
        fila = {"% revisado": f"{pct}%", "Créditos a revisar": k, "Probabilidad mínima": f"{rk['probabilidad'].iloc[k - 1] * 100:.1f}%"}
        if tiene_y:
            enc = int(rk["objetivo_atraso"].iloc[:k].sum())
            fila["Retrasos encontrados"] = f"{enc} de {int(rk['objetivo_atraso'].sum())}"
        filas.append(fila)
    vista = rk.assign(probabilidad=(rk["probabilidad"] * 100).round(1), percentil_top=rk["percentil_top"].round(1))
    cols = ["posicion", "grupo", "prioridad", "probabilidad", "percentil_top"] + [c for c in FEATURES if c in vista.columns]
    if tiene_y:
        cols.append("objetivo_atraso")
    vista = vista[cols].rename(columns={"posicion": "Posición", "grupo": "Grupo", "prioridad": "Prioridad", "probabilidad": "Prob. retraso (%)",
                                        "percentil_top": "Top (%)", "objetivo_atraso": "Retraso real"})
    out = [html.Div([html.H3(f"Resultado: {n} créditos ordenados"), tabla(pd.DataFrame(filas)),
                     html.P("Revise los créditos de arriba hacia abajo: los primeros tienen mayor probabilidad de retraso.", className="small"),
                     html.Button("Descargar ranking (CSV)", id="p-dl-btn", className="btn"), dcc.Download(id="p-dl")], className="card"),
           html.Div([html.H3("Ranking (primeros 100)"), tabla(vista.head(100), page_size=15)], className="card")]
    return out, vista.to_dict("records")


@app.callback(Output("p-dl", "data"), Input("p-dl-btn", "n_clicks"), State("p-store", "data"), prevent_initial_call=True)
def descargar(_, datos):
    if not datos:
        return dash.no_update
    return dcc.send_data_frame(pd.DataFrame(datos).to_csv, "ranking_creditos.csv", index=False)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8050")), debug=False)
