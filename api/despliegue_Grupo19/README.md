# Proyecto Grupo 19 — Despliegue de soluciones analíticas (ELCO 2022)

Solución para priorizar créditos por probabilidad de retraso: **paquete del modelo → API → tablero → Docker → AWS EC2**.

- Modelo final: HistGradientBoostingClassifier, variables del crédito (conjunto A), configuración 11, umbral 0,675.
  Prueba (592 créditos, 95 retrasos): PR-AUC 0,4576 · precisión 0,5111 · recall 0,4842 · F1 0,4973. Revisando el 10 % superior se hallan 30 de 95 retrasos.
- `package-src/` modelo empaquetado (tox, tests, wheel) · `api/` FastAPI · `dashboard/` Dash · `docker-compose.yml`.
- Inicio rápido: `docker compose up --build -d` → tablero en :8050, API en :8001/docs.
- Guías: [manual de usuario](docs/manual_usuario.md) · [manual de instalación](docs/manual_instalacion.md) · [lista de verificación](docs/lista_verificacion_rubrica.md).

Fuente: DANE – ELCO 2022. Ejercicio académico; no es un instrumento de aprobación de crédito.
