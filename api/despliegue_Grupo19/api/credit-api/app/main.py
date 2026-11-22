from typing import Any

from fastapi import APIRouter, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from loguru import logger

from app import __version__
from app.api import api_router
from app.config import settings, setup_app_logging

setup_app_logging(config=settings)

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=__version__,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
)

root_router = APIRouter()


@root_router.get("/")
def index(request: Request) -> Any:
    """Página de inicio con enlace a la documentación."""
    body = (
        "<html><body style='padding: 20px; font-family: sans-serif;'>"
        f"<h1>{settings.PROJECT_NAME}</h1>"
        f"<p>Versión {__version__}. Sirve el modelo Gradient Boosting (variables del crédito) "
        "que prioriza créditos por su probabilidad de retraso en el pago.</p>"
        "<p>Documentación interactiva: <a href='/docs'>/docs</a></p>"
        "<p>Estado: <a href='/api/v1/health'>/api/v1/health</a> · "
        "Ficha del modelo: <a href='/api/v1/model-info'>/api/v1/model-info</a></p>"
        "</body></html>"
    )
    return HTMLResponse(content=body)


app.include_router(api_router, prefix=settings.API_V1_STR)
app.include_router(root_router)

if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(o) for o in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

if __name__ == "__main__":
    import uvicorn

    logger.warning("Ejecución en modo desarrollo; en producción use run.sh")
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="debug")
