FROM python:3.12-slim

# Usuario sin privilegios
RUN adduser --disabled-password --gecos "" api-user
WORKDIR /opt/credit-api

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=8001

# Dependencias primero (aprovecha la caché de capas); el .whl del modelo está en model-pkg/
COPY credit-api/requirements.txt credit-api/requirements.txt
COPY credit-api/model-pkg credit-api/model-pkg
RUN pip install --no-cache-dir --upgrade pip && cd credit-api && pip install --no-cache-dir -r requirements.txt

COPY credit-api /opt/credit-api/credit-api
RUN chmod +x /opt/credit-api/credit-api/run.sh && chown -R api-user:api-user /opt/credit-api
USER api-user

WORKDIR /opt/credit-api/credit-api
EXPOSE 8001
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s CMD python -c "import urllib.request as u;u.urlopen('http://localhost:8001/api/v1/health')" || exit 1
CMD ["bash", "./run.sh"]
