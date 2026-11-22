FROM python:3.12-slim

RUN adduser --disabled-password --gecos "" dash-user
WORKDIR /opt/dashboard

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=8050

COPY app/requirements.txt app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r app/requirements.txt

COPY app /opt/dashboard/app
RUN chmod +x /opt/dashboard/app/run.sh && chown -R dash-user:dash-user /opt/dashboard
USER dash-user

WORKDIR /opt/dashboard/app
EXPOSE 8050
CMD ["bash", "./run.sh"]
