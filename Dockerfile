FROM python:3.9-slim

WORKDIR /app

# System-Abhängigkeiten
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Python-Abhängigkeiten kopieren und installieren
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bot Code und Config kopieren



COPY  Hiperparameters.csv .
COPY . .
# Bot starten
CMD ["python",  "realtrade.py"]