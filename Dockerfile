# ─────────────────────────────────────────────────────────────
# IRAS Scanner — image de base (CLI uniquement, pas de web ici)
# ─────────────────────────────────────────────────────────────

# Image de base : Python "slim" = Debian allégé + Python préinstallé.
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

# --no-cache-dir : évite de stocker le cache pip dans l'image (l'image
# finale n'a pas besoin de ce cache, ça économise de l'espace disque).
RUN pip install --no-cache-dir -r requirements.txt

# On copie tout le reste du code. Cette étape sera
# invalidée (re-exécutée) à chaque changement de code.
COPY ./iras_scanner_unified .

# Point d'entrée : lance le vrai script principal du projet, celui qui
# fait réellement `if __name__ == "__main__"` (cli.py n'est qu'un module
# de définitions, il n'est jamais exécuté seul).
# Les arguments passés à `docker run` après le nom de l'image (ex:
# --mode rtltcp --rtl-tcp-host ...) sont transmis ici automatiquement
# grâce à la forme "exec" (liste), qui laisse ENTRYPOINT + CMD se
# combiner proprement.
ENTRYPOINT ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]