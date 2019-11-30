FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

# default is app.main
ENV MODULE_NAME="src.api"

COPY . /app