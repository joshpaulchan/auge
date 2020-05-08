FROM python:3.7-slim-stretch

ENV PORT 8000

WORKDIR /opt/auge

RUN apt update
RUN apt install -y python3-dev gcc

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY src src
COPY ml ml

EXPOSE $PORT
CMD ["uvicorn", "src.api:app"]