FROM python:3.7

ENV PORT 8000

WORKDIR /opt/auge

COPY ./Pipfile* .

RUN pip install pipenv
RUN pipenv install --ignore-pipfile --deploy

EXPOSE $PORT
CMD ["uvicorn", "src.api:app"]