# Auge

A service for recognizing objects in images and requesting translations.

### Running

Running locally:

```
uvicorn src.api:app # start
uvicorn src.api:app --reload # start with reload
```

Running via docker:

```
./docker_build_tag.sh <service> latest # build
docker run --rm -e PORT=8000 -it -p 8000:8000 <service> # run
```
