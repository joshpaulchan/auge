# Auge

A service for recognizing objects in images and requesting translations.

## Installation

We use `pipenv` for managing packagins with `pyenv` for managing python versions. To setup this project, run:

```shell
$ pipenv sync
```

## Usage

To use, first start up the server then send a request with a file to infer.

Running locally:

```shell
$ uvicorn src.api:app # start the server in this terminal
$ curl --request POST \
  --url 'http://localhost:8000/detect?output=german' \
  -F "image=@/Users/joshpaulchan/Downloads/run.jpg"
```

## Development

### Running in Reload Mode

```shell
$ uvicorn src.api:app --reload # start with reload
```

### Running via Docker

Running via docker:

```shell
$ ./docker_build_tag.sh <service> latest # build
$ docker run --rm -it -p 8000:8000 <service> # run
```

### Running Tests

We use `pytest`.

```shell
$ pytest
```


## Roadmap

- [ ] fix/retrain the model since it appears to have been generated with `pickle` for an older version of python and FastAI, and cannot be successfully deserialized:

```
AttributeError: Custom classes or functions exported with your `Learner` not available in namespace.\Re-declare/import before loading:
	Can't get attribute 'FlattenedLoss' on <module 'fastai.layers' from '~/workspace/auge/.venv/lib/python3.10/site-packages/fastai/layers.py'>
```

- [ ]  parametrize so we can easily support multiple models and formats (domain agnostic and specific)