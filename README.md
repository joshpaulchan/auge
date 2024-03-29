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

### Adding new Models

Drop models (`*.pkl`) into the [`./ml` ](./ml) folder. NOTE: the models are pickled, and so you need to make sure you're training them with the same dependencies you're deploying them to (check pinned versions here) otherwise they'll fail to deserialize properly.


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

Update: Ok, so I've managed to get it roughly working using Pytorch directly and using the pre-built models. It should be [somewhat possible](https://docs.fast.ai/examples/migrating_pytorch_verbose.html#exporting-and-predicting) to load up my old model and get that working too, but might be better to isolate fastai to the training loop and keep it to pytorch for deployment.

- [ ]  parametrize so we can easily support multiple models and formats (domain agnostic and specific)
- [ ]  rebuild dockerfile to use pipenv instead of requirements