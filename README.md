# Auge

A service for recognizing objects in images and requesting translations.

### Running

Running locally:

```shell
$ uvicorn src.api:app # start
$ uvicorn src.api:app --reload # start with reload
```

Running via docker:

```shell
$ ./docker_build_tag.sh <service> latest # build
$ docker run --rm -it -p 8000:8000 <service> # run
```

Testing:

```shell
$ curl --request POST \
  --url 'http://localhost:8000/detect?output=german' \
  -F "image=@/Users/joshpaulchan/Downloads/run.jpg"
```
