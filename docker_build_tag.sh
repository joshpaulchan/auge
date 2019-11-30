#!/bin/bash
# Usage: ./docker_build_tag.sh <PROJECT_ID> <TAG>

if [ -z "$1" ]
then
  echo "Usage: `basename $0` <PROJECT_ID> <TAG>"
  exit $E_ARG_ERR
fi

PROJECT_ID=$1
TAG=$2

# function maps '/' in branch names to '-'
function escape_tag() {
    echo $(tr '/' '-' <<< "$1")
    return 0
}

# TODO: Detect if tag is a semver tag or other name
ESCAPED_TAG=$(escape_tag $TAG)
echo "Tagging '$PROJECT_ID' with: $ESCAPED_TAG."
docker build --rm -t $PROJECT_ID:$ESCAPED_TAG .
docker push $PROJECT_ID:$ESCAPED_TAG