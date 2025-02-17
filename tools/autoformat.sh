#!/bin/bash

# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company.

set -ex pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
CHECK_ONLY=${CHECK_ONLY:-false}

export BASE_REF='v1.19.0_next'
if [ ! -z ${GIT_TARGET_BRANCH} ]; then
    BASE_REF="${GIT_TARGET_BRANCH}"
fi

CHANGED_FILES=$(git diff --name-only --diff-filter=d --merge-base $(git remote)/${BASE_REF} megatron/core | grep '\.py$' || true)
ADDITIONAL_ARGS=""
ADDITIONAL_BLACK_ARGS=""

if [[ $CHECK_ONLY == true ]]; then
    ADDITIONAL_ARGS="--check"
    ADDITIONAL_BLACK_ARGS="--diff"
fi

# for now we just format core
if [[ -n "$CHANGED_FILES" ]]; then
    black $ADDITIONAL_ARGS $ADDITIONAL_BLACK_ARGS --verbose $CHANGED_FILES
    isort $ADDITIONAL_ARGS $CHANGED_FILES
else
    echo Changeset is empty, all good.
fi
