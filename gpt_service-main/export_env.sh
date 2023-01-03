#!/usr/bin/env bash
# shellcheck disable=SC2046
export $(grep -v '^#' .env | xargs)