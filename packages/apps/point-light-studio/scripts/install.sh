#!/usr/bin/env bash

curl -fsSL https://pixi.sh/install.sh | bash
pixi global install pre-commit
pre-commit install
