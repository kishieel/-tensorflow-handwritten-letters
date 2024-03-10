#!/usr/bin/env bash

docker run -it -v $(pwd)/app:/app -v $(pwd)/data:/data  tensorflow/tensorflow:latest python /app/main.py
