#!/bin/bash

# Make /fastdata directory if it doesn't exist
DATA_DIR="/data/$(whoami)/distilbert/dataset"
if [ ! -d "$DATA_DIR" ]; then
  # Make personal fastdata directory
  mkdir -p "$DATA_DIR"
  # Make it private to your user
  chmod 700 "$DATA_DIR"
fi

cp -R distilbert/dataset/* "$DATA_DIR"
