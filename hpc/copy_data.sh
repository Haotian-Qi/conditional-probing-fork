#!/bin/bash

# Make /fastdata directory if it doesn't exist
FASTDATA_DIR="/fastdata/$(whoami)"
if [ ! -d "$FASTDATA_DIR" ]; then
  # Make personal fastdata directory
  mkdir "$FASTDATA_DIR"
  # Make it private to your user
  chmod 700 "$FASTDATA_DIR"
fi

cp -R distilbert/dataset "$FASTDATA_DIR/distilbert/dataset"
