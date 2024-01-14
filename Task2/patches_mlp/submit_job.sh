#!/bin/bash

# This is a wrapper script so that we can create directories for each experiment

# If parmeter not passed save in /Default/
if [ -z "$1" ]
  then
    $1 = "Default"
fi

SAVE_DIR=$1
mkdir -p $SAVE_DIR

sbatch --output=$SAVE_DIR/%x_%u_%j.out --error=$SAVE_DIR/%x_%u_%j.err job $SAVE_DIR
