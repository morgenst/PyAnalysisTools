#!/usr/bin/env bash

if [[ $(hostname -s) = *lxplus* ]] || [[ $(hostname -s) = *stbc-* ]] || [[ $(hostname -s) == *romanescu* ]]; then
    lsetup "sft releases/pytools/1.9_python2.7-5c0ab"
fi

export PYTHONPATH=${PWD}:${PYTHONPATH}