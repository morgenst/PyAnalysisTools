#!/usr/bin/env bash

if [[ $(hostname -s) = *lxplus* ]] || [[ $(hostname -s) == *romanescu* ]] || [[ $(hostname -s) = *pc-tbed-pub-* ]]; then
    export PYTHONPATH=~/pythonmodules/lib/python2.7/site-packages/:$PYTHONPATH
fi

export PYTHONPATH=${PWD}:${PYTHONPATH}
