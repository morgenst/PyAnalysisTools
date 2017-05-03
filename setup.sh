#!/usr/bin/env bash

if [[ $(hostname -s) = *lxplus* ]] || [[ $(hostname -s) = *stbc-* ]] || [[ $(hostname -s) == *romanescu* ]] ||
[[ $(hostname -s) = *pc-tbed-pub-* ]]; then
    lsetup "sft releases/pytools/1.9_python2.7-5c0ab"
fi

if [[ $(hostname -s) = *lxplus* ]] || [[ $(hostname -s) == *romanescu* ]] || [[ $(hostname -s) = *pc-tbed-pub-* ]]; then
    export PYTHONPATH=~/pythonmodules/lib/python2.7/site-packages/:$PYTHONPATH
fi

export PYTHONPATH=${PWD}:${PYTHONPATH}