#!/usr/bin/env bash

if [[ $(hostname -s) = *lxplus* ]] || [[ $(hostname -s) == *romanescu* ]] || [[ $(hostname -s) = *pc-tbed-pub-* ]]; then
    export PYTHONPATH=~/pythonmodules/lib/python2.7/site-packages/:$PYTHONPATH
fi

called=$_
[[ $called != $0 ]]

export PYTHONPATH=`dirname "${BASH_SOURCE[@]}"`:${PYTHONPATH}
export PATH=`dirname "${BASH_SOURCE[@]}"`/run_scripts:${PATH}
