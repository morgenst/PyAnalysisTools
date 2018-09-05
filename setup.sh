#!/usr/bin/env bash

if [[ $(hostname -s) = *lxplus* ]] || [[ $(hostname -s) == *romanescu* ]] || [[ $(hostname -s) = *pc-tbed-pub-* ]]; then
    export PYTHONPATH=~/pythonmodules/lib/python2.7/site-packages/:$PYTHONPATH
fi

if [[ $(hostname -s) = *lxplus* ]] || [[ $(hostname -s) == *romanescu* ]] || [[ $(hostname -s) = *pc-tbed-pub-* ]]; then
    CWD=`dirname $(readlink -f "$0")`
elif [[ $(hostname -s) = *stbc-* ]]; then
    CWD=`dirname $(readlink -f "${BASH_SOURCE[0]}")`
else
    called=$_
    [[ $called != $0 ]]
    CWD=`dirname "${BASH_SOURCE[@]}"`
fi

export PYTHONPATH=${CWD}:${PYTHONPATH}
export PATH=${CWD}/run_scripts:${PATH}