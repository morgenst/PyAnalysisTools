if [[ $(hostname -s) = *lxplus* ]] || [[ $(hostname -s) == *romanescu* ]] || [[ $(hostname -s) = *pc-tbed-pub-* ]]; then
    export PYTHONPATH=~/pythonmodules/lib/python2.7/site-packages/:$PYTHONPATH
fi

export CWD="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH=${CWD}:${PYTHONPATH}

chmod -R 755 ${CWD}/run_scripts/*.py

export PATH=${PATH}:${CWD}/run_scripts/

if [[ "$1" != "disable_dep_check" ]]; then
    python ${CWD}/.check_dependencies.py
else
    python ${CWD}/.check_dependencies.py --basic
fi