#!/usr/bin/env csh

if ( `hostname -s` =~ *"lxplus"* || `hostname -s` =~ *"stbc"* || `hostname -s` =~ *"romanescu"* || `hostname -s` =~ *"pc-tbed-pub-"*) then
    echo "setup pytools"
    lsetup "lcgenv -p LCG_latest"
endif

if ( `hostname -s` =~ *"lxplus"* || `hostname -s` =~ *"romanescu"* || `hostname -s` =~ *"pc-tbed-pub-"*) then
        setenv PYTHONPATH ~/pythonmodules/lib/python2.7/site-packages/:$PYTHONPATH
endif

setenv PYTHONPATH ${PWD}:${PYTHONPATH}
setenv PYTHONPATH /afs/cern.ch/user/m/morgens/yaml/lib/python2.7/site-packages/:${PYTHONPATH}