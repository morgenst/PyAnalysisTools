#!/usr/bin/env bash

doUpdate=false
while getopts ":u" opt; do
  case $opt in
    u)
      echo "updating spdlog and tablulate"
      doUpdate=true
      ;;
    \?)
      echo "Invalid option"
      ;;
  esac
done
shift 
#hack to ignore cmd line args for further source commands
OPTIND=0

#set env variables
ANALYSIS_DIR=${PWD}
export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
if [ -z "${ATLAS_LOCAL_SETUP_OPTIONS}" ]; then
    source ${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh &> /dev/null
fi

if [[ $(hostname -s) = *lxplus* ]]; then
	lsetup git &> /dev/null
fi

cd ../

asetup AnalysisBase,21.2.56,gcc62,here
cd - &> /dev/null

#load PyAnalyisTools
python -c "import PyAnalysisTools" &> /dev/null
if [ $? -ne  0 ]; then
	source ../PyAnalysisTools/setup.sh
fi

#setup package environment variables
python .setup_package.py
source .set_env.sh
rm .set_env.sh
	
cd ../ELCore
export PYTHONPATH=${PWD}/python:${PYTHONPATH}

cd ${ANALYSIS_DIR}
