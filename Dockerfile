FROM atlas/analysisbase:latest

ADD . /analysis/src
WORKDIR /analysis/src
RUN /bin/bash -c "whoami && \
    	      	  source /home/atlas/release_setup.sh && \
    	      	  pip install --user requirements"
