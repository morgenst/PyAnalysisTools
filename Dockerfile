FROM atlas/analysisbase:latest

ADD ./src /analysis/src
WORKDIR /analysis/src
RUN /bin/bash -c "whoami && \
    	      	  source /home/atlas/release_setup.sh && \
    	      	  pip install --user requirements"
