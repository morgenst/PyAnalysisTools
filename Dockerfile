FROM atlas/analysisbase:latest

ADD . /analysis/src
WORKDIR /analysis/src
RUN /bin/bash -c "whoami && \
    	      	  source /home/atlas/release_setup.sh && \
    	      	  ls && \
    	      	  pwd && \
    	      	  ls requirements.txt && \
    	      	  pip install --user -r requirements.txt"
