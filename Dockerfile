#
# Docker configuration.
#
# The configuration needs to be a bit elaborate, as it has to work correctly
# both as part of the CI build, and when used "standalone".
#  - When used as part of the CI, it picks up the RPM file that was produced
#    by one of the previous build steps, and installs that into the output
#    image.
#  - When used outside of the CI, it builds the code of the repository itself,
#    producing an RPM file in the process, and then installs that RPM file on
#    the system.
#

# Set up which image we base the analysis repository's image on.
FROM atlas/analysisbase:21.2.19

# Set TERM
ENV TERM=xterm-256color

# Copy everything into the image.
COPY . /home/atlas/source/
# Switch to the root user, as the source directory at this point can only be
# modified with super user priviliges.
USER root

# If this included an RPM file as well, then we're running as part of the CI
# system, and we'll just use that RPM file. But if not, then the image build
# has to produce the RPM file itself.
# RUN rpmFile=(/home/atlas/source/*.rpm); \
#    if [ ! -e "${rpmFile[0]}" ] 2>&1; then \
#    source /home/atlas/release_setup.sh && \
#    cd /home/atlas/source/ && ls -l setup.sh && source /home/atlas/source/setup.sh \
#    ; fi

#RUN ls -l source/
# Install the RPM file:
#RUN rpm -i /home/atlas/source/*.rpm

# Clean up:
#RUN rm -rf /home/atlas/source

# Switch back to the ATLAS account:
USER atlas

RUN source /home/atlas/source/setup.sh

# Set up the environment setup script:
# RUN mv /afs/cern.ch/work/m/morgens/buildTest/release_setup.sh /afs/cern.ch/work/m/morgens/buildTest/analysis_release_setup.sh
# COPY docker/release_setup.sh /afs/cern.ch/work/m/morgens/buildTest/
#COPY docker/release_setup.sh /home/atlas/

# Set up the message of the day:
#COPY docker/motd /etc/

# Start the image with BASH by default, after having printed the message
# of the day.
CMD cat /etc/motd && /bin/bash