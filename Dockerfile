FROM jupyter/scipy-notebook:1145fb1198b2

USER root
COPY requirements.txt /tmp/

# From: https://hub.docker.com/r/riordan/docker-jupyter-scipy-notebook-libpostal/~/dockerfile/
# LIBPOSTAL
# Install Libpostal dependencies
RUN apt-get update &&\
    apt-get install -y \
        git \
        make \
        curl \
        autoconf \
        automake \
        libtool \
        pkg-config

# Download libpostal source to /usr/local/libpostal
RUN cd /usr/local && \
    git clone https://github.com/openvenues/libpostal

# Create Libpostal data directory at /var/libpostal/data
RUN cd /var && \
    mkdir libpostal && \
    cd libpostal && \
    mkdir data

# Install Libpostal from source
RUN cd /usr/local/libpostal && \
    ./bootstrap.sh && \
    ./configure --datadir=/var/libpostal/data && \
    make -j4 && \
    make install && \
  ldconfig

USER ${NB_USER}
RUN pip install -r /tmp/requirements.txt
CMD ["jupyter", "notebook", "--ip", "0.0.0.0"]
