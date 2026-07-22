# ssl-kaldi: Kaldi (CUDA) + SSL feature extraction requirements
# Build:  docker build -t ssl-kaldi docker/
# Run:    docker run --gpus all -it ssl-kaldi
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Kaldi build & runtime requirements
RUN apt-get update && apt-get install -y --no-install-recommends \
        g++ make automake autoconf bzip2 unzip wget curl sox flac gawk \
        libtool git subversion zlib1g-dev gfortran ca-certificates patch \
        ffmpeg python3 python3-pip python3-dev \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/local/bin/python

# Build Kaldi (tools with MKL, then src)
RUN git clone --depth 1 https://github.com/kaldi-asr/kaldi.git /opt/kaldi \
    && cd /opt/kaldi/tools \
    && ./extras/install_mkl.sh \
    && make -j"$(nproc)" \
    && cd /opt/kaldi/src \
    && ./configure --shared --use-cuda=yes \
    # Maximum speed, no checking
    && sed -i 's/-O0 -DKALDI_PARANOID/-O3 -DNDEBUG/g' kaldi.mk \
    && make depend -j"$(nproc)" \
    && make -j"$(nproc)" \
    && find /opt/kaldi -type f -name '*.o' -delete

# Kaldi scripts pass --project_output, removed in OpenFST 1.8.x (tools above
# install 1.8.4). Repoint to --project_type. No-op if upstream fixes it.
RUN grep -rl 'project_output' /opt/kaldi/egs/wsj/s5/steps /opt/kaldi/egs/wsj/s5/utils \
    | xargs -r sed -i 's/--project_output=true/--project_type=output/g; s/--project_output=false/--project_type=input/g'

# SSL requirements (pinned lock for reproducible, README-matching results)
RUN git clone --depth 1 https://github.com/ialmajai/ssl-kaldi.git /opt/ssl-kaldi \
    && pip3 install --no-cache-dir -r /opt/ssl-kaldi/requirements.lock

ENV KALDI_ROOT=/opt/kaldi
WORKDIR /opt/ssl-kaldi
