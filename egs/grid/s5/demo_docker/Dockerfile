# Minimal runtime for the Kaldi lipreading demo (local/demo/demo.py).
#
# Same philosophy as icefall's demo_docker/Dockerfile (egs/grid/VSR): the
# heavy environment is NOT baked in. run.sh bind-mounts, read-only:
#   - the ssl-kaldi repo (this recipe, including the vendored av_hubert/
#     checkout and dlib/mediapipe model files under it)
#   - the ssl-kaldi conda env (torch, fairseq, dlib, mediapipe, kaldiio, ...)
#   - the compiled Kaldi toolkit at KALDI_ROOT (this recipe shells out to
#     real Kaldi binaries -- compute-cmvn-stats, nnet3-latgen-faster,
#     lattice-to-ctm-conf, logistic-regression-eval, ... -- unlike icefall's
#     pure-PyTorch/k2 pipeline, so this image needs that extra mount)
#   - the host's CUDA toolkit (Kaldi's own binaries were compiled against
#     it directly via --cudatk-dir, separate from the pip-bundled CUDA
#     runtime libs torch/fairseq use)
# This image only provides the system pieces the demo shells out to
# (ffmpeg) or links against (OpenCV's/dlib's libGL/glib). GPU access comes
# from `docker run --gpus all` via the NVIDIA container toolkit.
#
# Base image version matters here in a way it doesn't for icefall's (which
# stays on ubuntu:22.04 fine): our Kaldi binaries are compiled fresh on
# the host (Ubuntu 25.10, glibc 2.42) and dynamically link against its
# libc -- running them against an older glibc fails at load time
# (`GLIBC_2.38' not found), not at Kaldi's own build time. Match the
# host's Ubuntu release rather than defaulting to 22.04.
FROM ubuntu:25.10

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

ENV CONDA_ENV=/data/miniconda3/envs/ssl-kaldi
ENV NVLIBS=$CONDA_ENV/lib/python3.8/site-packages/nvidia
ENV KALDI_ROOT=/data/git/kaldi
ENV PATH=$CONDA_ENV/bin:$PATH \
    LD_LIBRARY_PATH=$NVLIBS/cuda_nvrtc/lib:$NVLIBS/cublas/lib:$NVLIBS/cuda_runtime/lib:$NVLIBS/cudnn/lib:$NVLIBS/cufft/lib:$NVLIBS/curand/lib:$NVLIBS/cusolver/lib:$NVLIBS/cusparse/lib:$NVLIBS/nccl/lib:$NVLIBS/nvjitlink/lib:$NVLIBS/nvtx/lib:/usr/local/cuda/lib64 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_ANALYTICS_ENABLED=False \
    HOME=/tmp

WORKDIR /data/learn/ssl-kaldi/egs/grid/s5
