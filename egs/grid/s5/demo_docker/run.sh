#!/usr/bin/env bash
# Run the Kaldi lipreading demo web UI in an isolated container.
#
# Isolation: the container sees the ssl-kaldi repo, its conda env, the
# compiled Kaldi toolkit, and the host CUDA toolkit read-only, can write
# only to its private tmpfs /tmp plus one narrow exception (below) for the
# e2e model's decode_demo/ output, and runs as the invoking (non-root)
# user with all capabilities dropped.
# It lives on the internal-only 'kaldi-vsr-net' network: no internet
# egress, no LAN access, no published host port -- the only way in is a
# cloudflared container (run_tunnel.sh) on the same network, or the local
# quota-exempt access this script prints at the end.
#
# Distinct container/network names from icefall's own demo_docker/ (vsr-demo,
# vsr-net) since both can run on this box at once.
#
# Runs detached with a restart policy, so the demo survives crashes and
# reboots. Stop for good with:
#   docker rm -f kaldi-vsr-demo
set -euo pipefail
cd "$(dirname "$0")/.."  # egs/grid/s5
RECIPE_ROOT=/data/learn/ssl-kaldi

docker build -q -t kaldi-vsr-demo demo_docker

docker network inspect kaldi-vsr-net >/dev/null 2>&1 || \
    docker network create --internal kaldi-vsr-net
# av_hubert and grid-corpus are symlinks under the recipe root pointing
# OUTSIDE it (av_hubert -> /data/git/av_hubert, grid-corpus -> icefall's
# download/grid-corpus) -- a plain recipe-root bind mount doesn't follow
# them to a real target inside the container, so both need their own mount
# at the same absolute path. Found by testing: the dangling symlink caused
# ModuleNotFoundError('avhubert') deep inside model loading, not an
# obvious "file not found" at the actual point of failure -- checked every
# symlink under the recipe root for the same problem (`find . -type l` +
# compare each target against the mounted paths) rather than assuming
# these two were the only ones.
#
# /opt/intel is a different class of missing mount, not a repo symlink:
# Kaldi's binaries are dynamically linked against Intel MKL
# (libmkl_intel_lp64.so etc, RPATH'd to /opt/intel/mkl/lib/intel64 --
# see kaldi.mk's MKL_LDLIBS) as a host library dependency, independent of
# anything the demo itself references. Found the same way: reproduced
# `steps/compute_cmvn_stats.sh` directly inside the container on a small
# copied data dir once av_hubert/grid-corpus stopped being the blocker,
# and read the real linker error from the log
# (`error while loading shared libraries: libmkl_intel_lp64.so`) instead
# of trusting the generic "Error computing CMVN stats" the wrapper script
# prints. /opt/intel/mkl is itself a symlink into a versioned sibling
# directory, so the whole /opt/intel tree is mounted rather than chasing
# the exact resolved target.
#
# kaldi_decode_e2e.sh writes per-request decode output to
# exp/chain/<model>/decode_demo -- inside the read-only repo mount. Tried
# redirecting it to a scratch tmpfs path first (KALDI_DEMO_SCRATCH env var,
# still in kaldi_decode_e2e.sh/kaldi_calibrate_conf.sh/kaldi_decode.sh for
# anyone who wants it), but that broke a different Kaldi convention:
# steps/nnet3/decode.sh finds the model (final.mdl) by taking dirname() of
# the decode-dir argument, so decode-dir's parent MUST be the actual model
# directory -- moving decode_demo out from under exp/chain/<model> makes
# it look for the model in the scratch dir instead ("no such file
# .../final.mdl"). Simpler fix: leave the path exactly where bare metal
# puts it (KALDI_DEMO_SCRATCH unset), and shadow just that one nested
# directory with a writable tmpfs, layered inside the otherwise read-only
# repo mount -- Docker resolves mounts by specificity, so a --tmpfs at a
# path nested under a broader read-only -v wins for that subtree only.
E2E_MODEL_DIR="$RECIPE_ROOT/egs/grid/s5/exp/chain/e2e_tdnnf_lor10norelu_cmvn_aug2x"
docker rm -f kaldi-vsr-demo >/dev/null 2>&1 || true
docker run -d --restart unless-stopped --name kaldi-vsr-demo \
    --network kaldi-vsr-net \
    --gpus all \
    --user "$(id -u):$(id -g)" \
    --read-only --tmpfs "/tmp:size=4g,uid=$(id -u),gid=$(id -g)" \
    --tmpfs "$E2E_MODEL_DIR/decode_demo:size=512m,uid=$(id -u),gid=$(id -g)" \
    --cap-drop ALL --security-opt no-new-privileges \
    --memory 8g --pids-limit 512 --init \
    --log-opt max-size=20m --log-opt max-file=3 \
    -v /etc/localtime:/etc/localtime:ro \
    -v /data/miniconda3/envs/ssl-kaldi:/data/miniconda3/envs/ssl-kaldi:ro \
    -v "$RECIPE_ROOT:$RECIPE_ROOT:ro" \
    -v /data/git/kaldi:/data/git/kaldi:ro \
    -v /usr/local/cuda:/usr/local/cuda:ro \
    -v /data/git/av_hubert:/data/git/av_hubert:ro \
    -v /data/icefall/egs/grid/VSR/download/grid-corpus:/data/icefall/egs/grid/VSR/download/grid-corpus:ro \
    -v /opt/intel:/opt/intel:ro \
    kaldi-vsr-demo \
    python local/demo/demo.py --ui \
        "$@"

sleep 2
IP=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' kaldi-vsr-demo)
echo "Local (quota-exempt) URL: http://${IP}:7860"
