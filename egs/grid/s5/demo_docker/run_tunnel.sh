#!/usr/bin/env bash
# Join this demo's isolated network to the *existing* cloudflared-vsr tunnel
# container (started by icefall's egs/grid/VSR/demo_docker/run_tunnel.sh),
# so it can reach kaldi-vsr-demo:7860 in addition to icefall's vsr-demo:7860.
#
# Does NOT start a second cloudflared container or touch Cloudflare's own
# config -- reuses the one tunnel already running under the account owner's
# credentials (~/.cloudflared/*.json). Only the account owner can do the
# Cloudflare-side setup below (registers a public hostname against their
# domain); this script just does the local Docker network join once that's
# done.
#
# One-time manual setup (NOT run by this script -- needs Cloudflare auth
# this script doesn't have):
#   1. Pick a public hostname, e.g. kaldi-demo.ibrahimalmajai.com.
#   2. Add an ingress rule to ~/.cloudflared/config.yml, ABOVE the existing
#      catch-all `service: http_status:404` line (ingress rules are matched
#      top-to-bottom, first match wins):
#        - hostname: kaldi-demo.ibrahimalmajai.com
#          service: http://kaldi-vsr-demo:7860
#   3. Register the DNS route for it:
#        cloudflared tunnel route dns vsr-demo kaldi-demo.ibrahimalmajai.com
#   4. Restart cloudflared-vsr so it picks up the new config:
#        docker restart cloudflared-vsr
set -euo pipefail

docker network inspect kaldi-vsr-net >/dev/null 2>&1 || \
    docker network create --internal kaldi-vsr-net
docker network connect kaldi-vsr-net cloudflared-vsr 2>/dev/null || \
    echo "cloudflared-vsr already on kaldi-vsr-net (or not running -- start icefall's run_tunnel.sh first)"

echo "Docker-side wiring done. Complete the one-time Cloudflare setup in this"
echo "script's header comment (steps 1-4) if you haven't already, then the"
echo "demo will be reachable at whatever hostname you chose."
