#!/bin/sh
# Vendor the TrackNetV3 repo (plan stage S4) into pipeline/vendor/TrackNetV3.
#
# WHY vendored, not a dependency: research code with no package layout; we couple only to
# its predict.py CLI + ball CSV output (see synchro_pipeline/perception/tracknet_v3.py).
# The vendor/ directory is git-ignored — this script is the reproducible setup step.
set -eu

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PIPELINE_DIR=$(dirname "$SCRIPT_DIR")
DEST="$PIPELINE_DIR/vendor/TrackNetV3"
REPO_URL="https://github.com/qaz812345/TrackNetV3"

if [ -e "$DEST" ]; then
    echo "TrackNetV3 already vendored at $DEST — skipping clone."
else
    mkdir -p "$PIPELINE_DIR/vendor"
    git clone --depth 1 "$REPO_URL" "$DEST"
fi

cat <<EOF

TrackNetV3 vendored at: $DEST

License: MIT (verify the LICENSE file in the clone). Per docs/plan.md, weight
*provenance* (training-data terms) must still be confirmed with the authors before
any commercial launch — vendoring the code is not that sign-off.

Pretrained weights are NOT downloaded by this script. Get them per the vendor
README ($DEST/README.md) — the repo links its checkpoints (TrackNet_best.pt and
InpaintNet_best.pt). Then point the pipeline at them:

    export SYNCHRO_TRACKNET_WEIGHTS=/path/to/TrackNet_best.pt
    # optional; defaults to a sibling InpaintNet_best.pt if present:
    export SYNCHRO_TRACKNET_INPAINT_WEIGHTS=/path/to/InpaintNet_best.pt

Torch is required at inference time: uv sync --all-packages --extra ml
EOF
