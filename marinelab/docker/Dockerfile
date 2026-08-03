# Build on the isaaclab base image.
# The isaaclab base image is built from the clean isaaclab fork's docker/ config
# (standard Isaac Lab source-build path).  Build it first, then reference it here.
ARG ISAACLAB_IMAGE=isaaclab:latest
FROM ${ISAACLAB_IMAGE}

# Note on the pip launcher: inside the isaaclab container the standard launchers are
#   ${ISAACLAB_PATH}/isaaclab.sh -p -m pip   (preferred, respects isaaclab.sh env setup)
#   /isaac-sim/python.sh -m pip               (direct path, used here for simplicity)
# If your isaaclab image uses a different layout, replace /isaac-sim/python.sh with
# the correct path (check ${ISAACLAB_PATH}/_isaac_sim/python.sh or the isaaclab.sh alias).

# Expose Isaac Sim's python.sh as PATH `python` so training can be launched from the
# overlay repos (e.g. `cd /workspace/constrained-albc && python scripts/train.py ...`).
# This is the official extension-template pattern (run from the external repo).
# A wrapper, NOT a symlink: python.sh resolves siblings (setup_python_env.sh, kit/) via
# `dirname BASH_SOURCE`, so it must run at its real path. exec preserves that; a symlink
# elsewhere makes BASH_SOURCE point at the link dir and the sibling lookup fails.
RUN printf '%s\n' '#!/bin/bash' 'exec /isaac-sim/python.sh "$@"' > /usr/local/bin/python \
    && chmod +x /usr/local/bin/python

COPY . /workspace/marinelab

# Pull Git LFS objects that were not included in the COPY context, then install.
# The "|| true" on git lfs pull is intentional: if LFS objects are already present
# in the build context (e.g. CI with lfs=true), the pull is a no-op and must not fail.
RUN cd /workspace/marinelab \
    && git lfs install \
    && git lfs pull || true \
    && /isaac-sim/python.sh -m pip install -e /workspace/marinelab

WORKDIR /workspace/marinelab
