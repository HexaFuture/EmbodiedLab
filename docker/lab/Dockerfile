# EmbodiedLab lab image: Isaac Sim 6.0.1 with the EmbodiedLab environment runtime and the benchmark benches.
# Build from the repository root with ./embodiedlab build.
FROM nvcr.io/nvidia/isaac-sim:6.0.1
USER root
ARG PIP_INDEX_URL=
COPY docker/lab/requirements.txt /tmp/embodiedlab-requirements.txt
RUN /isaac-sim/python.sh -m pip install --no-cache-dir --no-deps --target /opt/embodiedlab-python \
        ${PIP_INDEX_URL:+--index-url "$PIP_INDEX_URL"} \
        -r /tmp/embodiedlab-requirements.txt \
 && rm /tmp/embodiedlab-requirements.txt
COPY assets/workspace/ /workspace/
COPY assets/scenes/ /isaac-sim/documents/
COPY sdk/src/embodiedlab_sdk/ /opt/embodiedlab-sdk/embodiedlab_sdk/
COPY benchmark/core/ /opt/embodiedlab/bench/
ENV PYTHONPATH=/opt/embodiedlab-python:/opt/embodiedlab-sdk:/opt/embodiedlab/bench \
    PYTHONUNBUFFERED=1 \
    EMBODIEDLAB_ENV_BOOTSTRAP=embodiedlab_sdk.robot.embodiedlab.bootstrap:bootstrap \
    EMBODIEDLAB_BENCH_PRIVATE_ROOT=/var/lib/embodiedlab/private
RUN mkdir -p /var/lib/embodiedlab/private && chmod 700 /var/lib/embodiedlab/private
# The Worker probes readiness with `docker exec embodiedlab-env-client ready`.
RUN printf '#!/bin/sh\nexec /isaac-sim/kit/python/bin/python3 /opt/embodiedlab-sdk/embodiedlab_sdk/env_client.py "$@"\n' \
        > /usr/local/bin/embodiedlab-env-client \
 && chmod 755 /usr/local/bin/embodiedlab-env-client
ARG EMBODIEDLAB_VERSION=0.1.0
LABEL io.embodiedlab.image=lab \
      io.embodiedlab.interface=public-device-v1 \
      io.embodiedlab.version=${EMBODIEDLAB_VERSION}
HEALTHCHECK --interval=30s --timeout=10s --start-period=10m \
  CMD /isaac-sim/python.sh -c 'import os,urllib.request; urllib.request.urlopen(os.environ.get("EMBODIEDLAB_ENVD_URL","http://127.0.0.1:8765")+"/ready",timeout=5).read()' || exit 1
WORKDIR /opt/embodiedlab/bench
ENTRYPOINT ["/isaac-sim/python.sh", "-m", "embodiedlab_sdk.envd"]
CMD ["--host", "127.0.0.1", "--port", "8765"]
