FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# ---------- System dependencies (same as commonocean-sim) ----------
RUN apt-get update && apt-get install -y \
    software-properties-common \
    git \
    wget \
    build-essential \
    gfortran \
    gcc \
    g++ \
    libopenblas-dev \
    liblapack-dev \
    libblas-dev \
    libatlas-base-dev \
    pkg-config \
    cmake \
    libgmp-dev \
    libmpfr-dev \
    libmpc-dev \
    libboost-all-dev \
    libeigen3-dev \
    libgl1-mesa-glx \
    libglu1-mesa \
    libx11-6 \
    libxext6 \
    libxrender1 \
    xvfb \
    x11vnc \
    fluxbox \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
RUN python3 -m pip install --upgrade pip setuptools wheel
RUN python3 -m pip install "Cython<3.0" numpy

# ---------- Clone commonocean-sim ----------
WORKDIR /app
RUN git clone https://github.com/CommonOcean/commonocean-sim.git /app/commonocean-sim

WORKDIR /app/commonocean-sim
RUN git submodule update --init --recursive

# ---------- Build commonocean-sim dependencies ----------
RUN pip install git+https://github.com/drufat/triangle.git@master || true

RUN sed -i "/        'commonroad-drivability-checker==/d" install/commonocean-drivability-checker/setup.py && \
    sed -i "/        'numpy~/d" install/commonocean-drivability-checker/setup.py && \
    sed -i "/        'scipy<=/d" install/commonocean-drivability-checker/setup.py && \
    sed -i "/        'triangle'/d" install/commonocean-drivability-checker/setup.py && \
    pip install --only-binary scipy,numpy scipy numpy simple-pid

RUN bash install.sh

# Fix shapely 2.0 compatibility
RUN sed -i 's/from shapely.ops import unary_union, cascaded_union/from shapely.ops import unary_union/' \
    install/commonocean_rules/rules/predicates/predicate_collection_ship.py && \
    sed -i 's/cascaded_union/unary_union/g' \
    install/commonocean_rules/rules/predicates/predicate_collection_ship.py

# ---------- commonroad-drivability-checker ----------
# FCL (a transitive C++ dep) hardcodes x86 SSE flags that don't exist on ARM.
# On aarch64 we strip them with a thin compiler wrapper.
ARG TARGETARCH
RUN if [ "$(uname -m)" = "aarch64" ]; then \
      printf '#!/bin/bash\nargs=()\nfor arg in "$@"; do\n  case "$arg" in\n    -mfpmath=sse|-msse|-msse2|-msse3|-mssse3) ;;\n    *) args+=("$arg") ;;\n  esac\ndone\nexec /usr/bin/g++ "${args[@]}"\n' \
        > /usr/local/bin/g++-nosse && chmod +x /usr/local/bin/g++-nosse && \
      CXX=/usr/local/bin/g++-nosse pip install commonroad-drivability-checker==2023.1 && \
      rm /usr/local/bin/g++-nosse; \
    else \
      pip install commonroad-drivability-checker==2023.1; \
    fi

# ---------- GUROBI (required by commonocean-sim MPC controller) ----------
RUN if [ "$TARGETARCH" = "arm64" ]; then \
      wget https://packages.gurobi.com/11.0/gurobi11.0.3_armlinux64.tar.gz && \
      tar xvfz gurobi11.0.3_armlinux64.tar.gz && \
      mv gurobi1103 /opt/gurobi && \
      rm gurobi11.0.3_armlinux64.tar.gz; \
    else \
      wget https://packages.gurobi.com/11.0/gurobi11.0.3_linux64.tar.gz && \
      tar xvfz gurobi11.0.3_linux64.tar.gz && \
      mv gurobi1103 /opt/gurobi && \
      rm gurobi11.0.3_linux64.tar.gz; \
    fi

ENV GUROBI_HOME=/opt/gurobi/linux64
ENV PATH="${PATH}:${GUROBI_HOME}/bin"
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${GUROBI_HOME}/lib"
RUN python3 -m pip install gurobipy

# ---------- VNC / noVNC for visualisation ----------
RUN apt-get update && apt-get install -y \
    novnc \
    websockify \
    xterm \
    python3-tk \
    tk-dev \
    xdotool \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN echo '#!/bin/bash\n\
Xvfb :99 -screen 0 1280x1024x24 &\n\
export DISPLAY=:99\n\
fluxbox &\n\
x11vnc -display :99 -forever -rfbport 5900 -shared -nopw &\n\
websockify --web=/usr/share/novnc 6080 localhost:5900 &\n\
echo "VNC server started on port 5900"\n\
echo "noVNC web interface available at http://localhost:6080/vnc.html"\n\
tail -f /dev/null' > /usr/local/bin/start-vnc && \
    chmod +x /usr/local/bin/start-vnc

ENV DISPLAY=:99
EXPOSE 5900 6080

# ---------- Install local dependencies (not yet on PyPI) ----------
# Pulled in via additional_contexts in docker-compose.yml
COPY --from=hybrid-automaton . /tmp/hybrid-automaton
RUN pip install /tmp/hybrid-automaton && rm -rf /tmp/hybrid-automaton

COPY --from=unsafe-set . /tmp/unsafe-set
RUN pip install /tmp/unsafe-set && rm -rf /tmp/unsafe-set

# ---------- Install usv-navigation ----------
COPY . /app/usv-navigation
RUN pip install -e /app/usv-navigation[viz]

# commonocean-sim is not pip-installable; keep it on PYTHONPATH
# Also add usv-navigation root so commonocean_integration package is importable
ENV PYTHONPATH="/app/commonocean-sim/src:/app/usv-navigation:${PYTHONPATH}"

WORKDIR /app/usv-navigation

# Entrypoint that starts VNC then runs the command
RUN echo '#!/bin/bash\n\
# Start VNC services in background\n\
Xvfb :99 -screen 0 1280x1024x24 > /dev/null 2>&1 &\n\
sleep 1\n\
x11vnc -display :99 -forever -rfbport 5900 -shared -nopw > /dev/null 2>&1 &\n\
websockify --web=/usr/share/novnc 6080 localhost:5900 > /dev/null 2>&1 &\n\
echo ""\n\
echo "=== USV Navigation Simulator ==="\n\
echo ""\n\
echo "  VNC:   vnc://localhost:5900"\n\
echo "  Web:   http://localhost:6080/vnc.html"\n\
echo ""\n\
echo "  Run head-on collision test:"\n\
echo "    cd /app/commonocean-sim/src && python3 /app/usv-navigation/commonocean_integration/scripts/commonocean_collision_test.py"\n\
echo ""\n\
echo "  Run CommonOcean scenario:"\n\
echo "    cd /app/commonocean-sim/src && python3 /app/usv-navigation/commonocean_integration/scripts/commonocean_scenario.py <scenario.xml>"\n\
echo ""\n\
echo "  Output files: /app/usv-navigation/output/"\n\
echo ""\n\
# Run the passed command or bash\n\
exec "$@"' > /entrypoint.sh && chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]
