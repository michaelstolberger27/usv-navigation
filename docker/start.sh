#!/bin/bash
cd "$(dirname "$0")"

# Ensure output directory exists on host (for volume mount)
mkdir -p ../output

if [ "$1" = "-it" ] || [ "$1" = "--interactive" ]; then
    # Interactive: attach directly to the container
    docker compose run --rm --service-ports sim
else
    # Detached: run in background
    docker compose up -d
    echo ""
    echo "=== USV Navigation Simulator ==="
    echo ""
    echo "  VNC:   vnc://localhost:5900"
    echo "  Web:   http://localhost:6080/vnc.html"
    echo ""
    echo "  Run head-on collision test:"
    echo "    docker exec docker-sim-1 bash -c \"cd /app/commonocean-sim/src && python3 /app/usv-navigation/examples/commonocean_collision_test.py\""
    echo ""
    echo "  Run CommonOcean scenario:"
    echo "    docker exec docker-sim-1 bash -c \"cd /app/commonocean-sim/src && python3 /app/usv-navigation/examples/commonocean_scenario.py <scenario.xml>\""
    echo ""
    echo "  Attach to container:"
    echo "    docker exec -it docker-sim-1 bash"
    echo ""
    echo "  Output files: output/"
    echo ""
fi
