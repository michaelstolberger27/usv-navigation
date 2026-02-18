#!/bin/bash
cd "$(dirname "$0")"

# Ensure output directory exists on host (for volume mount)
mkdir -p ../output

# Stop any existing containers that hold the required ports
docker compose down --remove-orphans 2>/dev/null
docker ps -q --filter "publish=5900" | xargs -r docker stop 2>/dev/null

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
    echo "    docker exec docker-sim-1 bash -c \"cd /app/commonocean-sim/src && python3 /app/usv-navigation/commonocean_integration/scripts/commonocean_collision_test.py\""
    echo ""
    echo "  Run CommonOcean scenario:"
    echo "    docker exec docker-sim-1 bash -c \"cd /app/commonocean-sim/src && python3 /app/usv-navigation/commonocean_integration/scripts/commonocean_scenario.py <scenario.xml>\""
    echo ""
    echo "  Attach to container:"
    echo "    docker exec -it docker-sim-1 bash"
    echo ""
    echo "  Output files: output/"
    echo ""
fi
