#!/usr/bin/env bash
# OpenStinger startup script
# Starts: FalkorDB + FalkorDB Browser + MCP server + Datasette (SQLite browser)
#
# Usage:
#   ./scripts/start.sh           — start Tier 1 MCP server (default)
#   ./scripts/start.sh tier2     — start Tier 2 (memory + vault)
#   ./scripts/start.sh tier3     — start Tier 3 (memory + vault + alignment)
#   ./scripts/start.sh stop      — stop MCP server and Datasette
#   ./scripts/start.sh status    — show what's running

set -e

TIER="${1:-tier1}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$ROOT_DIR/.openstinger/logs"
mkdir -p "$LOG_DIR"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# ─── helper functions ────────────────────────────────────────────────────────

check_port() {
    netstat -ano 2>/dev/null | grep -q ":$1 " && return 0 || return 1
}

print_urls() {
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}  OpenStinger is running${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo -e "  ${GREEN}●${NC} MCP Server (SSE)       http://localhost:8765/sse"
    echo -e "  ${GREEN}●${NC} FalkorDB Browser       http://localhost:3000"
    echo -e "  ${GREEN}●${NC} Datasette (SQLite)     http://localhost:8001"
    echo ""
    echo -e "  Connect FalkorDB Browser to: ${YELLOW}localhost:6379${NC} (no password)"
    echo -e "  Logs: ${YELLOW}$LOG_DIR/${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
}

# ─── stop ────────────────────────────────────────────────────────────────────

if [ "$TIER" = "stop" ]; then
    echo "Stopping OpenStinger processes..."
    pkill -f "openstinger.mcp.server" 2>/dev/null && echo "  ✓ MCP server stopped" || echo "  - MCP server not running"
    pkill -f "openstinger.scaffold.mcp.server" 2>/dev/null && echo "  ✓ Scaffold server stopped" || true
    pkill -f "openstinger.gradient.mcp.server" 2>/dev/null && echo "  ✓ Gradient server stopped" || true
    pkill -f "datasette" 2>/dev/null && echo "  ✓ Datasette stopped" || echo "  - Datasette not running"
    exit 0
fi

# ─── status ──────────────────────────────────────────────────────────────────

if [ "$TIER" = "status" ]; then
    echo "OpenStinger service status:"
    check_port 6379 && echo -e "  ${GREEN}●${NC} FalkorDB       localhost:6379   RUNNING" \
                    || echo -e "  ${RED}●${NC} FalkorDB       localhost:6379   STOPPED"
    check_port 3000 && echo -e "  ${GREEN}●${NC} Graph Browser  http://localhost:3000  RUNNING" \
                    || echo -e "  ${RED}●${NC} Graph Browser  http://localhost:3000  STOPPED"
    check_port 8765 && echo -e "  ${GREEN}●${NC} MCP Server     http://localhost:8765  RUNNING" \
                    || echo -e "  ${RED}●${NC} MCP Server     http://localhost:8765  STOPPED"
    check_port 8001 && echo -e "  ${GREEN}●${NC} Datasette      http://localhost:8001  RUNNING" \
                    || echo -e "  ${RED}●${NC} Datasette      http://localhost:8001  STOPPED"
    exit 0
fi

# ─── activate venv ───────────────────────────────────────────────────────────

cd "$ROOT_DIR"

if [ -f ".venv/Scripts/activate" ]; then
    source .venv/Scripts/activate   # Windows
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate       # Linux/Mac
else
    echo -e "${RED}ERROR: .venv not found. Run: python -m venv .venv && pip install -e '.[dev,tools]'${NC}"
    exit 1
fi

# ─── Step 1: FalkorDB + Browser ───────────────────────────────────────────────

echo -e "${BLUE}[1/4] Starting FalkorDB + Browser...${NC}"
docker compose up -d
echo -e "  ${GREEN}✓${NC} FalkorDB running on localhost:6379"
echo -e "  ${GREEN}✓${NC} FalkorDB Browser at http://localhost:3000"
echo -e "       (Connect to: localhost:6379, no password)"

# ─── Step 2: Wait for FalkorDB healthy ────────────────────────────────────────

echo -e "${BLUE}[2/4] Waiting for FalkorDB to be healthy...${NC}"
for i in $(seq 1 10); do
    if docker exec openstinger_falkordb redis-cli ping 2>/dev/null | grep -q PONG; then
        echo -e "  ${GREEN}✓${NC} FalkorDB healthy"
        break
    fi
    if [ $i -eq 10 ]; then
        echo -e "  ${RED}✗${NC} FalkorDB did not become healthy in time"
        exit 1
    fi
    sleep 1
done

# ─── Step 3: MCP Server ───────────────────────────────────────────────────────

case "$TIER" in
    tier1)   MCP_CMD="python -m openstinger.mcp.server";           TIER_LABEL="Tier 1 (Memory)" ;;
    tier2)   MCP_CMD="python -m openstinger.scaffold.mcp.server";  TIER_LABEL="Tier 2 (Memory + Vault)" ;;
    tier3)   MCP_CMD="python -m openstinger.gradient.mcp.server";  TIER_LABEL="Tier 3 (Memory + Vault + Gradient)" ;;
    *)       echo -e "${RED}Unknown tier: $TIER. Use tier1, tier2, tier3, stop, or status${NC}"; exit 1 ;;
esac

echo -e "${BLUE}[3/4] Starting MCP Server ($TIER_LABEL)...${NC}"

# Kill any existing MCP server on port 8765
if check_port 8765; then
    echo -e "  ${YELLOW}⚠${NC}  Port 8765 in use — killing existing process"
    # Windows
    netstat -ano 2>/dev/null | grep ":8765 " | awk '{print $5}' | xargs -r taskkill //F //PID 2>/dev/null || true
    # Linux/Mac
    lsof -ti:8765 2>/dev/null | xargs -r kill -9 2>/dev/null || true
    sleep 2
fi

nohup $MCP_CMD > "$LOG_DIR/mcp-server.log" 2>&1 &
MCP_PID=$!
sleep 5

if check_port 8765; then
    echo -e "  ${GREEN}✓${NC} MCP Server running at http://localhost:8765/sse (PID $MCP_PID)"
else
    echo -e "  ${RED}✗${NC} MCP Server failed to start. Check: $LOG_DIR/mcp-server.log"
    exit 1
fi

# ─── Step 4: Datasette ────────────────────────────────────────────────────────

echo -e "${BLUE}[4/4] Starting Datasette (SQLite browser)...${NC}"

if ! python -c "import datasette" 2>/dev/null; then
    echo -e "  ${YELLOW}⚠${NC}  datasette not installed — run: pip install -e '.[tools]'"
    echo -e "      Skipping Datasette."
else
    pkill -f "datasette" 2>/dev/null || true
    sleep 1
    nohup datasette .openstinger/openstinger.db \
        --port 8001 --host 0.0.0.0 \
        > "$LOG_DIR/datasette.log" 2>&1 &
    sleep 2
    if check_port 8001; then
        echo -e "  ${GREEN}✓${NC} Datasette running at http://localhost:8001"
    else
        echo -e "  ${YELLOW}⚠${NC}  Datasette failed to start — check $LOG_DIR/datasette.log"
    fi
fi

print_urls
