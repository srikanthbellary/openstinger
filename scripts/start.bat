@echo off
REM OpenStinger startup script for Windows
REM Usage:
REM   scripts\start.bat           — start Tier 1 MCP server
REM   scripts\start.bat tier2     — start Tier 2 (memory + vault)
REM   scripts\start.bat tier3     — start Tier 3 (memory + vault + alignment)
REM   scripts\start.bat stop      — stop MCP server and Datasette
REM   scripts\start.bat status    — show what's running

SET TIER=%1
IF "%TIER%"=="" SET TIER=tier1

SET ROOT=%~dp0..
SET LOG_DIR=%ROOT%\.openstinger\logs

IF NOT EXIST "%LOG_DIR%" mkdir "%LOG_DIR%"

IF "%TIER%"=="stop" GOTO :STOP
IF "%TIER%"=="status" GOTO :STATUS

ECHO.
ECHO [1/4] Starting FalkorDB + Browser...
cd "%ROOT%"
docker compose up -d
IF ERRORLEVEL 1 (
    ECHO ERROR: docker compose failed. Is Docker Desktop running?
    EXIT /B 1
)
ECHO   OK  FalkorDB on localhost:6379
ECHO   OK  FalkorDB Browser at http://localhost:3000

ECHO.
ECHO [2/4] Waiting for FalkorDB to be healthy...
:WAIT_FALKORDB
docker exec openstinger_falkordb redis-cli ping 2>NUL | findstr /C:"PONG" >NUL
IF ERRORLEVEL 1 (
    timeout /t 1 /nobreak >NUL
    GOTO :WAIT_FALKORDB
)
ECHO   OK  FalkorDB healthy

ECHO.
ECHO [3/4] Starting MCP Server (%TIER%)...

REM Kill existing on port 8765
FOR /F "tokens=5" %%P IN ('netstat -ano ^| findstr ":8765 "') DO (
    taskkill /F /PID %%P >NUL 2>&1
)
timeout /t 2 /nobreak >NUL

IF "%TIER%"=="tier1"  SET MCP_CMD=python -m openstinger.mcp.server
IF "%TIER%"=="tier2"  SET MCP_CMD=python -m openstinger.scaffold.mcp.server
IF "%TIER%"=="tier3"  SET MCP_CMD=python -m openstinger.gradient.mcp.server

IF "%MCP_CMD%"=="" (
    ECHO ERROR: Unknown tier '%TIER%'. Use tier1, tier2, tier3, stop, or status.
    EXIT /B 1
)

CALL .venv\Scripts\activate.bat
START /B cmd /C "%MCP_CMD% > %LOG_DIR%\mcp-server.log 2>&1"
timeout /t 6 /nobreak >NUL

netstat -ano | findstr ":8765 " >NUL
IF ERRORLEVEL 1 (
    ECHO ERROR: MCP server failed to start. Check %LOG_DIR%\mcp-server.log
    EXIT /B 1
)
ECHO   OK  MCP Server at http://localhost:8765/sse

ECHO.
ECHO [4/4] Starting Datasette (SQLite browser)...
python -c "import datasette" >NUL 2>&1
IF ERRORLEVEL 1 (
    ECHO   SKIP datasette not installed. Run: pip install -e ".[tools]"
) ELSE (
    taskkill /F /IM datasette.exe >NUL 2>&1
    START /B cmd /C "datasette .openstinger\openstinger.db --port 8001 --host 0.0.0.0 > %LOG_DIR%\datasette.log 2>&1"
    timeout /t 2 /nobreak >NUL
    ECHO   OK  Datasette at http://localhost:8001
)

ECHO.
ECHO ===================================================
ECHO   OpenStinger is running
ECHO ===================================================
ECHO.
ECHO   MCP Server (SSE)    http://localhost:8765/sse
ECHO   FalkorDB Browser    http://localhost:3000
ECHO   Datasette (SQLite)  http://localhost:8001
ECHO.
ECHO   Connect graph browser to: localhost:6379 (no password)
ECHO   Logs: %LOG_DIR%\
ECHO ===================================================
ECHO.
GOTO :EOF

:STOP
ECHO Stopping OpenStinger...
FOR /F "tokens=5" %%P IN ('netstat -ano ^| findstr ":8765 "') DO taskkill /F /PID %%P >NUL 2>&1
FOR /F "tokens=5" %%P IN ('netstat -ano ^| findstr ":8001 "') DO taskkill /F /PID %%P >NUL 2>&1
ECHO   Done.
GOTO :EOF

:STATUS
ECHO OpenStinger service status:
netstat -ano | findstr ":6379 " >NUL && ECHO   RUNNING  FalkorDB       localhost:6379 || ECHO   STOPPED  FalkorDB       localhost:6379
netstat -ano | findstr ":3000 " >NUL && ECHO   RUNNING  Graph Browser  http://localhost:3000 || ECHO   STOPPED  Graph Browser  http://localhost:3000
netstat -ano | findstr ":8765 " >NUL && ECHO   RUNNING  MCP Server     http://localhost:8765 || ECHO   STOPPED  MCP Server     http://localhost:8765
netstat -ano | findstr ":8001 " >NUL && ECHO   RUNNING  Datasette      http://localhost:8001 || ECHO   STOPPED  Datasette      http://localhost:8001
GOTO :EOF
