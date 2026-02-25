@echo off
cd /d C:\Users\bells\CLAUDE_CODE\openstinger
echo Starting OpenStinger MCP server...
.venv\Scripts\python.exe -m openstinger.gradient.mcp.server >> .openstinger\openstinger.log 2>&1
echo Server exited.
