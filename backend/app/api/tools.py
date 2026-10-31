from fastapi import APIRouter

from app.mcp.server import mcp_server

router = APIRouter(prefix="/api/tools", tags=["tools"])


@router.get("")
async def list_tools():
    """列出 Agent 可用的 MCP 工具。"""
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema,
        }
        for tool in await mcp_server.list_tools()
    ]
