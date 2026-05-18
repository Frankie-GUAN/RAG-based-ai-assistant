from fastapi import APIRouter
from pydantic import BaseModel

from app.tools.registry import tool_registry

router = APIRouter(prefix="/api/tools", tags=["tools"])


class ToolExecuteRequest(BaseModel):
    name: str
    params: dict = {}


@router.get("/")
async def list_tools():
    return tool_registry.list_tools()


@router.post("/execute")
async def execute_tool(request: ToolExecuteRequest):
    result = tool_registry.execute(request.name, **request.params)
    return {"result": result}
