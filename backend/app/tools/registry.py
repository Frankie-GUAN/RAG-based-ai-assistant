from typing import Any, Callable

from app.tools.schemas import TOOL_SCHEMAS


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Callable] = {}

    def register(self, name: str, fn: Callable) -> None:
        self._tools[name] = fn

    def get(self, name: str) -> Callable | None:
        return self._tools.get(name)

    def list_tools(self) -> list[dict[str, Any]]:
        return [
            {"name": name, "description": schema.description, "parameters": schema.parameters}
            for name, schema in TOOL_SCHEMAS.items()
        ]

    def execute(self, name: str, **kwargs) -> str:
        fn = self.get(name)
        if fn is None:
            return f"Tool '{name}' not found"
        return fn(**kwargs)


tool_registry = ToolRegistry()
