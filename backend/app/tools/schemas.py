from pydantic import BaseModel, Field


class ToolParameter(BaseModel):
    type: str
    description: str


class ToolSchema(BaseModel):
    name: str
    description: str
    parameters: dict[str, ToolParameter]


TOOL_SCHEMAS: dict[str, ToolSchema] = {
    "search_documents": ToolSchema(
        name="search_documents",
        description="在私有文档知识库中进行混合检索（语义+关键词），返回最相关的文档片段",
        parameters={
            "query": ToolParameter(type="string", description="搜索查询"),
            "top_k": ToolParameter(type="integer", description="返回结果数量，默认4"),
        },
    ),
    "search_web": ToolSchema(
        name="search_web",
        description="在互联网上搜索最新信息，用于时效性问题和实时数据",
        parameters={
            "query": ToolParameter(type="string", description="搜索查询"),
            "num": ToolParameter(type="integer", description="返回结果数量，默认5"),
        },
    ),
    "parse_document": ToolSchema(
        name="parse_document",
        description="解析上传的文档（PDF/TXT），提取文本内容并建立索引",
        parameters={
            "file_path": ToolParameter(type="string", description="上传文件的本地路径"),
        },
    ),
}
