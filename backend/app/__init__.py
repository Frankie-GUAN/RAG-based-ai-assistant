"""个人知识库助手 —— 后端应用包。

**这里必须先于一切 import pyarrow**，原因见下。

`app.rag.loader` 导入链会走到 sentence_transformers，而它内部 `import datasets`
→ `pyarrow.dataset` → 加载 pyarrow 的原生扩展。此时 torch（sentence_transformers
先加载它）已在进程里占掉了大量地址空间，pyarrow 的 DLL 映射就会失败，进程直接
崩在 `create_module` 阶段 —— 不是 Python 异常，try/except 接不住，`pytest` 只留下
一行 "Segmentation fault"（exit 139），`python -c "import app.main"` 则毫无输出。

实测（2026-10-14，本机 Python 3.13.2 / pyarrow 24.0.0 / torch 2.12.0+cpu）：

    python -c "import app.main"                      # 段错误，exit 139
    python -c "import pyarrow; import app.main"      # 正常，随后报业务层的 ImportError

因此把 pyarrow 的加载提到包初始化：无论入口是 uvicorn `app.main:app`、pytest
（先导入 conftest → app.config → app 包）还是脚本，只要碰得到这个包，pyarrow 就
一定在 torch 之前完成加载。

这是**环境层的权宜之计**，不是修复。根因是 venv 与文档声明不一致：
CLAUDE.md 记的是 Python 3.11，实际 venv 是 3.13.2，且所有依赖都只有 `>=` 下限、
没有锁文件。根治办法是把版本钉死（或按文档重建 3.11 venv），见
`docs/plans/2026-10-14-personal-kb-revival.md` 的 backlog。
届时本文件可以整段删除。
"""

# ruff: noqa: E402  —— 必须在任何业务 import 之前执行
import pyarrow as _pyarrow  # noqa: F401