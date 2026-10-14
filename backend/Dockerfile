# Python 版本与本机实测环境一致（venv 是 3.13.2）。此前写的是 3.11-slim，
# 而 requirements 里的版本是按 3.13 解析与验证的 —— 两者不一致时，镜像里装到的
# 传递依赖可能与本机不同，等于「同一份代码、两套依赖」。
FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
# 依赖已全部钉死（见 requirements.txt 顶部说明），这里装出来应当与本机一致。
# torch 只被 sentence-transformers 间接拉入，未钉版本：它体积大且平台相关，
# 钉死反而不利于跨平台使用。
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]