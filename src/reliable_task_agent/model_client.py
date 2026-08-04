from __future__ import annotations

import os
import httpx

from dotenv import load_dotenv
from openai import OpenAI


def create_client() -> tuple[OpenAI, str]:
    """读取环境变量并创建模型客户端。"""
    load_dotenv()

    api_key = os.getenv("LLM_API_KEY")
    base_url = os.getenv("LLM_BASE_URL")
    model = os.getenv("LLM_MODEL")

    if not api_key:
        raise RuntimeError("缺少 LLM_API_KEY，请检查项目根目录中的 .env 文件。")

    if not model:
        raise RuntimeError("缺少 LLM_MODEL，请检查项目根目录中的 .env 文件。")

    http_client = httpx.Client(
        trust_env=False,
        timeout=httpx.Timeout(
            timeout=60.0,
            connect=20.0,
        ),
    )

    client = OpenAI(
        api_key=api_key,
        base_url=base_url or None,
        timeout=60.0,
        max_retries=2,
        http_client=http_client,
    )
    return client, model


def test_model_connection() -> str:
    """发送一个最小请求，测试模型是否可以正常调用。"""
    client, model = create_client()

    # response = client.chat.completions.create(
    #     model=model,
    #     messages=[
    #         {
    #             "role": "system",
    #             "content": "你是一个工程任务智能体，请严格遵循用户要求。",
    #         },
    #         {
    #             "role": "user",
    #             "content": "请只回复：模型连接成功",
    #         },
    #     ],
    # )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "你是一个工程任务智能体，请严格遵循用户要求。",
            },
            {
                "role": "user",
                "content": "请只回复：模型连接成功",
            },
        ],
        extra_body={
            "thinking": {
                "type": "disabled",
            }
        },
    )
    content = response.choices[0].message.content

    if not content:
        raise RuntimeError("模型返回成功，但回复内容为空。")

    return content.strip()