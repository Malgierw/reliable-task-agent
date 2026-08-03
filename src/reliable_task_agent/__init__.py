from reliable_task_agent.model_client import test_model_connection


def main() -> None:
    """项目命令行入口。"""
    try:
        result = test_model_connection()
        print(result)
    except Exception as exc:
        print(f"模型连接失败：{type(exc).__name__}: {exc}")
        raise SystemExit(1) from exc