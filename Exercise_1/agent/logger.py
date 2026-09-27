# -*- coding: utf-8 -*-
"""日志模块：标准库 logging 封装与 JSONL 追踪事件写入。"""

import json
import logging
from datetime import datetime
from pathlib import Path


def get_logger(name: str) -> logging.Logger:
    """获取带 StreamHandler 的 logger；避免重复添加 handler。"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def trace_event(event: dict, trace_file: str = "logs/trace.jsonl") -> None:
    """将追踪事件以 JSONL 追加写入文件；任何异常都吞掉，绝不影响主流程。"""
    try:
        record = dict(event)
        if "ts" not in record:
            record["ts"] = datetime.now().isoformat(timespec="milliseconds")
        path = Path(trace_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:  # 写日志失败只告警，不向上抛
        logging.getLogger(__name__).warning("trace_event 写入失败: %s", exc)
