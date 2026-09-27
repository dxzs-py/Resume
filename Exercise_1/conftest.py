# -*- coding: utf-8 -*-
"""pytest 配置：把项目根目录加入 sys.path，保证任意目录下都能导入 agent 包。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
