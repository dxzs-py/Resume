"""pytest 根配置：把项目根注入 sys.path，保证 `import rate_limiter` 在任意工作目录可用。"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
