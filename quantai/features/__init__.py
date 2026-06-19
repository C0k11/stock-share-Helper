"""quantai.features —— 特征层（技术因子 + 市场状态检测）。

全部为**因果**计算，无 lookahead（截断不变性测试保证）。
新闻类特征不在本模块（随后续 news 模块迁入）。

用法：
    from quantai.features import compute_technical_features, RegimeDetector
"""

from .regime import MarketRegime, RegimeDetector, detect_regime
from .technical import TechnicalFeatures, compute_technical_features

__all__ = [
    "TechnicalFeatures",
    "compute_technical_features",
    "RegimeDetector",
    "MarketRegime",
    "detect_regime",
]
