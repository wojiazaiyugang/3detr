from typing import Dict, Any
from dataclasses import dataclass, field
from algorithm_assistant import ToothDetectResult3D

@dataclass()
class ToothDetect(ToothDetectResult3D):
    """
    拓展ToothDetectResult3D
    """
    data: Dict[str, Any] = field(default_factory=lambda: {})  # 额外数据，可以是任何类型
