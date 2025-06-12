"""
Audio module of Modeldeploy.
"""
from __future__ import annotations
import modeldeploy
__all__ = ['Kokoro']
class Kokoro(modeldeploy.BaseModel):
    sample_rate: int
    def __init__(self, arg0: str, arg1: str, arg2: list[str], arg3: str, arg4: str, arg5: str, arg6: modeldeploy.RuntimeOption) -> None:
        ...
    def predict(self, arg0: str, arg1: str, arg2: float) -> list[float]:
        ...
