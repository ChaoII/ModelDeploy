"""
Audio module of Modeldeploy.
"""
from __future__ import annotations
import modeldeploy.modeldeploy
__all__ = ['Kokoro']
class Kokoro(modeldeploy.modeldeploy.BaseModel):
    sample_rate: int
    def __init__(self, model_file_path: str, token_path_str: str, lexicons: list[str], voices_bin: str, jieba_dir: str, text_normalization_dir: str, option: modeldeploy.modeldeploy.RuntimeOption) -> None:
        ...
    def predict(self, text: str, voice: str, speed: float) -> list[float]:
        ...
