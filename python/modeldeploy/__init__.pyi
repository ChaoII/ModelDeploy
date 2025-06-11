"""
Make programmer easier to deploy deeplearning model, save time to save the world!
"""
from __future__ import annotations
import typing
from . import audio
from . import vision
__all__ = ['Backend', 'BaseModel', 'CPU', 'Device', 'GPU', 'NONE', 'ORT', 'OrtBackendOption', 'RuntimeOption', 'audio', 'vision']
class Backend:
    """
    Members:
    
      NONE
    
      ORT
    """
    NONE: typing.ClassVar[Backend]  # value = <Backend.NONE: 1>
    ORT: typing.ClassVar[Backend]  # value = <Backend.ORT: 0>
    __members__: typing.ClassVar[dict[str, Backend]]  # value = {'NONE': <Backend.NONE: 1>, 'ORT': <Backend.ORT: 0>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class BaseModel:
    runtime_option: ...
    def __init__(self) -> None:
        """
        Default Constructor
        """
    def get_custom_meta_data(self) -> ...:
        ...
    def get_input_info(self, arg0: int) -> ...:
        ...
    def get_output_info(self, arg0: int) -> ...:
        ...
    def initialized(self) -> bool:
        ...
    def model_name(self) -> str:
        ...
    def num_inputs(self) -> int:
        ...
    def num_outputs(self) -> int:
        ...
class Device:
    """
    Members:
    
      CPU
    
      GPU
    """
    CPU: typing.ClassVar[Device]  # value = <Device.CPU: 0>
    GPU: typing.ClassVar[Device]  # value = <Device.GPU: 1>
    __members__: typing.ClassVar[dict[str, Device]]  # value = {'CPU': <Device.CPU: 0>, 'GPU': <Device.GPU: 1>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class OrtBackendOption:
    device: Device
    device_id: int
    enable_fp16: bool
    enable_trt: bool
    execution_mode: int
    external_stream: capsule
    graph_optimization_level: int
    inter_op_num_threads: int
    intra_op_num_threads: int
    model_buffer: str
    model_filepath: str
    model_from_memory: bool
    optimized_model_filepath: str
    trt_max_shape: str
    trt_min_shape: str
    trt_opt_shape: str
    def __init__(self) -> None:
        ...
    def set_cpu_thread_num(self, arg0: int) -> None:
        ...
class RuntimeOption:
    backend: Backend
    cpu_thread_num: int
    device: Device
    device_id: int
    enable_fp16: bool
    enable_trt: bool
    model_buffer: str
    model_file: str
    model_from_memory: bool
    ort_option: ...
    def __init__(self) -> None:
        ...
    def set_cpu_thread_num(self, arg0: int) -> None:
        ...
    def set_external_raw_stream(self, arg0: int) -> None:
        ...
    def set_external_stream(self, arg0: capsule) -> None:
        ...
    def set_model_path(self, arg0: str) -> None:
        ...
    def set_ort_graph_opt_level(self, arg0: int) -> None:
        ...
    def use_cpu(self) -> None:
        ...
    def use_gpu(self, arg0: int) -> None:
        ...
    def use_ort_backend(self) -> None:
        ...
CPU: Device  # value = <Device.CPU: 0>
GPU: Device  # value = <Device.GPU: 1>
NONE: Backend  # value = <Backend.NONE: 1>
ORT: Backend  # value = <Backend.ORT: 0>
