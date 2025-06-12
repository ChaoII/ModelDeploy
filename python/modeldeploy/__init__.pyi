"""
Make programmer easier to deploy deeplearning model, save time to save the world!
"""
from __future__ import annotations
import typing
from . import audio
from . import vision
__all__ = ['Backend', 'BaseModel', 'CPU', 'DataType', 'Device', 'FP32', 'FP64', 'GPU', 'INT32', 'INT64', 'INT8', 'NONE', 'ORT', 'OrtBackendOption', 'RuntimeOption', 'TensorInfo', 'UINT8', 'UNKNOW', 'audio', 'vision']
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
    def __init__(self) -> None:
        """
        Default Constructor
        """
    def get_custom_meta_data(self) -> dict:
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
class DataType:
    """
    Members:
    
      FP32
    
      FP64
    
      INT32
    
      INT64
    
      INT8
    
      UINT8
    
      UNKNOW
    """
    FP32: typing.ClassVar[DataType]  # value = <DataType.FP32: 0>
    FP64: typing.ClassVar[DataType]  # value = <DataType.FP64: 1>
    INT32: typing.ClassVar[DataType]  # value = <DataType.INT32: 2>
    INT64: typing.ClassVar[DataType]  # value = <DataType.INT64: 3>
    INT8: typing.ClassVar[DataType]  # value = <DataType.INT8: 5>
    UINT8: typing.ClassVar[DataType]  # value = <DataType.UINT8: 4>
    UNKNOW: typing.ClassVar[DataType]  # value = <DataType.UNKNOW: 6>
    __members__: typing.ClassVar[dict[str, DataType]]  # value = {'FP32': <DataType.FP32: 0>, 'FP64': <DataType.FP64: 1>, 'INT32': <DataType.INT32: 2>, 'INT64': <DataType.INT64: 3>, 'INT8': <DataType.INT8: 5>, 'UINT8': <DataType.UINT8: 4>, 'UNKNOW': <DataType.UNKNOW: 6>}
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
    def __init__(self) -> None:
        ...
    def set_cpu_thread_num(self, arg0: int) -> None:
        ...
    def set_external_raw_stream(self, arg0: int) -> None:
        ...
    def set_external_stream(self, stream: capsule) -> None:
        """
        A pointer to an external stream
        """
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
class TensorInfo:
    dtype: DataType
    name: str
    shape: list[int]
    def __init__(self) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
CPU: Device  # value = <Device.CPU: 0>
FP32: DataType  # value = <DataType.FP32: 0>
FP64: DataType  # value = <DataType.FP64: 1>
GPU: Device  # value = <Device.GPU: 1>
INT32: DataType  # value = <DataType.INT32: 2>
INT64: DataType  # value = <DataType.INT64: 3>
INT8: DataType  # value = <DataType.INT8: 5>
NONE: Backend  # value = <Backend.NONE: 1>
ORT: Backend  # value = <Backend.ORT: 0>
UINT8: DataType  # value = <DataType.UINT8: 4>
UNKNOW: DataType  # value = <DataType.UNKNOW: 6>
