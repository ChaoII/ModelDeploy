import json
import numpy as np
import time
import traceback
import sys
import pickle
import modeldeploy as md
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        try:
            print("[preprocess] initialize start")
            self.model_config = json.loads(args['model_config'])
            print("model_config:", self.model_config)

            self.input_names = []
            for input_config in self.model_config["input"]:
                self.input_names.append(input_config["name"])
            print("input names:", self.input_names)

            self.output_names = []
            self.output_dtype = []
            for output_config in self.model_config["output"]:
                self.output_names.append(output_config["name"])
                dtype = pb_utils.triton_string_to_numpy(output_config["data_type"])
                self.output_dtype.append(dtype)
            print("output names:", self.output_names)

            self.preprocessor_ = md.vision.UltralyticsPreprocessor()
            print("[preprocess] initialize done")
        except Exception as e:
            print("[preprocess] initialize failed:", e)
            traceback.print_exc(file=sys.stdout)
            raise

    def execute(self, requests):
        print("[preprocess] execute called, batch size =", len(requests))
        responses = []
        try:
            for request in requests:
                data = pb_utils.get_input_tensor_by_name(request, self.input_names[0])
                data = data.as_numpy()
                print("input shape:", data.shape, data.dtype)
                # 调用你的 C++ 绑定预处理
                outputs, im_infos = self.preprocessor_.run(data)
                im_infos_dict = [pickle.dumps(i) for i in im_infos]
                out_tensor = outputs[0].to_numpy()
                output_tensor_0 = pb_utils.Tensor(self.output_names[0], out_tensor)
                output_tensor_1 = pb_utils.Tensor(self.output_names[1], np.array(im_infos_dict, dtype=np.object_))
                inference_response = pb_utils.InferenceResponse(
                    output_tensors=[output_tensor_0, output_tensor_1])
                responses.append(inference_response)
        except Exception as e:
            print("[preprocess] ERROR during execute:", e)
            traceback.print_exc(file=sys.stdout)
            for _ in requests:
                responses.append(pb_utils.InferenceResponse(
                    output_tensors=[], error=pb_utils.TritonError(str(e))))
        return responses

    def finalize(self):
        print('[preprocess] finalize called')
