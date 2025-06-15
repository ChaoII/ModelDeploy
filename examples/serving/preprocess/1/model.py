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
        responses = []
        try:
            im_list = []
            for request in requests:
                arr = pb_utils.get_input_tensor_by_name(request, self.input_names[0]).as_numpy()
                if arr.shape[0] != 1:
                    raise ValueError("Only per-request batch size of 1 is supported in dynamic batching mode.")
                im_list.append(arr[0])  # shape: [H, W, 3]
            batch_input = np.stack(im_list, axis=0)  # shape: [N, H, W, 3]
            print("batch_input shape: ", batch_input.shape)
            outputs, im_infos = self.preprocessor_.run(batch_input)
            for i in range(len(requests)):
                out_tensor = outputs[0].to_numpy()[i]
                im_info_bytes = pickle.dumps(im_infos[i])
                output_tensor_0 = pb_utils.Tensor(self.output_names[0], out_tensor)
                output_tensor_1 = pb_utils.Tensor(self.output_names[1], np.array(im_info_bytes, dtype=np.object_))
                responses.append(pb_utils.InferenceResponse(output_tensors=[output_tensor_0, output_tensor_1]))

        except Exception as e:
            print("Error:", e)
            for _ in requests:
                responses.append(pb_utils.InferenceResponse(
                    output_tensors=[], error=pb_utils.TritonError(str(e))))
        return responses

    def finalize(self):
        print('[preprocess] finalize called')
