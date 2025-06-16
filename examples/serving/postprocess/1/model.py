import json
import numpy as np
import time
import modeldeploy as md
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        # You must parse model_config. JSON string is not parsed here
        self.model_config = json.loads(args['model_config'])
        print("model_config:", self.model_config)
        self.input_names = []
        for input_config in self.model_config["input"]:
            self.input_names.append(input_config["name"])
        print("postprocess input names:", self.input_names)
        self.output_names = []
        self.output_dtype = []
        for output_config in self.model_config["output"]:
            self.output_names.append(output_config["name"])
            dtype = pb_utils.triton_string_to_numpy(output_config["data_type"])
            self.output_dtype.append(dtype)
        print("postprocess output names:", self.output_names)
        self.postprocessor_ = md.vision.UltralyticsPostprocessor()

    def execute(self, requests):
        infer_outputs = []
        im_infos = []
        for request in requests:
            #[1,84,6400]
            infer_output = pb_utils.get_input_tensor_by_name(request, self.input_names[0]).as_numpy()
            #[1,1]
            im_info = pb_utils.get_input_tensor_by_name(request, self.input_names[1]).as_numpy()
            if infer_output.shape[0] != 1 or im_info.shape[0] !=1:
                raise ValueError("Only per-request batch size of 1 is supported in dynamic batching mode.")
            infer_outputs.append(infer_output)
            im_infos.append(md.vision.LetterBoxRecord.from_dict(json.loads(im_info[0][0])))
        
        # 组batch
        batch_input0 = np.concatenate(infer_outputs, axis=0)  # shape: [N,84,6400]
        print("batch_input0 shape: ", batch_input0.shape)
        # list[np.array],list[LetterBoxRecord]
        # 其中list[0].shape[0] == len()// 可能存在多输入的情况所以是list，im_infos的长度应该就是batch的长度
        results = self.postprocessor_.run([batch_input0], im_infos)
        responses = []
        #results list[list[DetectionResult]]
        for _results in results:
            #_results:list[DetectionResult]
            result_ = [] #这是一个batch的结果多个检测框
            for result in _results:
                # result：DetectionResult
                result_.append(result.to_dict())
            result_bytes = json.dumps(result_) # 如果直接np.array(result_bytes)就是一个0维tensor
            output_tensor = pb_utils.Tensor(self.output_names[0], np.array([result_bytes], dtype=np.object_))
            responses.append(pb_utils.InferenceResponse(output_tensors=[output_tensor]))
        return responses

    def finalize(self):
        print('Cleaning up...')
