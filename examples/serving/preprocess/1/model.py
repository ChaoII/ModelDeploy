import json
import numpy as np
import time
import traceback
import sys
import json
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
                #[1,H,W,3]每个请求的第一维必须为1
                image_array = pb_utils.get_input_tensor_by_name(request, self.input_names[0]).as_numpy()
                if image_array.shape[0] != 1:
                    raise ValueError("Only per-request batch size of 1 is supported in dynamic batching mode.")
                im_list.append(image_array[0])
            print("batch_input batch_size: ", len(im_list))
            # 输出为list[np.array]也就是多张图片 outputs：List[Tensor]
            #其中len(outputs)是一个模型的输出数量，有些模型为多输出，当然yolo是单输出，
            #Tensor是带batch的Tensor，其batch数与requests数相同，im_infos:List[LetterBoxRecord],len(im_infos)为batch数
            outputs, im_infos = self.preprocessor_.run(im_list)
            for i in range(len(requests)):
                # 取索引0因为yolo11n预处理后只有一个输出Tensor，然后拿出每一个batch中的数据
                # shape=[3,H,W]
                out_images = outputs[0].to_numpy()[i]
                # z注意由于tritonbatch推理的需要，必须将输出打包成[N,...]
                #np.array(json.dumps(im_infos[i].to_dict()))是0维tensor
                #np.array([json.dumps(im_infos[i].to_dict())])是1维tensor，那么shape就是（1，）
                out_image_infos = np.array([json.dumps(im_infos[i].to_dict())], dtype=np.object_)
                # 返回时扩充batch维度[1,3,H,W]
                output_tensor_0 = pb_utils.Tensor(self.output_names[0], np.expand_dims(out_images,0))
                # 返回时扩充维度[1,1]
                output_tensor_1 = pb_utils.Tensor(self.output_names[1], np.expand_dims(out_image_infos,0))
                responses.append(pb_utils.InferenceResponse(output_tensors=[output_tensor_0, output_tensor_1]))
        except Exception as e:
            print("Error:", e)
            for _ in requests:
                responses.append(pb_utils.InferenceResponse(
                    output_tensors=[], error=pb_utils.TritonError(str(e))))
        return responses

    def finalize(self):
        print('[preprocess] finalize called')
