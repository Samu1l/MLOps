import pickle

import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        self.scaler = pickle.load(open("/assets/scaler.pkl", "rb"))

    def execute(self, requests):
        responses = []
        for request in requests:
            features = pb_utils.get_input_tensor_by_name(request, "feature").as_numpy()
            features = self.scaler.transform(features)
            output_scaled_features = pb_utils.Tensor("input", features)

            inference_response = pb_utils.InferenceResponse(output_tensors=[output_scaled_features])
            responses.append(inference_response)
        return responses
