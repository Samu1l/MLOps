from functools import lru_cache
from typing import List

import numpy as np
from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput
from tritonclient.utils import np_to_triton_dtype


@lru_cache
def get_client():
    return InferenceServerClient(url="0.0.0.0:8500")


def call_triton_classifier(feature: List[float]):
    triton_client = get_client()
    feature = np.array([feature], dtype=np.float32)

    input_feature = InferInput(name="input", shape=feature.shape, datatype=np_to_triton_dtype(feature.dtype))
    input_feature.set_data_from_numpy(feature, binary_data=True)

    infer_output = InferRequestedOutput("output", binary_data=True)
    query_response = triton_client.infer("onnx-classifier", [input_feature], outputs=[infer_output])
    predicted_classes = query_response.as_numpy("output")[0]
    return predicted_classes


def call_triton_ensemble_classifier(feature: List[float]):
    triton_client = get_client()
    feature = np.array([feature], dtype=np.float32)

    input_feature = InferInput(name="feature", shape=feature.shape, datatype=np_to_triton_dtype(feature.dtype))
    input_feature.set_data_from_numpy(feature, binary_data=True)

    infer_output = InferRequestedOutput("output", binary_data=True)
    query_response = triton_client.infer("ensemble-onnx", [input_feature], outputs=[infer_output])
    predicted_classes = query_response.as_numpy("output")[0]
    return predicted_classes


def main():
    # проверяем, правильно ли работает scaling
    features = [
        [12.86, 1.35, 2.32, 18.0, 122.0, 1.51, 1.25, 0.21, 0.94, 4.1, 0.76, 1.29, 630.0],
        [12.67, 0.98, 2.24, 18.0, 99.0, 2.2, 1.94, 0.3, 1.46, 2.62, 1.23, 3.16, 450.0],
    ]

    scaled_features = [
        [
            0.42727273,
            0.09368635,
            0.51612903,
            0.3814433,
            0.60273973,
            0.18275862,
            0.26686217,
            0.13461538,
            0.16719243,
            0.19457014,
            0.24175824,
            0.00819672,
            0.27738377,
        ],
        [
            0.36969697,
            0.01832994,
            0.47311828,
            0.3814433,
            0.28767123,
            0.42068966,
            0.46920821,
            0.30769231,
            0.33123028,
            0.06063348,
            0.75824176,
            0.77459016,
            0.1355398,
        ],
    ]

    preds = np.array([call_triton_classifier(feature) for feature in scaled_features])
    preds_ens = np.array([call_triton_ensemble_classifier(feature) for feature in features])
    assert np.allclose(preds, preds_ens, atol=1e-5)


if __name__ == "__main__":
    main()
