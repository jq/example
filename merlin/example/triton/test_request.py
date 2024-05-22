import os

from train import generate_random_samples, args

num_gpu = 1
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(num_gpu)))

import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import *


def send_inference_requests(num_requests, num_samples):
    triton_client = httpclient.InferenceServerClient(url="localhost:8000", verbose=True)
    triton_client.is_server_live()
    triton_client.get_model_repository_index()

    for i in range(num_requests):
        print("--------------------------Request {}--------------------------".format(i))
        key_tensor, dense_tensor, _ = generate_random_samples(num_samples, args["vocabulary_range_per_slot"], args["dense_dim"])

        inputs = [
            httpclient.InferInput("input_6",
                                  key_tensor.shape,
                                  np_to_triton_dtype(np.int64)),
            httpclient.InferInput("input_7",
                                  dense_tensor.shape,
                                  np_to_triton_dtype(np.float32)),
        ]

        inputs[0].set_data_from_numpy(key_tensor)
        inputs[1].set_data_from_numpy(dense_tensor)
        outputs = [
            httpclient.InferRequestedOutput("output_1")
        ]

        # print("Input key tensor is \n{}".format(key_tensor))
        # print("Input dense tensor is \n{}".format(dense_tensor))
        model_name = "hps_tf_triton"
        with httpclient.InferenceServerClient("localhost:8000") as client:
            response = client.infer(model_name,
                                    inputs,
                                    outputs=outputs)
            result = response.get_response()

            print("Response details:\n{}".format(result))

send_inference_requests(num_requests = 5, num_samples = 128)
