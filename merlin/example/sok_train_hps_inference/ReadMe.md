# SOK train HPS inferece example

### Requirements
It is recommended to use the nvcr image for testing. Currently, the Merlin image available at <mark>nvcr.io/nvidia/merlin/merlin-tensorflow:nightly</mark> includes both SOK and HPS installed. However, there has not been a recent release of HugeCTR, so the code is not up-to-date. If you want to use the latest environment, you can choose the <mark>nvcr.io/nvidia/tensorflow:24.03-tf2-py3</mark> image and install SOK and HPS for testing.

pip install scikit-build
pip install sparse_operation_kit --no-build-isolation

sok missing safe_embedding_lookup_sparse

dump_model dump dense and sok, and then
generate_kv_file_for_hps gen the sparse for hps from the sok files
by convert_sok_weights_for_hps, it is simply dtype + dim to convert the sok weights to hps weights

then, if we can create convert_tfra_for_hps, then we can use the tfra for hps


### How to test
Follow these steps to test:

1. **SOK train**  
   horovodrun -np 2 python sok_train.py

2. **HPS inference**  
   python hps_use_sok_weight_inference.py

