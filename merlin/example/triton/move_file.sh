mkdir -p model_repo/hps_tf_triton/1
mv hps_tf_triton_tf_saved_model model_repo/hps_tf_triton/1/model.savedmodel
mv hps_tf_triton_sparse_0.model model_repo/hps_tf_triton
cp hps_tf_triton.json model_repo/hps_tf_triton
cp config.pbtxt model_repo/hps_tf_triton
tree model_repo/hps_tf_triton