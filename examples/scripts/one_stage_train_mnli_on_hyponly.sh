CUDA_DEVICE=0
SEED=1010

METRICS='{"accuracy":{"type":"categorical_accuracy"}}'
OUTPUT_FOLDER=histories/test

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE READER_DEBUG=1 python -W ignore::UserWarning __main__.py assebmle_debiased_train \
    --train-set-param-path examples/configs/dataset/combined/mnli_train_combine_hyponly.jsonnet \
    --validation-set-param-path examples/configs/dataset/mnli_dev.jsonnet \
    --training-param-path examples/configs/training/mnli_bucket_loader_training_hard_lr_5.jsonnet \
    --main-model-param-path examples/configs/main_model/basic_bert_classifier.jsonnet \
    --bias-only-model-param-path examples/configs/bias_only_model/non_linear_3_layer_classifier.jsonnet \
    -s $OUTPUT_FOLDER --force \
    --ebd-mode one_stage_cascade_partial_input \
    --ebd-loss poe \
    --metrics $METRICS \
    --seed ${SEED}




        # --projection-head-param-path configs/model/projection_head/non_linear_3layer.jsonnet \
    # --contrastive-loss info-nce --contrastive-loss-args ${LOSS_ARGS} \