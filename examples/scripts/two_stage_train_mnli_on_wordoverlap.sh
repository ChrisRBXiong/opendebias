CUDA_DEVICE=0
SEED=1010

TRAIN_BIAS='{"dataset_name":"mnli_train","file_name":"examples/bias/mnli-hans/mind-trade-bias/train.json"}'
METRICS='{"accuracy":{"type":"categorical_accuracy"}}'
OUTPUT_FOLDER=examples/histories/two_stage_train_mnli_on_wordoverlap

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE READER_DEBUG=0 python -W ignore::UserWarning __main__.py assebmle_debiased_train \
    --train-set-param-path examples/configs/dataset/mnli_train.jsonnet \
    --validation-set-param-path examples/configs/dataset/mnli_dev.jsonnet \
    --training-param-path examples/configs/training/mnli_bucket_loader_training_hans_lr_5.jsonnet \
    --main-model-param-path examples/configs/main_model/basic_bert_classifier.jsonnet \
    --bias-file-args "[${TRAIN_BIAS}]" \
    -s $OUTPUT_FOLDER --force \
    --ebd-mode two_stage \
    --ebd-loss poe \
    --metrics $METRICS \
    --seed ${SEED}