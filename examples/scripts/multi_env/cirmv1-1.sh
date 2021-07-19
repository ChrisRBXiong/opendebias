CUDA_DEVICE=2

PREDEFINED_GROUP_FILE='{"dataset_name":"mnli_train","file_name":"data/bias/mnli-hans/mind-trade-bias/train-eiil-random-1010-group.json"}'
METRICS='{"accuracy":{"type":"categorical_accuracy"}}'
# OUTPUT_FOLDER=examples/histories/irm_v1/basic_bert_lr_5_test/


for METHOD in cIRMv1
do
    for SEED in 13214 37462 54324 28987 54673
    do
        for ADAPT_WEIGHT in 1e-3 1e-2 1e-1 1
        do
            for ASCEND_RATE in 0.3 0.6
            do
                OUTPUT_FOLDER=examples/histories/multi_env/${METHOD}/basic_bert_lr_5_epoch_3_lambda_${ADAPT_WEIGHT}_${ASCEND_RATE}-SEED-${SEED}/
                FILE=$OUTPUT_FOLDER/metrics_epoch_2.json
                if [ ! -f "$FILE" ]; then
                    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE READER_DEBUG=0 python -W ignore::UserWarning __main__.py multi_environment_train \
                        --train-set-param-path examples/configs/dataset/mnli_train.jsonnet \
                        --validation-set-param-path examples/configs/dataset/mnli_dev.jsonnet \
                        --training-param-path examples/configs/training/mnli_group_base_hans_lr_5.jsonnet \
                        --model-param-path examples/configs/main_model/basic_bert_classifier.jsonnet \
                        --predefined-group-file "[${PREDEFINED_GROUP_FILE}]" \
                        --multi-env-loss ${METHOD} \
                        --loss-args '{"weight_adapt":'${ADAPT_WEIGHT}'}' \
                        --sampler predefined_group \
                        -s $OUTPUT_FOLDER --force \
                        --metrics $METRICS \
                        --seed ${SEED} \
                        --overrides '{"trainer":{"num_epochs":3,"weight_adapt_ascend_ratio":'${ASCEND_RATE}'}}'
                fi
            done
        done
    done
done