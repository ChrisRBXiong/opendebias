CUDA_DEVICE=0
METRICS='{"accuracy":{"type":"categorical_accuracy"}}'

function Train()
# $1 OUTPUT_FOLDER
# $2 SEED
# $3 DEBIAS_MODE
# $4 BIAS_FILE_ARGS
# $5 FILE
# $6 DEBIAS_ARGS
{
    OUTPUT_FOLDER=$1
    ACCUMULATE=$2
    SEED=$3
    DEBIAS_MODE=$4
    FILE=${OUTPUT_FOLDER}/$5
    BIAS_FILE_ARGS=$6
    LOSS_ARGS=$7
    SAMPLER_ARGS=$8
    DEBIAS_ARGS=$9
    
    # Train Model
    if [ ! -f "$FILE" ]; then
        CUDA_VISIBLE_DEVICES=$CUDA_DEVICE READER_DEBUG=0 python -W ignore::UserWarning __main__.py assebmle_debiased_train \
            --train-set-param-path examples/configs/dataset/mnli_train.jsonnet \
            --validation-set-param-path examples/configs/dataset/mnli_dev.jsonnet \
            --training-param-path examples/configs/training/mnli_bucket_loader_training_hans_lr_5.jsonnet \
            --main-model-param-path examples/configs/main_model/basic_bert_classifier.jsonnet \
            --ebd-mode two_stage \
            --ebd-loss ${DEBIAS_MODE} --ebd-loss-args ${DEBIAS_ARGS} --bias-file-args "[${BIAS_FILE_ARGS}]" \
            --projection-head-param-path examples/configs/projection_head/non_linear_3layer.jsonnet \
            --contrastive-loss info-nce --contrastive-loss-args ${LOSS_ARGS} --contrastive \
            --lambda-contrastive 0.10 --lambda-ebd 0.90 \
            --sampler bias-free  --sampler-args ${SAMPLER_ARGS} \
            -s $OUTPUT_FOLDER --force \
            --seed ${SEED} \
            --metrics $METRICS \
            --overrides '{"trainer":{"num_gradient_accumulation_steps":'${ACCUMULATE}'}}'
    fi
}



TRAIN_BLOCKING_BIAS='{"dataset_name":"mnli_train","file_name":"../data/bias/mnli-hans/mind-trade-bias/train-blocking-10.0-15.json"}'
TRAIN_BIAS='{"dataset_name":"mnli_train","file_name":"data/bias/mnli-hans/mind-trade-bias/train.json"}'


OUTPUT_BASE_FOLDER=examples/histories/contrastive_ebd/mnli/bias-free/basic_bert_lr_5_${SEED}-2-P-1-N-1/

SINGLE_BATCH_SIZE=32
FULL_POS_DENOMINATOR=false

for SEED in 3420
do
    for DEBIAS_MODE in poe
    do
        LOSS=info-nce
        for REAL_BATCH_SIZE in 32
        do
            for PN in "1 1"
            do
                for TAU in 0.5
                do

                    PN_ARRAY=($PN)
                    K_POS=${PN_ARRAY[0]}
                    K_NEG=${PN_ARRAY[1]}

                    ACCUMULATE=`expr ${REAL_BATCH_SIZE} / ${SINGLE_BATCH_SIZE}`
                    OUTPUT_FOLDER=${OUTPUT_BASE_FOLDER}/bias-free/${DEBIAS_MODE}/uncali/basic_bert-lr-5-KP-${K_POS}-KN-${K_NEG}-${REAL_BATCH_SIZE}-tau-${TAU}-lamda-${LAMDA}-SEED-${SEED}
                    LOSS_ARGS='{"tau":'${TAU}',"full_pos_denominator":'${FULL_POS_DENOMINATOR}'}'
                    SAMPLER_ARGS='{"bias_prediction_file":'[${TRAIN_BLOCKING_BIAS}]',"K_pos":'${K_POS}',"K_neg":'${K_NEG}',"batch_size":'${SINGLE_BATCH_SIZE}'}'
                    DEBIAS_ARGS='{}'
                    if [ "$DEBIAS_MODE" == "learned-mixin" ]; then
                        DEBIAS_ARGS='{"input_dim":768,"penalty":0.03}'
                    fi
                    Train $OUTPUT_FOLDER $ACCUMULATE $SEED $DEBIAS_MODE metrics_epoch_2.json $TRAIN_BIAS $LOSS_ARGS $SAMPLER_ARGS $DEBIAS_ARGS
                done
            done
        done
    done
done

# ,${ANTI_BIASED_BIAS}