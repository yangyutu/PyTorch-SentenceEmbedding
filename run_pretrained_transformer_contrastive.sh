export CUDA_VISIBLE_DEVICES="0"
pretrained_model_name="bert-base-uncased"
#pretrained_model_name="sentence-transformers/bert-base-nli-cls-token"
#pretrained_model_name="sentence-transformers/bert-base-nli-mean-tokens"
dataset_name="snli"


python tasks/run_pretrained_transformer_contrastive.py \
--dataset_name ${dataset_name} \
--pretrained_model_name ${pretrained_model_name} \
--gpus 1 \
--max_epochs 1 \
--lr 2e-4 \
--truncate 128 \
--batch_size 64 \
--num_workers 16 \
--precision 32 \
--project_name sentence_bert_finetune \
--pooling_method mean_pooling \
--default_root_dir ./experiments/logs \
--normalize_embeddings \
--val_step_interval 1000 \
#--log_model \
