#RAFI OPTIMIZATION: your all gathers weren't being hidden!
export LIBTPU_INIT_ARGS="--xla_enable_async_all_gather=true TPU_MEGACORE=MEGACORE_DENSE"
python -m EasyLM.models.llama.llama_train \
    --mesh_dim='1,-1,1' \
    --total_steps=300 \
    --log_freq=10 \
    --load_llama_config='3b' \
    --tokenizer.vocab_file='gs://max-experiments/open_llama_2_tokenizer.model' \
    --train_dataset.type='json' \
    --train_dataset.text_processor.fields='inputs,targets' \
    --train_dataset.json_dataset.path='gs://max-experiments/cot_fs_noopt_train.jsonl' \
    --train_dataset.json_dataset.seq_length=1024 \
    --train_dataset.json_dataset.batch_size=32 \
    --train_dataset.json_dataset.tokenizer_processes=16
