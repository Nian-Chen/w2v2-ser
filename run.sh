#!/bin/bash
stage=1
stop_stage=2
train_root_dir=data/train
test_root_dir=data/Data_B
. kaldi_utils/parse_options.sh || exit 1;
stage=$stage
stop_stage=$stop_stage
train_root_dir=$train_root_dir
test_root_dir=$test_root_dir

# 默认使用所有GPU，也可自定义cuda_visible_devices,n_gpus
n_gpus=`nvidia-smi -L|wc -l`
# 转数组
gpu_list=(`seq $n_gpus`)
# [1 2 3 4] --> [0 1 2 3] --> 0,1,2,3
for i in `seq $n_gpus`
do
index=$[i-1]
gpu_list[index]=$[${gpu_list[index]}-1]
# echo ${gpu_list[index]}
done
cuda_visible_devices=`echo ${gpu_list[*]}|awk '{gsub(" ",",");print $0}'`
# 0,1,2,3

# ！！！！！！！！！！！若自定义只使用前两块gpu，可解除如下注释！！！！！！！！！！！
#cuda_visible_devices=0,1
#n_gpus=2
echo "n_gpus = $n_gpus"
echo "cuda_visible_devices = $cuda_visible_devices"


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  # do augmentation by separating
  # 耗时较长，已将分离后的人声音频放在data/train和data/Data_B下了，故此步不需要运行(功能是没问题的)
  echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
  echo "stage 0: do augmentation by separating"
  python separate_aug.py --datasets_path=hf_datasets/train --wav_root_dir=$train_root_dir --processor_path=processor || exit 1;
  python separate_aug.py --datasets_path=hf_datasets/testb --wav_root_dir=$test_root_dir --processor_path=processor || exit 1;
fi
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  # train
  echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
  echo "stage 0: do training"
  NCCL_P2P_LEVEL=NVL CUDA_VISIBLE_DEVICES=$cuda_visible_devices python -m  torch.distributed.launch  \
  --nproc_per_node 4 \
  --master_port  6666 finetune_w2v2_classify.py \
  --output_dir=training_output \
  --logging_dir=training_output/log \
  --num_train_epochs=30 \
  --logging_strategy=steps \
  --logging_steps=10 \
  --logging_first_step \
  --per_device_train_batch_size=16 \
  --per_device_eval_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --save_total_limit=5 \
  --evaluation_strategy=steps \
  --save_strategy=steps \
  --save_steps=500 \
  --eval_steps=500 \
  --learning_rate=5e-5 \
  --lr_scheduler_type=linear \
  --use_czc_lr_scheduler=True \
  --warmup_ratio=0.1 \
  --group_by_length \
  --fp16=True \
  --model_name_or_path=XLSR-53 \
  --dataset_path=hf_datasets \
  --processor_path=processor \
  --dataloader_num_workers=2 \
  --freeze_feature_extractor=True \
  --speed_perturb=False \
  --prediction_loss_only=False \
  --verbose_log=False \
  --ignore_data_skip \
  --metric_for_best_model=eval_f1score \
  --greater_is_better=True \
  --wav_root_dir=data/train || exit 1;
fi
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  # decode
  # 榜上成绩对应的那个模型不小心被我删除了，以下两个是重新训的，应该不会比榜上成绩低
  model_path=best-checkpoint-4000
  # model_path=training_output/-checkpoint-4000
  output_file=submission.csv
  echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
  echo "stage 2: decoding"
  python decode.py \
  --datasets_path=hf_datasets/testb \
  --wav_root_dir=data/Data_B \
  --processor_path=processor \
  --model_path=$model_path \
  --output_file_name=$output_file || exit 1;
fi
