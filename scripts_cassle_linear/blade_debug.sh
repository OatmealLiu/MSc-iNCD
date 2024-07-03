UCL="swav"
DATASET_NAME="cifar100"
ENCODER_NAME="vit_dino"
BS=128
TASKS=5
EPOCHS=1

for step in $(seq 0 4)
do
  python -W ignore train_cassle_linear_iNCD.py \
          --ucl_method $UCL \
          --epochs $EPOCHS \
          --batch_size $BS \
          --dataset_name $DATASET_NAME \
          --num_steps $TASKS \
          --model_name $ENCODER_NAME \
          --current_step $step \
          --apply_l2weights \
          --num_mlp_layers 1 \
          --dino_pretrain_path ./models/dino_weights/dino_vitbase16_pretrain.pth \
          --model_head LinearHead \
          --seed 10 \
          --exp_root ./outputs_cassle_swav/ \
          --weights_root ./models/single_weights_cassle_swav/ \
          --wandb_mode online \
          --wandb_entity oatmealliu
done