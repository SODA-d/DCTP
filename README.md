## Environment Setup

Conda environment download:

```
https://pan.quark.cn/s/9bb36810e143?pwd=dES7
```


```bash
# Create directory and extract
mkdir -p ~/anaconda3/envs/DCTP
tar -xzvf learn-to-follow.tar.gz -C ~/anaconda3/envs/DCTP

# Activate environment
conda activate DCTP
```

## Training

```bash
python main.py  --actor_critic_share_weights=True --batch_size=16384 --env=PogemaMazes-v0 --exploration_loss_coeff=0.023 --extra_fc_layers=1 --gamma=0.9756 --hidden_size=512 --intrinsic_target_reward=0.01 --learning_rate=0.00022 --lr_schedule=constant --network_input_radius=5 --num_filters=64 --num_res_blocks=8 --num_workers=8 --optimizer=adam --ppo_clip_ratio=0.2   --train_for_env_steps=1000000000 --use_rnn=True
```

## Evaluation

```bash
python eval.py 
```
