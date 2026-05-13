# Next-Token-Failures

# This is a fork for the paper: "Predicting the Order of Upcoming Tokens Improves Language Modeling"

Commands used in the paper:
```
NTP (3, 3): https://wandb.ai/zaydzuhri/next-token-failures/runs/5qjv5knp (Number of parameters: 14.22M  Number of non-embedding parameters: 14.20M)
python3 train.py --model gpt --n_layers 8 --n_embd 384 --n_head 6 --n_train 300000 --n_test 10000 --epoch 100 --batch_size 4096 --dataset graph --deg 3 --path 3 --num_nodes 30 --lr 0.0003 --use_wandb --wandb_entity zaydzuhri
TOP (3, 3): https://wandb.ai/zaydzuhri/next-token-failures/runs/soxlc5i7 (Number of parameters: 14.24M  Number of non-embedding parameters: 14.21M)
python3 train.py --model gpt --n_layers 8 --n_embd 384 --n_head 6 --n_train 300000 --n_test 10000 --epoch 100 --batch_size 4096 --dataset graph --deg 3 --path 3 --num_nodes 30 --lr 0.0003 --use_wandb --wandb_entity zaydzuhri --use_top
MTP-2 (3, 3): https://wandb.ai/zaydzuhri/next-token-failures/runs/kqxm1j01 (Number of parameters: 16.00M  Number of non-embedding parameters: 15.97M)
python3 train.py --model gpt --n_layers 8 --n_embd 384 --n_head 6 --n_train 300000 --n_test 10000 --epoch 100 --batch_size 4096 --dataset graph --deg 3 --path 3 --num_nodes 30 --lr 0.0003 --use_wandb --wandb_entity zaydzuhri --use_mtp --n_future_tokens 2
MTP-4 (3, 3): https://wandb.ai/zaydzuhri/next-token-failures/runs/l1viha9e (Number of parameters: 19.55M  Number of non-embedding parameters: 19.52M)
python3 train.py --model gpt --n_layers 8 --n_embd 384 --n_head 6 --n_train 300000 --n_test 10000 --epoch 100 --batch_size 4096 --dataset graph --deg 3 --path 3 --num_nodes 30 --lr 0.0003 --use_wandb --wandb_entity zaydzuhri --use_mtp --n_future_tokens 4
DS-MTP-2 (3,3) https://wandb.ai/zaydzuhri/next-token-failures/runs/lb0545nc:
python3 train.py --model gpt --n_layers 8 --n_embd 384 --n_head 6 --n_train 300000 --n_test 10000 --epoch 100 --batch_size 4096 --dataset graph --deg 3 --path 3 --num_nodes 30 --lr 0.0003 --use_wandb --wandb_entity zaydzuhri --use_dsmtp --n_future_tokens 2
DS-MTP-4 (3,3) https://wandb.ai/zaydzuhri/next-token-failures/runs/8mkkylmo:
python3 train.py --model gpt --n_layers 8 --n_embd 384 --n_head 6 --n_train 300000 --n_test 10000 --epoch 100 --batch_size 4096 --dataset graph --deg 3 --path 3 --num_nodes 30 --lr 0.0003 --use_wandb --wandb_entity zaydzuhri --use_dsmtp --n_future_tokens 4

NTP (3, 5): https://wandb.ai/zaydzuhri/next-token-failures/runs/scjrf5yy
python3 train.py --model gpt --n_layers 8 --n_embd 384 --n_head 6 --n_train 300000 --n_test 10000 --epoch 100 --batch_size 4096 --dataset graph --deg 3 --path 5 --num_nodes 30 --lr 0.0003 --use_wandb --wandb_entity zaydzuhri
TOP (3, 5): https://wandb.ai/zaydzuhri/next-token-failures/runs/59asffo3
python3 train.py --model gpt --n_layers 8 --n_embd 384 --n_head 6 --n_train 300000 --n_test 10000 --epoch 100 --batch_size 4096 --dataset graph --deg 3 --path 5 --num_nodes 30 --lr 0.0003 --use_wandb --wandb_entity zaydzuhri --use_top
MTP-2 (3, 5): https://wandb.ai/zaydzuhri/next-token-failures/runs/o8cc7lms
python3 train.py --model gpt --n_layers 8 --n_embd 384 --n_head 6 --n_train 300000 --n_test 10000 --epoch 100 --batch_size 4096 --dataset graph --deg 3 --path 5 --num_nodes 30 --lr 0.0003 --use_wandb --wandb_entity zaydzuhri --use_mtp --n_future_tokens 2
MTP-4 (3, 5): https://wandb.ai/zaydzuhri/next-token-failures/runs/ipd4s4qn
python3 train.py --model gpt --n_layers 8 --n_embd 384 --n_head 6 --n_train 300000 --n_test 10000 --epoch 100 --batch_size 4096 --dataset graph --deg 3 --path 5 --num_nodes 30 --lr 0.0003 --use_wandb --wandb_entity zaydzuhri --use_mtp --n_future_tokens 4


NTP (5, 3): https://wandb.ai/zaydzuhri/next-token-failures/runs/pjnrcf3g
python3 train.py --model gpt --n_layers 8 --n_embd 384 --n_head 6 --n_train 300000 --n_test 10000 --epoch 100 --batch_size 4096 --dataset graph --deg 5 --path 3 --num_nodes 30 --lr 0.0003 --use_wandb --wandb_entity zaydzuhri
TOP (5, 3): https://wandb.ai/zaydzuhri/next-token-failures/runs/z6dd70rk
python3 train.py --model gpt --n_layers 8 --n_embd 384 --n_head 6 --n_train 300000 --n_test 10000 --epoch 100 --batch_size 4096 --dataset graph --deg 5 --path 3 --num_nodes 30 --lr 0.0003 --use_wandb --wandb_entity zaydzuhri --use_top
MTP-2 (5, 3): https://wandb.ai/zaydzuhri/next-token-failures/runs/hq9jqrxy
python3 train.py --model gpt --n_layers 8 --n_embd 384 --n_head 6 --n_train 300000 --n_test 10000 --epoch 100 --batch_size 4096 --dataset graph --deg 5 --path 3 --num_nodes 30 --lr 0.0003 --use_wandb --wandb_entity zaydzuhri --use_mtp --n_future_tokens 2
MTP-4 (5, 3): https://wandb.ai/zaydzuhri/next-token-failures/runs/42y0drxw
python3 train.py --model gpt --n_layers 8 --n_embd 384 --n_head 6 --n_train 300000 --n_test 10000 --epoch 100 --batch_size 4096 --dataset graph --deg 5 --path 3 --num_nodes 30 --lr 0.0003 --use_wandb --wandb_entity zaydzuhri --use_mtp --n_future_tokens 4


NTP (5, 5): https://wandb.ai/zaydzuhri/next-token-failures/runs/s5z5wjx4
python3 train.py --model gpt --n_layers 8 --n_embd 384 --n_head 6 --n_train 300000 --n_test 10000 --epoch 100 --batch_size 4096 --dataset graph --deg 5 --path 5 --num_nodes 30 --lr 0.0003 --use_wandb --wandb_entity zaydzuhri
TOP (5, 5): https://wandb.ai/zaydzuhri/next-token-failures/runs/02edlxcl
python3 train.py --model gpt --n_layers 8 --n_embd 384 --n_head 6 --n_train 300000 --n_test 10000 --epoch 100 --batch_size 4096 --dataset graph --deg 5 --path 5 --num_nodes 30 --lr 0.0003 --use_wandb --wandb_entity zaydzuhri --use_top
MTP-2 (5, 5): https://wandb.ai/zaydzuhri/next-token-failures/runs/bcx0fh9b
python3 train.py --model gpt --n_layers 8 --n_embd 384 --n_head 6 --n_train 300000 --n_test 10000 --epoch 100 --batch_size 4096 --dataset graph --deg 5 --path 5 --num_nodes 30 --lr 0.0003 --use_wandb --wandb_entity zaydzuhri --use_mtp --n_future_tokens 2
MTP-4 (5, 5): https://wandb.ai/zaydzuhri/next-token-failures/runs/lyofozh0
python3 train.py --model gpt --n_layers 8 --n_embd 384 --n_head 6 --n_train 300000 --n_test 10000 --epoch 100 --batch_size 4096 --dataset graph --deg 5 --path 5 --num_nodes 30 --lr 0.0003 --use_wandb --wandb_entity zaydzuhri --use_mtp --n_future_tokens 4
```

![](https://github.com/gregorbachmann/Next-Token-Failures/blob/main/imgs/cleverhans.png)

This is the code used to produce the results presented in the paper <https://arxiv.org/abs/2403.06963>.

## Requirements
The following packages are needed to run the code:
1. *torch* 2.2.0
2. *transformers* 4.37.2
3. *numpy* 1.26.3
4. *tqdm* 4.66.1
5. *wandb* 0.16.2


## Usage
In order to train a GPT-style model from scratch with standard next-token prediction on a star graph with degree 2 and path length 5 with 50 possible node values, run the command
> python3 train.py --model gpt --n_layers 6 --n_embd 384 --n_head 6 --n_train 200000 --n_test 20000  --batch_size 256 --dataset graph --deg 2 --path 5 --num_nodes 50 --lr 0.0001

To train the same model using the reverse encoding, add the flag *--reverse*. In order to train with our teacherless objective, add the flag --teacherless. 

To finetune a pre-trained model like GPT2-large, run the command
>python3 finetune.py --model gpt2-large --n_train 200000 --n_test 20000  --batch_size 16 --dataset graph --deg 2 --path 5 --num_nodes 50 --lr 0.00001
>
Similarly, you can finetune a Pythia model using the flag --model pythia-410m-deduped. You can also add the flags for reversing and teacherless training as outlined above.
