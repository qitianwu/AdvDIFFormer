## GCN encoder

# ours

python main.py --dataset arxiv --method ours --lr 1e-2 --weight_decay 0. --num_layers 2 \
--hidden_channels 64 --num_heads 1 --kernel simple --use_residual --use_weight --use_bn \
--use_reg --reg_weight 1e4 --num_aug_branch 5 --modify_ratio 0.2 --rewiring_type replace \
--runs 5 --epochs 500 --seed 123 --device 1

python main.py --dataset arxiv --method ours --lr 1e-2 --weight_decay 0. --num_layers 2 \
--hidden_channels 64 --num_heads 1 --kernel simple --use_weight --use_bn --use_residual \
--runs 5 --epochs 500 --seed 123 --device 1

python main.py --dataset arxiv --method ours2 --lr 1e-3 --weight_decay 0. --num_layers 2 \
--hidden_channels 128 --num_heads 1 --K_order 3 --kernel simple --use_residual --use_bn \
--use_reg --reg_weight 1e5 --num_aug_branch 5 --modify_ratio 0.1 --rewiring_type replace \
--runs 5 --epochs 500 --seed 123 --device 1

python main.py --dataset arxiv --method ours2 --lr 1e-3 --weight_decay 0. --num_layers 1 \
--hidden_channels 128 --num_heads 1 --K_order 3 --kernel simple --use_residual --use_bn \
--use_reg --reg_weight 1e5 --num_aug_branch 5 --modify_ratio 0.2 --rewiring_type replace \
--runs 5 --epochs 500 --seed 123 --device 1

python main.py --dataset arxiv --method ours3 --lr 1e-3 --weight_decay 0. --num_layers 1 --beta 0.8 \
--hidden_channels 128 --num_heads 1 --K_order 8 --kernel simple --use_residual --use_bn \
--use_reg --reg_weight 1e4 --num_aug_branch 5 --modify_ratio 0.2 --rewiring_type replace \
--runs 5 --epochs 500 --seed 123 --device 2



python main.py --dataset twitch --method ours --lr 1e-3 --weight_decay 0. --num_layers 2 \
--hidden_channels 64 --num_heads 1 --kernel simple --use_residual --use_weight --use_block \
--use_reg --reg_weight 1e5  --num_aug_branch 5 --modify_ratio 0.2 --rewiring_type replace \
--runs 5 --epochs 500 --seed 123 --device 1

python main.py --dataset twitch --method ours --lr 1e-3 --weight_decay 0. --num_layers 2 \
--hidden_channels 64 --num_heads 1 --kernel simple --use_residual --use_weight --use_block \
--runs 5 --epochs 500 --seed 123 --device 1

python main.py --dataset twitch --method ours2 --lr 1e-4 --weight_decay 0. --num_layers 1 \
--hidden_channels 64 --num_heads 3 --K_order 3 --kernel simple --use_residual --use_block \
--use_reg --reg_weight 1e3  --num_aug_branch 5 --modify_ratio 0.2 --rewiring_type replace \
--runs 5 --epochs 500 --seed 123 --device 1

python main.py --dataset twitch --method ours2 --lr 1e-4 --weight_decay 0. --num_layers 1 \
--hidden_channels 64 --num_heads 3 --K_order 3 --kernel simple --use_residual --use_block \
--use_reg --reg_weight 1e4  --num_aug_branch 5 --modify_ratio 0.2 --rewiring_type replace \
--runs 5 --epochs 500 --seed 123 --device 1


python main.py --dataset twitch --method ours3 --lr 1e-4 --weight_decay 0. --num_layers 1 --beta 0.5 \
--hidden_channels 64 --num_heads 1 --K_order 3 --kernel simple --use_residual --use_block \
--use_reg --reg_weight 1e4  --num_aug_branch 5 --modify_ratio 0.2 --rewiring_type replace \
--runs 5 --epochs 500 --seed 123 --device 1

# baselines
python main.py --method erm --encoder gcn --dataset arxiv --lr 0.01 --num_layers 3 --hidden_channels 128 --weight_decay 0. --epochs 500 --device 0
python main.py --method erm --encoder gcn --dataset twitch --lr 0.01 --num_layers 2 --hidden_channels 64 --weight_decay 0. --epochs 500 --device 3

python main.py --method dropedge --encoder gcn --dataset arxiv --lr 0.01 --num_layers 2 --hidden_channels 128 --weight_decay 0. --modify_ratio 0.2 --epochs 500 --device 3
python main.py --method dropedge --encoder gcn --dataset twitch --lr 0.01 --num_layers 2 --hidden_channels 64 --weight_decay 0. --modify_ratio 0.2 --epochs 500 --device 3

python main.py --method reg --encoder gcn --dataset arxiv --lr 0.01 --num_layers 3 --hidden_channels 128 --weight_decay 0. \
 --reg_weight 1e3 --num_aug_branch 5 --modify_ratio 0.2 --rewiring_type replace --runs 5 --epochs 500 --seed 123 --device 3

python main.py --method reg --encoder gcn --dataset twitch --lr 0.01 --num_layers 2 --hidden_channels 64 --weight_decay 0. \
 --reg_weight 1e4 --num_aug_branch 5 --modify_ratio 0.2 --rewiring_type replace --runs 5 --epochs 500 --seed 123 --device 3


python main.py --dataset arxiv --method ours3 --lr 1e-3 --weight_decay 0. --num_layers 1 --beta 0.5 \
--hidden_channels 128 --num_heads 1 --K_order 4 --kernel simple --use_residual --use_bn \
--runs 5 --epochs 500 --seed 123 --device 1

python main.py --dataset arxiv --method ours3 --lr 1e-3 --weight_decay 0. --num_layers 1 --beta 0. \
--hidden_channels 128 --num_heads 1 --K_order 4 --kernel simple --use_residual --use_bn \
--use_reg --reg_weight 1e4 --num_aug_branch 5 --modify_ratio 0.2 --rewiring_type replace \
--runs 5 --epochs 500 --seed 123 --device 2

python main.py --dataset twitch --method ours3 --lr 1e-4 --weight_decay 0. --num_layers 1 --beta 0.5 \
--hidden_channels 64 --num_heads 1 --K_order 3 --kernel simple --use_residual --use_block \
--runs 5 --epochs 500 --seed 123 --device 1

python main.py --dataset twitch --method ours3 --lr 1e-4 --weight_decay 0. --num_layers 1 --beta 0. \
--hidden_channels 64 --num_heads 1 --K_order 3 --kernel simple --use_residual --use_block \
--use_reg --reg_weight 1e4  --num_aug_branch 5 --modify_ratio 0.2 --rewiring_type replace \
--runs 5 --epochs 500 --seed 123 --device 1


# new for ours3

python main.py --dataset arxiv --method ours3 --lr 1e-3 --weight_decay 0. --num_layers 1 --beta 0.8 \
--hidden_channels 128 --num_heads 2 --K_order 1 --theta 1.0 --kernel simple --use_residual --use_bn \
--runs 5 --epochs 500 --seed 123 --device 2

ours3 gcn: 0.001: 128: 1 0.8 2 series 1.0 1
60.96 $\pm$ nan 54.96 $\pm$ nan 53.59 $\pm$ nan 51.38 $\pm$ nan 49.73 $\pm$ nan 51.57 ± nan


python main.py --dataset twitch --method ours3 --lr 1e-4 --weight_decay 0. --num_layers 1 --beta 1.0 \
--hidden_channels 64 --num_heads 2 --solver series --K_order 1 --use_residual \
--runs 5 --epochs 500 --seed 123 --device 1

ours3 gcn: 0.0001: 64: 1 1.0 2 series 1.0 1
75.16 $\pm$ nan 63.39 $\pm$ nan 66.87 $\pm$ nan 63.43 $\pm$ nan 65.80 $\pm$ nan 56.42 $\pm$ nan 60.79 $\pm$ nan 62.66 ± nan


python main.py --dataset twitch --method ours3 --lr 1e-3 --weight_decay 0. --num_layers 1 --beta 0.8 \
--hidden_channels 64 --num_heads 2 --solver inverse --theta 0.0 --use_residual \
--runs 5 --epochs 500 --seed 123 --device 1

ours3 gcn: 0.001: 64: 1 0.8 2 inverse 0.0 3
74.94 $\pm$ nan 63.57 $\pm$ nan 66.89 $\pm$ nan 64.26 $\pm$ nan 65.95 $\pm$ nan 56.13 $\pm$ nan 60.46 $\pm$ nan 62.74 ± nan


python main.py --dataset twitch --method ours3 --lr 1e-3 --weight_decay 0. --num_layers 1 --beta 0. \
--hidden_channels 64 --num_heads 2 --solver inverse --theta 0.0 --use_residual \
--runs 5 --epochs 500 --seed 123 --device 0