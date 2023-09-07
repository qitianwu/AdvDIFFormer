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

python main.py --dataset arxiv --method ours3 --lr 1e-3 --weight_decay 0. --num_layers 1 --beta 0.5 \
--hidden_channels 128 --num_heads 1 --K_order 8 --kernel simple --use_residual --use_bn \
--runs 5 --epochs 500 --seed 123 --device 2

python main.py --dataset twitch --method ours3 --lr 1e-4 --weight_decay 0. --num_layers 1 --beta 0.5 \
--hidden_channels 64 --num_heads 2 --K_order 3 --kernel simple --use_residual \
--runs 5 --epochs 500 --seed 123 --device 1