#python main.py --method erm --encoder gcn --dataset arxiv --lr 0.01 --num_layers 3 --hidden_channels 128 --weight_decay 0. --epochs 500 --device 0 --save_result
#python main.py --method dropedge --encoder gcn --dataset arxiv --lr 0.01 --num_layers 3 --hidden_channels 128 --weight_decay 0. --modify_ratio 0.2 --epochs 500 --device 0 --save_result
#python main.py --method reg --encoder gcn --dataset arxiv --lr 0.01 --num_layers 3 --hidden_channels 128 --weight_decay 0. \
# --reg_weight 1.0 --num_aug_branch 5 --modify_ratio 0.2 --rewiring_type replace --runs 5 --epochs 500 --seed 123 --device 0 --save_result
#python main.py --method reg --encoder gcn --dataset arxiv --lr 0.01 --num_layers 3 --hidden_channels 128 --weight_decay 0. \
# --reg_weight 1.0 --num_aug_branch 5 --modify_ratio 0.2 --rewiring_type delete --runs 5 --epochs 500 --seed 123 --device 0 --save_result
#
#python main.py --method erm --encoder gat --dataset arxiv --lr 0.01 --num_layers 3 --hidden_channels 128 --weight_decay 0. --epochs 500 --device 0 --save_result
#python main.py --method dropedge --encoder gat --dataset arxiv --lr 0.01 --num_layers 3 --hidden_channels 128 --weight_decay 0. --modify_ratio 0.2 --epochs 500 --device 0 --save_result
#python main.py --method reg --encoder gat --dataset arxiv --lr 0.01 --num_layers 3 --hidden_channels 128 --weight_decay 0. \
# --reg_weight 1.0 --num_aug_branch 5 --modify_ratio 0.2 --rewiring_type replace --runs 5 --epochs 500 --seed 123 --device 0 --save_result
#python main.py --method reg --encoder gat --dataset arxiv --lr 0.01 --num_layers 3 --hidden_channels 128 --weight_decay 0. \
# --reg_weight 1.0 --num_aug_branch 5 --modify_ratio 0.2 --rewiring_type delete --runs 5 --epochs 500 --seed 123 --device 0 --save_result
#
#python main.py --method erm --encoder difformer --dataset arxiv --lr 0.01 --num_layers 3 --hidden_channels 64 --weight_decay 0. --use_weight --use_residual --use_bn --epochs 500 --device 0 --save_result
#python main.py --method dropedge --encoder difformer --dataset arxiv --lr 0.01 --num_layers 3 --hidden_channels 64 --weight_decay 0. --modify_ratio 0.2 --use_weight --use_residual --use_bn --epochs 500 --device 0 --save_result
#python main.py --method reg --encoder difformer --dataset arxiv --lr 0.01 --num_layers 3 --hidden_channels 64 --weight_decay 0. --use_weight --use_residual --use_bn \
# --reg_weight 1.0 --num_aug_branch 5 --modify_ratio 0.2 --rewiring_type replace --runs 5 --epochs 500 --seed 123 --device 0 --save_result
#python main.py --method reg --encoder difformer --dataset arxiv --lr 0.01 --num_layers 3 --hidden_channels 64 --weight_decay 0. --use_weight --use_residual --use_bn \
# --reg_weight 1.0 --num_aug_branch 5 --modify_ratio 0.2 --rewiring_type delete --runs 5 --epochs 500 --seed 123 --device 0 --save_result
#
#
#python main.py --dataset arxiv --method ours3 --lr 1e-3 --weight_decay 0. --num_layers 1 --beta 0.5 \
#--hidden_channels 128 --num_heads 1 --K_order 8 --kernel simple --use_residual --use_bn \
#--use_reg --reg_weight 0.5 --num_aug_branch 5 --modify_ratio 0.4 --rewiring_type replace \
#--runs 5 --epochs 500 --seed 123 --device 2
#
#
#python main.py --method erm --encoder gcn --dataset twitch --lr 0.01 --num_layers 2 --hidden_channels 64 --weight_decay 0. --epochs 500 --device 0 --save_result
#python main.py --method dropedge --encoder gcn --dataset twitch --lr 0.01 --num_layers 2 --hidden_channels 64 --weight_decay 0. --modify_ratio 0.2 --epochs 500 --device 0 --save_result
#python main.py --method reg --encoder gcn --dataset twitch --lr 0.01 --num_layers 2 --hidden_channels 64 --weight_decay 0. \
# --reg_weight 1e4 --num_aug_branch 5 --modify_ratio 0.2 --rewiring_type replace --runs 5 --epochs 500 --seed 123 --device 0 --save_result
#python main.py --method reg --encoder gcn --dataset twitch --lr 0.01 --num_layers 2 --hidden_channels 64 --weight_decay 0. \
# --reg_weight 1e4 --num_aug_branch 5 --modify_ratio 0.2 --rewiring_type delete --runs 5 --epochs 500 --seed 123 --device 0 --save_result
#
#python main.py --method erm --encoder gat --dataset twitch --lr 0.01 --num_layers 2 --hidden_channels 64 --weight_decay 0. --epochs 500 --device 0 --save_result
#python main.py --method dropedge --encoder gat --dataset twitch --lr 0.01 --num_layers 2 --hidden_channels 64 --weight_decay 0. --modify_ratio 0.2 --epochs 500 --device 0 --save_result
#python main.py --method reg --encoder gat --dataset twitch --lr 0.01 --num_layers 2 --hidden_channels 64 --weight_decay 0. \
# --reg_weight 1e4 --num_aug_branch 5 --modify_ratio 0.2 --rewiring_type replace --runs 5 --epochs 500 --seed 123 --device 0 --save_result
#python main.py --method reg --encoder gat --dataset twitch --lr 0.01 --num_layers 2 --hidden_channels 64 --weight_decay 0. \
# --reg_weight 1e4 --num_aug_branch 5 --modify_ratio 0.2 --rewiring_type delete --runs 5 --epochs 500 --seed 123 --device 0 --save_result

python main.py --method erm --encoder difformer --dataset twitch --lr 0.01 --num_layers 2 --hidden_channels 64 --weight_decay 0. --use_weight --use_residual --epochs 500 --device 0 --save_result
python main.py --method dropedge --encoder difformer --dataset twitch --lr 0.01 --num_layers 2 --hidden_channels 64 --weight_decay 0. --use_weight --use_residual --modify_ratio 0.2 --epochs 500 --device 0 --save_result
python main.py --method reg --encoder difformer --dataset twitch --lr 0.01 --num_layers 2 --hidden_channels 64 --weight_decay 0. --use_weight --use_residual \
 --reg_weight 1e4 --num_aug_branch 5 --modify_ratio 0.2 --rewiring_type replace --runs 5 --epochs 500 --seed 123 --device 0 --save_result
python main.py --method reg --encoder difformer --dataset twitch --lr 0.01 --num_layers 2 --hidden_channels 64 --weight_decay 0. --use_weight --use_residual \
 --reg_weight 1e4 --num_aug_branch 5 --modify_ratio 0.2 --rewiring_type delete --runs 5 --epochs 500 --seed 123 --device 0 --save_result

python main.py --dataset twitch --method ours3 --lr 1e-4 --weight_decay 0. --num_layers 1 --beta 0.5 \
--hidden_channels 64 --num_heads 2 --K_order 3 --kernel simple --use_residual \
--runs 5 --epochs 500 --seed 123 --device 1

python main.py --dataset twitch --method ours3 --lr 1e-4 --weight_decay 0. --num_layers 1 --beta 0. \
--hidden_channels 64 --num_heads 2 --K_order 3 --kernel simple --use_residual \
--use_reg --reg_weight 1.0  --num_aug_branch 5 --modify_ratio 0.2 --rewiring_type replace \
--runs 5 --epochs 500 --seed 123 --device 1

python main.py --dataset twitch --method ours3 --lr 1e-4 --weight_decay 0. --num_layers 1 --beta 0.5 \
--hidden_channels 64 --num_heads 2 --K_order 3 --kernel simple --use_residual \
--use_reg --reg_weight 1.0  --num_aug_branch 5 --modify_ratio 0.2 --rewiring_type replace \
--runs 5 --epochs 500 --seed 123 --device 1



python main.py --dataset twitch --method ours3 --lr 1e-4 --weight_decay 0. --num_layers 1 --beta 0.5 \
--hidden_channels 64 --num_heads 2 --solver inverse --theta 1.0 --use_residual \
--runs 5 --epochs 500 --seed 123 --device 1

python main.py --dataset twitch --method ours3 --lr 1e-4 --weight_decay 0. --num_layers 1 --beta 0.2 \
--hidden_channels 64 --num_heads 2 --solver inverse --theta 1.0 --use_residual \
--runs 5 --epochs 500 --seed 123 --device 1


python main.py --dataset arxiv --method ours3 --lr 1e-3 --weight_decay 0. --num_layers 1 --beta 0.5 \
--hidden_channels 64 --num_heads 1 --solver inverse --theta 1.0 --use_residual --use_bn \
--runs 5 --epochs 500 --seed 123 --device 0

python main.py --dataset arxiv --method ours3 --lr 1e-3 --weight_decay 0. --num_layers 1 --beta 0.2 \
--hidden_channels 128 --num_heads 1 --solver inverse --theta 1.0 --use_residual --use_bn \
--runs 5 --epochs 500 --seed 123 --device 2




python main.py --dataset synthetic --method ours3 --lr 1e-2 --weight_decay 0. --num_layers 1 --beta 0. \
--hidden_channels 8 --num_heads 2 --solver series --K_order 8 --use_residual \
--runs 1 --epochs 500 --seed 123 --device 1

python main.py --dataset synthetic --method ours3 --lr 1e-2 --weight_decay 0. --num_layers 1 --beta 1.0 \
--hidden_channels 8 --num_heads 2 --solver series --K_order 8 --use_residual \
--runs 1 --epochs 500 --seed 123 --device 1

python main.py --dataset synthetic --method ours3 --lr 1e-2 --weight_decay 0. --num_layers 1 --beta 0.8 \
--hidden_channels 8 --num_heads 2 --solver series --K_order 8 --use_residual \
--runs 1 --epochs 500 --seed 123 --device 1


python main.py --dataset synthetic --method ours3 --lr 1e-2 --weight_decay 0. --num_layers 1 --beta 1.0 \
--hidden_channels 8 --num_heads 1 --solver inverse --theta 1.0 --use_residual \
--runs 1 --epochs 500 --seed 123 --device 1

python main.py --dataset synthetic --method ours3 --lr 1e-2 --weight_decay 0. --num_layers 1 --beta 0. \
--hidden_channels 8 --num_heads 1 --solver inverse --theta 1.0 --use_residual \
--runs 1 --epochs 500 --seed 123 --device 1

python main.py --dataset synthetic --method ours3 --lr 1e-2 --weight_decay 0. --num_layers 1 --beta 0.2 \
--hidden_channels 8 --num_heads 1 --solver inverse --theta 1.0 --use_residual \
--runs 1 --epochs 500 --seed 123 --device 1

python main.py --dataset synthetic --method ours3 --lr 1e-2 --weight_decay 0. --num_layers 1 --beta 1.0 \
--hidden_channels 8 --num_heads 1 --solver inverse --theta 1.0 --use_residual \
--use_reg --reg_weight 1.0  --num_aug_branch 5 --modify_ratio 0.2 --rewiring_type replace \
--runs 5 --epochs 500 --seed 123 --device 1

python main.py --dataset synthetic --method ours3 --lr 1e-2 --weight_decay 0. --num_layers 1 --beta 0.2 \
--hidden_channels 8 --num_heads 1 --solver inverse --theta 1.0 --use_residual \
--use_reg --reg_weight 1.0  --num_aug_branch 5 --modify_ratio 0.2 --rewiring_type replace \
--runs 5 --epochs 500 --seed 123 --device 1
