## GCN encoder

# ours

python main_full.py --dataset arxiv --method ours --lr 1e-2 --weight_decay 0. --num_layers 2 \
--hidden_channels 64 --num_heads 1 --kernel simple --use_residual --use_weight --use_bn \
--use_reg --reg_weight 1e4 --num_aug_branch 5 --modify_ratio 0.2 --rewiring_type replace \
--runs 5 --epochs 500 --seed 123 --device 1

python main_full.py --dataset arxiv --method ours --lr 1e-2 --weight_decay 0. --num_layers 2 \
--hidden_channels 64 --num_heads 1 --kernel simple --use_weight --use_bn --use_residual \
--runs 5 --epochs 500 --seed 123 --device 1


python main_full.py --dataset twitch --method ours --lr 1e-3 --weight_decay 0. --num_layers 2 \
--hidden_channels 64 --num_heads 1 --kernel simple --use_residual --use_weight --use_block \
--use_reg --reg_weight 1e5  --num_aug_branch 5 --modify_ratio 0.2 --rewiring_type replace \
--runs 5 --epochs 500 --seed 123 --device 1

python main_full.py --dataset twitch --method ours --lr 1e-3 --weight_decay 0. --num_layers 2 \
--hidden_channels 64 --num_heads 1 --kernel simple --use_residual --use_weight --use_block \
--runs 5 --epochs 500 --seed 123 --device 1

# baselines
python main_full.py --method erm --encoder gcn --dataset arxiv --lr 0.01 --num_layers 2 --hidden_channels 64 --weight_decay 0. --device 3
python main_full.py --method erm --encoder gcn --dataset twitch --lr 0.01 --num_layers 2 --hidden_channels 64 --weight_decay 0. --device 3

python main_full.py --method dropedge --encoder gcn --dataset arxiv --lr 0.01 --num_layers 2 --hidden_channels 64 --weight_decay 0. --modify_ratio 0.2 --device 3
python main_full.py --method dropedge --encoder gcn --dataset twitch --lr 0.01 --num_layers 2 --hidden_channels 64 --weight_decay 0. --modify_ratio 0.2 --device 3

python main_full.py --method reg --encoder gcn --dataset arxiv --lr 0.01 --num_layers 2 --hidden_channels 64 --weight_decay 0. \
 --reg_weight 1e4 --num_aug_branch 5 --modify_ratio 0.2 --rewiring_type replace --runs 5 --epochs 500 --seed 123 --device 3

python main_full.py --method reg --encoder gcn --dataset twitch --lr 0.01 --num_layers 2 --hidden_channels 64 --weight_decay 0. \
 --reg_weight 1e4 --num_aug_branch 5 --modify_ratio 0.2 --rewiring_type replace --runs 5 --epochs 500 --seed 123 --device 3

#
#python main_full.py --method irm --encoder gcn --dataset twitch --lr 0.01 --num_layers 2 --hidden_channels 64 --weight_decay 5e-4 --dropout 0.0 --device 3 --irm_lambda 0.8 --irm_penalty_anneal_iter 100
#python main_full.py --method mixup --encoder gcn --dataset twitch --lr 0.01 --num_layers 2 --hidden_channels 64 --weight_decay 5e-4 --dropout 0.0 --device 3 --mixup_prob 0.8 --mixup_alpha 0.5 --label_smooth_val 0.01
#python main_full.py --method coral --encoder gcn --dataset twitch --lr 0.01 --num_layers 2 --hidden_channels 64 --weight_decay 0 --dropout 0.2 --device 1 --coral_penalty_weight 0.005
#python main_full.py --method groupdro --encoder gcn --dataset twitch --lr 0.01 --num_layers 2 --hidden_channels 64 --weight_decay 5e-4 --dropout 0.0 --device 1 --groupdro_step_size 0.01
#python main_full.py --method dann --encoder gcn --dataset twitch --lr 0.01 --num_layers 2 --hidden_channels 64 --weight_decay 5e-4 --dropout 0.0 --device 1 --dann_alpha 0.001
#python main_full.py --method srgnn --encoder gcn --dataset twitch --lr 0.01 --num_layers 2 --hidden_channels 64 --weight_decay 0 --dropout 0.2 --device 1
#python main_full.py --method eerm --encoder gcn --dataset twitch --lr 0.01 --num_layers 2 --hidden_channels 64 --weight_decay 5e-4 --dropout 0.5 --beta 2 --K 3 --lr_a 0.005 --device 1