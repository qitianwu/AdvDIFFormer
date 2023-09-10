python main.py --method erm --encoder sgc --dataset arxiv --lr 0.01 --num_layers 3 --hidden_channels 128 --weight_decay 0. --epochs 500 --device 0 --save_result
python main.py --method erm --encoder gcn --dataset arxiv --lr 0.01 --num_layers 3 --hidden_channels 128 --weight_decay 0. --epochs 500 --device 0 --save_result
python main.py --method erm --encoder gat --dataset arxiv --lr 0.01 --num_layers 3 --hidden_channels 128 --weight_decay 0. --epochs 500 --device 0 --save_result
python main.py --method erm --encoder difformer --dataset arxiv --lr 0.01 --num_layers 3 --hidden_channels 64 --weight_decay 0. --use_weight --use_residual --use_bn --epochs 500 --device 0 --save_result

python main.py --dataset arxiv --method ours3 --lr 1e-3 --weight_decay 0. --num_layers 1 --beta 0. \
--hidden_channels 128 --num_heads 2 --K_order 1 --theta 1.0 --kernel simple --use_residual --use_bn \
--runs 5 --epochs 500 --seed 123 --device 2

python main.py --method erm --encoder sgc --dataset twitch --lr 0.01 --num_layers 2 --hidden_channels 64 --weight_decay 0. --epochs 500 --device 0 --save_result
python main.py --method erm --encoder gcn --dataset twitch --lr 0.01 --num_layers 2 --hidden_channels 64 --weight_decay 0. --epochs 500 --device 0 --save_result
python main.py --method erm --encoder gat --dataset twitch --lr 0.01 --num_layers 2 --hidden_channels 64 --weight_decay 0. --epochs 500 --device 0 --save_result
python main.py --method erm --encoder difformer --dataset twitch --lr 0.01 --num_layers 2 --hidden_channels 64 --weight_decay 0. --use_weight --use_residual --epochs 500 --device 0 --save_result

python main.py --dataset twitch --method ours3 --lr 1e-4 --weight_decay 0. --num_layers 1 --beta 0. \
--hidden_channels 64 --num_heads 2 --solver series --K_order 1 --use_residual \
--runs 5 --epochs 500 --seed 123 --device 1