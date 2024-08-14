# advdifformer
python main.py --dataset arxiv --method advdifformer --lr 1e-3 --weight_decay 0. --num_layers 1 --beta 0.8 \
--hidden_channels 128 --num_heads 2 --K_order 1 --theta 1.0 --kernel simple --use_residual --use_bn \
--runs 5 --epochs 500 --seed 123 --device 2

python main.py --dataset twitch --method advdifformer --lr 1e-4 --weight_decay 0. --num_layers 1 --beta 1.0 \
--hidden_channels 64 --num_heads 2 --solver series --K_order 1 --use_residual \
--runs 5 --epochs 500 --seed 123 --device 1

# difformer
python main.py --dataset arxiv --method difformer --lr 0.01 --num_layers 3 \
--hidden_channels 128 --weight_decay 0. --use_weight --use_residual --use_bn \
--runs 5 --epochs 500 --seed 123 --device 1

python main.py --dataset twitch --method difformer --lr 0.01 --num_layers 2 \
--hidden_channels 64 --weight_decay 0. --use_weight --use_residual \
--runs 5 --epochs 500 --seed 123 --device 1
