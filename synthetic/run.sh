# homophily shift
python main.py --dataset synthetic --syn_type homophily --method advdifformer --lr 1e-2 --weight_decay 0. --num_layers 1 --beta 0.2 \
--hidden_channels 8 --num_heads 1 --solver inverse --theta 1.0 \
--runs 5 --epochs 500 --seed 123 --device 2

python main.py --dataset synthetic --syn_type homophily --method advdifformer --lr 1e-2 --weight_decay 0. --num_layers 1 --beta 0.2 \
--hidden_channels 8 --num_heads 1 --solver series --K_order 1 \
--runs 5 --epochs 500 --seed 123 --device 2

# density shift
python main.py --dataset synthetic --syn_type density --method advdifformer --lr 1e-2 --weight_decay 0. --num_layers 1 --beta 0.2 \
--hidden_channels 8 --num_heads 1 --solver inverse --theta 1.0 \
--runs 5 --epochs 500 --seed 123 --device 2

python main.py --dataset synthetic --syn_type density --method advdifformer --lr 1e-2 --weight_decay 0. --num_layers 1 --beta 0.2 \
--hidden_channels 8 --num_heads 1 --solver series --K_order 1 \
--runs 5 --epochs 500 --seed 123 --device 2

# block shift
python main.py --dataset synthetic --syn_type block --method advdifformer --lr 1e-2 --weight_decay 0. --num_layers 1 --beta 0.2 \
--hidden_channels 8 --num_heads 1 --solver inverse --theta 1.0 \
--runs 5 --epochs 500 --seed 123 --device 2

python main.py --dataset synthetic --syn_type block --method advdifformer --lr 1e-2 --weight_decay 0. --num_layers 1 --beta 0.2 \
--hidden_channels 8 --num_heads 1 --solver series --K_order 1 \
--runs 5 --epochs 500 --seed 123 --device 2



