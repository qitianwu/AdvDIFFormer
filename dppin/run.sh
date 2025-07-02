# Node-regression
python main.py --method advdifformer --task node-reg --lr 1e-3 --weight_decay 0. --num_layers 4 --beta 1.0 --hidden_channels 64 --num_heads 2 --solver series --K_order 1 --use_residual --runs 5 --epochs 1000 --seed 123 --device 2 --save_result

python main.py --method advdifformer --task node-reg --lr 1e-2 --weight_decay 0. --num_layers 1 --num_heads 3 --beta 1.0 --theta 1.6 --hidden_channels 128 --solver inverse --K_order 1 --use_residual --runs 3 --epochs 1000 --seed 123 --device 0 --save_result

# Edge-regression
python main.py --method advdifformer --task edge-reg --lr 1e-3 --weight_decay 0. --num_layers 1 --dropout 0.6 --beta 1.0 --hidden_channels 64 --num_heads 2 --solver series --K_order 1 --use_residual --runs 3 --epochs 1000 --seed 123 --device 2 --save_result

python main.py --method advdifformer --task edge-reg --lr 1e-2 --weight_decay 0. --num_layers 1 --beta 1.0 --theta 1.2 --hidden_channels 32 --num_heads 2 --solver inverse --K_order 1 --use_residual --runs 3 --epochs 1000 --seed 123 --device 0 --save_result
