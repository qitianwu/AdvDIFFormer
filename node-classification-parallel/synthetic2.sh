python main.py --dataset synthetic --syn_type both --method ours3 --lr 1e-2 --weight_decay 0. --num_layers 1 --beta 0. \
--hidden_channels 8 --num_heads 1 --solver series --K_order 8 --use_residual \
--runs 5 --epochs 500 --seed 123 --device 1 --save_result

python main.py --dataset synthetic --syn_type both --method ours3 --lr 1e-2 --weight_decay 0. --num_layers 1 --beta 0.1 \
--hidden_channels 8 --num_heads 1 --solver series --K_order 8 --use_residual \
--runs 5 --epochs 500 --seed 123 --device 1 --save_result

python main.py --dataset synthetic --syn_type both --method ours3 --lr 1e-2 --weight_decay 0. --num_layers 1 --beta 0.2 \
--hidden_channels 8 --num_heads 1 --solver series --K_order 8 --use_residual \
--runs 5 --epochs 500 --seed 123 --device 1 --save_result

python main.py --dataset synthetic --syn_type both --method ours3 --lr 1e-2 --weight_decay 0. --num_layers 1 --beta 0.5 \
--hidden_channels 8 --num_heads 1 --solver series --K_order 8 --use_residual \
--runs 5 --epochs 500 --seed 123 --device 1 --save_result

python main.py --dataset synthetic --syn_type both --method ours3 --lr 1e-2 --weight_decay 0. --num_layers 1 --beta 1.0 \
--hidden_channels 8 --num_heads 1 --solver series --K_order 8 --use_residual \
--runs 5 --epochs 500 --seed 123 --device 1 --save_result



python main.py --dataset synthetic --syn_type both --method ours3 --lr 1e-2 --weight_decay 0. --num_layers 1 --beta 0. \
--hidden_channels 8 --num_heads 1 --solver inverse --theta 1.0 --use_residual \
--runs 5 --epochs 500 --seed 123 --device 1 --save_result

python main.py --dataset synthetic --syn_type both --method ours3 --lr 1e-2 --weight_decay 0. --num_layers 1 --beta 0.1 \
--hidden_channels 8 --num_heads 1 --solver inverse --theta 1.0 --use_residual \
--runs 5 --epochs 500 --seed 123 --device 1 --save_result

python main.py --dataset synthetic --syn_type both --method ours3 --lr 1e-2 --weight_decay 0. --num_layers 1 --beta 0.2 \
--hidden_channels 8 --num_heads 1 --solver inverse --theta 1.0 --use_residual \
--runs 5 --epochs 500 --seed 123 --device 1 --save_result

python main.py --dataset synthetic --syn_type both --method ours3 --lr 1e-2 --weight_decay 0. --num_layers 1 --beta 0.5 \
--hidden_channels 8 --num_heads 1 --solver inverse --theta 1.0 --use_residual \
--runs 5 --epochs 500 --seed 123 --device 1 --save_result

python main.py --dataset synthetic --syn_type both --method ours3 --lr 1e-2 --weight_decay 0. --num_layers 1 --beta 1.0 \
--hidden_channels 8 --num_heads 1 --solver inverse --theta 1.0 --use_residual \
--runs 5 --epochs 500 --seed 123 --device 1 --save_result
