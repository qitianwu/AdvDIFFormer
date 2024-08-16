现在的ours3是我们模型的最终实现，对应使用Chebyshev近似求解PDE方程。
不同的近似方法具体对应了两种实现，分别是solver == 'series'和'inverse'。

模型层面的参数有
num_layers：控制带参的模型层数（优先考虑固定为1，此时PDE给出的是完整的diffusion轨迹）
beta：控制gcn/attn的权重
num_heads：控制多头数目（每个head的attention query/key参数和feature transformation参数不同）

1. series这种情况，多出来了参数K_order控制prop的层数（即几何级数近似的阶数）。这种模型牺牲了精度，但复杂度为线性NK

> python main.py --dataset twitch --method ours3 --lr 1e-4 --weight_decay 0. --num_layers 1 --beta 0.5 \
--hidden_channels 64 --num_heads 2 --solver series --K_order 3 --use_residual \
--runs 5 --epochs 500 --seed 123 --device 1


2. inverse这种情况，多出来了参数\theta控制单位阵的权重（取值可以在0-1之间，但设为0时可能会有数值不稳定）。这种模型
因为需要计算得到attn矩阵，以及对Laplacian矩阵求逆，复杂度至少为N^2，但在小图上（节点数<1000）效率还是不错的

> python main.py --dataset twitch --method ours3 --lr 1e-4 --weight_decay 0. --num_layers 1 --beta 0.5 \
--hidden_channels 64 --num_heads 2 --solver inverse --theta 1.0 --use_residual \
--runs 5 --epochs 500 --seed 123 --device 1

3. ham 数据集只有 series 的模型，没有inverse的模型

>python main.py --data_dir data --lr 5e-5 --dropout 0.1 --heads 2 --gnn ours-series --num_layer 5 --dim 256 --log_dir log --store_model --epoch 200 --device 0 --bs 256 --alpha 0.5 --K_order 4 
>
>