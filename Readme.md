 # Advective Diffusion Transformer

 ## Information Networks & Synthetic Data

 `node-classification`
 `synthetic`

 ## Dynamic Protein Interactions

 ## Molecular Mapping Generation

3. ham 数据集只有 series 的模型，没有inverse的模型

> python main.py --data_dir data --lr 5e-5 --dropout 0.1 --heads 2 --gnn ours-series --num_layer 5 --dim 256 --log_dir log --store_model --epoch 200 --device 0 --bs 256 --alpha 0.5 --K_order 4 