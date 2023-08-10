
# baseline
python baseline_ogb.py --dataset ogbg-molbace --num_layer 3 --emb_dim 256 --gnn gcn --device 1
python baseline_ogb.py --dataset ogbg-molsider --num_layer 3 --emb_dim 256 --gnn gcn --device 1

python baseline_ogb.py --dataset ogbg-molbace --num_layer 3 --emb_dim 256 --gnn gcn-virtual --device 1
python baseline_ogb.py --dataset ogbg-molsider --num_layer 3 --emb_dim 256 --gnn gcn-virtual --device 1



python main.py --dataset ogbg-molbace --method ours --learning_rate 0.001 --num_layer 3 --emb_dim 64 --dropout 0. --num_heads 1 --kernel simple --use_bn --use_weight --use_block --batch_size 32 --device 3
#Best validation score: 0.7080586080586081
#Test score: 0.8433315945053035

python main.py --dataset ogbg-molbace --method ours --learning_rate 0.0001 --num_layer 3 --emb_dim 256 --num_heads 1 --kernel simple --use_bn --use_weight --use_block --batch_size 32  --use_reg --reg_weight 1e3 --num_aug_branch 3 --modify_ratio 0.2 --device 1
#Best validation score: 0.7238095238095238
#Test score: 0.8523735002608241
Best validation score: 0.7333333333333334
Test score: 0.8471570161711006
Best validation score: 0.7311355311355311
Test score: 0.8297687358720223
Best validation score: 0.7318681318681319
Test score: 0.8612415232133542
python main.py --dataset ogbg-molbace --method ours --learning_rate 1e-4 --num_layer 3 --emb_dim 256 --num_heads 1 --kernel simple --use_bn --use_weight --use_block --batch_size 32  --use_reg --reg_weight 1e3 --num_aug_branch 3 --modify_ratio 0.2 --device 1


python main.py --dataset ogbg-molsider --method ours --learning_rate 0.001 --num_layer 3 --emb_dim 64 --num_heads 1 --kernel simple --use_bn --use_weight --use_block --batch_size 32  --use_reg --reg_weight 1.0 --num_aug_branch 3 --modify_ratio 0.2 --device 1
python main.py --dataset ogbg-molsider --method ours --learning_rate 0.001 --num_layer 3 --emb_dim 256 --num_heads 1 --kernel simple --use_bn --use_weight --use_block --batch_size 32  --use_reg --reg_weight 1e3 --num_aug_branch 3 --modify_ratio 0.2 --device 1
#Best validation score: 0.635600055031193
#Test score: 0.6162732647464053
python main.py --dataset ogbg-molsider --method ours --learning_rate 1e-4 --num_layer 3 --emb_dim 256 --num_heads 1 --kernel simple --use_bn --use_weight --use_block --batch_size 32  --use_reg --reg_weight 1e3 --num_aug_branch 3 --modify_ratio 0.2 --device 1
Best validation score: 0.6365459817962642
Test score: 0.611617679895502
Best validation score: 0.6285231976646999
Test score: 0.6140300304893389
