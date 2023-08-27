datasets=('arxiv' 'twitch')
dev=0

for d in ${datasets[@]}
  do
  for a in 1e5 1e4 1e3
  do
    for b in 5 3
    do
      for c in 0.3 0.2 0.1
      do
        for e in 'replace'
        do
          for k in 1e-3
          do
          python main.py --dataset twitch --method ours --lr $k --weight_decay 0. --num_layers 2 \
--hidden_channels 64 --num_heads 1 --kernel simple --use_residual --use_weight --use_block \
--use_reg --reg_weight $a --num_aug_branch $b --modify_ratio $c --rewiring_type $e \
--runs 5 --epochs 500 --seed 123 --device $dev --save_result
          done
        done
      done
    done
  done
done
