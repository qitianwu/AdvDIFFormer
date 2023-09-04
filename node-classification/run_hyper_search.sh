datasets=('arxiv' 'twitch')
dev=0

#for a in 0.1 0.3 0.5 0.7
#do
#  for b in 1
#    do
#    for c in 1 2 4 8 16
#    do
#      for d in 1e4
#      do
#        for e in 0.2
#        do
#          python main.py --dataset arxiv --method ours3 --lr 1e-3 --weight_decay 0. --num_layers 1 --beta $a \
#--hidden_channels 128 --num_heads $b --K_order $c --kernel simple --use_residual --use_bn \
#--use_reg --reg_weight $d --num_aug_branch 5 --modify_ratio $e --rewiring_type replace \
#--runs 1 --epochs 500 --seed 123 --device 1 --save_result
#        done
#      done
#    done
#  done
#done

for a in 0.5
do
  for b in 1 2 4
    do
    for c in 8
    do
      for d in 0.1 0.5 1.0 5.0 1e1
      do
        for e in 0.1 0.2 0.3 0.4 0.5
        do
          for f in 'replace' 'delete'
          do
          python main.py --dataset arxiv --method ours3 --lr 1e-3 --weight_decay 0. --num_layers 1 --beta $a \
--hidden_channels 128 --num_heads $b --K_order $c --kernel simple --use_residual --use_bn \
--use_reg --reg_weight $d --num_aug_branch 5 --modify_ratio $e --rewiring_type $f \
--runs 1 --epochs 500 --seed 123 --device 1 --save_result
          done
        done
      done
    done
  done
done

#for a in 1e5 1e4 1e3 1e2 1e1
#do
#  for c in 0.3 0.2 0.1
#    do
#    for b in 1 2 3
#    do
#      for e in 1 2 4
#      do
#        for f in 1 2 3
#        do
#          for g in 32 64 128
#          do
#          python main.py --dataset arxiv --method ours2 --lr 1e-3 --weight_decay 0. --num_layers $b --K_order $f \
#--hidden_channels $g --num_heads $e --kernel simple --use_residual --use_bn \
#--use_reg --reg_weight $a --num_aug_branch 5 --modify_ratio $c --rewiring_type replace \
#--runs 1 --epochs 1500 --seed 123 --device $dev --save_result
#          done
#        done
#      done
#    done
#  done
#done


#
#for a in 1e5 1e4 1e3
#do
#  for b in 5 3
#  do
#    for c in 0.3 0.2 0.1
#    do
#      for e in 'replace'
#      do
#        for k in 1e-2
#        do
#          python main.py --dataset arxiv --method ours --encoder gat --lr $k --weight_decay 0. --num_layers 2 \
#--hidden_channels 64 --num_heads 1 --kernel simple --use_residual --use_weight --use_bn \
#--use_reg --reg_weight $a --num_aug_branch $b --modify_ratio $c --rewiring_type $e \
#--runs 5 --epochs 500 --seed 123 --device $dev --save_result
#        done
#      done
#    done
#  done
#done