datasets=('arxiv' 'twitch')
dev=0

for a in 0.2 0.5 0.8 1.0
do
  for b in 1 2 4
    do
    for c in 1 2 4
    do
      for d in 1
      do
        for e in 1e-3
        do
          python main.py --dataset arxiv --method advdifformer --lr $e --num_layers $d --beta $a \
--hidden_channels 128 --num_heads $b --solver series --K_order $c --use_residual --use_bn \
--runs 1 --epochs 500 --seed 123 --device 1 --save_result
        done
      done
    done
  done
done

for a in 0.2 0.5 0.8 1.0
do
  for b in 1 2 4
    do
    for c in 1 2 4
    do
      for d in 1
      do
        for e in 1e-4
        do
          python main.py --dataset twitch --method ours3 --lr $e --num_layers $d --beta $a \
--hidden_channels 64 --num_heads $b --solver series --K_order $c --theta 1.0 --use_residual \
--runs 1 --epochs 500 --seed 123 --device 2 --save_result
        done
      done
    done
  done
done