 # Advective Diffusion Transformer

The official implementation for ICML2025 paper "Supercharging Graph Transformers with Advective Diffusion".

 ## Information Networks



 ## Dynamic Protein Interactions


 ## Molecular Mapping Generation

> python main.py --data_dir data --lr 5e-5 --dropout 0.1 --heads 2 --gnn ours-series --num_layer 5 --dim 256 --log_dir log --store_model --epoch 200 --device 0 --bs 256 --alpha 0.5 --K_order 4

## Run the codes

Please refer to the bash script `run.sh` in each folder for running the training and evaluation pipeline.

### Citation

If you find our code and model useful, please cite our work. Thank you!

```bibtex
      @inproceedings{
        wu2025advdifformer,
        title={Supercharging Graph Transformers with Advective Diffusion},
        author={Qitian Wu and Chenxiao Yang and Kaipeng Zeng and Michael Bronstein},
        booktitle={International Conference on Machine Learning (ICML)},
        year={2025}
        }
```

