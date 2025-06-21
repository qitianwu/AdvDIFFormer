 # Advective Diffusion Transformer (AdvDIFFormer)

The official implementation for ICML2025 paper "Supercharging Graph Transformers with Advective Diffusion" ([Paper](https://arxiv.org/pdf/2310.06417)).

AdvDIFFormer is a graph Transformer model derived from the closed-form solution of advective diffusion equation models that are provably resilient to distribution shifts of graph topologies. The model has two implementation versions **AdvDIFFormer-i** and **AdvDIFFormer-s** (with linear complexity w.r.t. node numbers).

AdvDIFFormer is built on our early works about scalable graph Transformers:

- [NodeFormer](https://github.com/qitianwu/NodeFormer): a scalable Transformer with linear complexity
- [DIFFormer](https://github.com/qitianwu/DIFFormer): a principled Transformer derived from diffusion equations with energy constraint
- [SGFormer](https://github.com/qitianwu/SGFormer): a simplified Transformer with single-layer efficient attention and approximation-free linear complexity


<img width="654" alt="image" src="https://github.com/user-attachments/assets/0ccba7e5-0eff-4185-a6fa-854fab464537" />

## Results

The model is applied to information networks, dynamic protein interactions and molecular mapping operator generation.

<img width="996" alt="image" src="https://github.com/user-attachments/assets/255e8bf2-8511-40a9-a5b2-45116ff44cb7" />

## Run the codes

- Information Networks: `node-classification`
- Dynamic Protein Interactions: `dppin`
- Molecular Mapping Generation: `ham`

Please refer to the bash script `run.sh` in each folder for running the training and evaluation pipeline.

## Citation

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

