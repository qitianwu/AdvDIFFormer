 # Advective Diffusion Transformer

The official implementation for ICML2025 paper "Supercharging Graph Transformers with Advective Diffusion".

<img width="654" alt="image" src="https://github.com/user-attachments/assets/0ccba7e5-0eff-4185-a6fa-854fab464537" />

## Results

The model is applied to information networks, dynamic protein interactions and molecular mapping operator generation.

<img width="996" alt="image" src="https://github.com/user-attachments/assets/255e8bf2-8511-40a9-a5b2-45116ff44cb7" />

## Run the codes

- Information Networks: `node-classification` and `synthetic`
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

