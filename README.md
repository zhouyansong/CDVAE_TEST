# Long-Range CDVAE

Our study presents an improved deep generative model based on the
[CDVAE](https://arxiv.org/abs/2110.06197) and [EwaldMP](https://arxiv.org/abs/2303.04791), LRCDVAE, which enables the inverse design of van der Waals (vdW) materials. 

## Environment
We recommend using Anaconda to manage Python environments. First, create and activate a new Python environment:
```
conda create --name concdvae python=3.11
conda activate concdvae
```

Then, use `requirements.txt` to install the Python packages.
```
pip install -r requirements.txt
pip install -e .
```
## Setting up environment variables
Modify the following environment variables in .env.
- `PROJECT_ROOT`: path to the folder that contains this repo
- `HYDRA_JOBS`: path to a folder to store hydra outputs
- `WABDB`: path to a folder to store wabdb outputs

## Datasets
You can find a small sample of the dataset in data/, including the data used for LRCDVAE training. The complete data are available from the corresponding author upon reasonable request.

## Training and evaluation

training command:
```
python lrcdvae/run.py data=2d expname=2d 
```
To generate materials, run the following command:
```
python scripts/evaluate.py --model_path MODEL_PATH --tasks recon gen opt
```
compute reconstruction & generation metrics (only on random gen data):
```
python scripts/compute_metrics.py --root_path ROOT_PATH --tasks recon gen opt
```

# References

CDVAE
```
@article{xie2021crystal,
title={Crystal Diffusion Variational Autoencoder for Periodic Material Generation},
author={Xie, Tian and Fu, Xiang and Ganea, Octavian-Eugen and Barzilay, Regina and Jaakkola, Tommi},
journal={arXiv preprint arXiv:2110.06197},
year={2021}
}
```
EwaldMP
```
@inproceedings{kosmala_ewaldbased_2023,
title = {Ewald-based Long-Range Message Passing for Molecular Graphs},
author = {Kosmala, Arthur and Gasteiger, Johannes and Gao, Nicholas and G{\"u}nnemann, Stephan},
booktitle={International Conference on Machine Learning (ICML)},
year = {2023} 
}
```
MatterSim
```
@article{yang2024mattersim,
title={MatterSim: A Deep Learning Atomistic Model Across Elements, Temperatures and Pressures},
author={Han Yang and Chenxi Hu and Yichi Zhou and Xixian Liu and Yu Shi and Jielan Li and Guanzhi Li and Zekun Chen and Shuizhou Chen and Claudio Zeni and Matthew Horton and Robert Pinsler and Andrew Fowler and Daniel Zügner and Tian Xie and Jake Smith and Lixin Sun and Qian Wang and Lingyu Kong and Chang Liu and Hongxia Hao and Ziheng Lu},
year={2024},
eprint={2405.04967},
archivePrefix={arXiv},
primaryClass={cond-mat.mtrl-sci},
url={https://arxiv.org/abs/2405.04967},
journal={arXiv preprint arXiv:2405.04967}
}
```
ComENet in DIG

```
@article{JMLR:v22:21-0343,
author  = {Meng Liu and Youzhi Luo and Limei Wang and Yaochen Xie and Hao Yuan and Shurui Gui and Haiyang Yu and Zhao Xu and Jingtun Zhang and Yi Liu and Keqiang Yan and Haoran Liu and Cong Fu and Bora M Oztekin and Xuan Zhang and Shuiwang Ji},
title   = {{DIG}: A Turnkey Library for Diving into Graph Deep Learning Research},
journal = {Journal of Machine Learning Research},
year    = {2021},
volume  = {22},
number  = {240},
pages   = {1-9},
url     = {http://jmlr.org/papers/v22/21-0343.html}
}
```