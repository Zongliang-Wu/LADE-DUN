
# LADE-DUN for CASSI

This repo is the implementation of paper "Latent Diffusion Prior Enhanced Deep Unfolding for Snapshot Spectral Compressive Imaging"

# Abstract

Snapshot compressive spectral imaging reconstruction aims to reconstruct three-dimensional spatial-spectral images from a single-shot two-dimensional compressed measurement. Existing state-of-the-art methods are mostly based on deep unfolding structures but have intrinsic performance bottlenecks: i) the ill-posed problem of dealing with heavily degraded measurement, and ii) the regression loss-based reconstruction models being prone to recover images with few details. In this paper, we introduce a generative model, namely the latent diffusion model (LDM), to generate degradation-free prior to enhance the regression-based deep unfolding method by a two-stage training procedure. Furthermore, we propose a Trident Transformer (TT), which extracts correlations among prior knowledge, spatial, and spectral features, to integrate knowledge priors in deep unfolding denoiser, and guide the reconstruction for compensating high-quality spectral signal details. To our knowledge, this is the first approach to integrate physics-driven deep unfolding with generative LDM in the context of CASSI reconstruction. Comparisons on synthetic and real-world datasets illustrate the superiority of our proposed method in both reconstruction quality and computational efficiency.


<!-- # Comparison with other Deep Unfolding Networks

<div align=center>
<img src="./figures/LADE_DUN_fig.png" width = "400" height = "300" alt="">
</div>

Comparison of PSNR-Params with previous HSI DUNs. The PSNR (in dB) is plotted on the vertical axis, while the number of parameters is represented on the horizontal axis. The proposed LADE-DUN outperforms the previous DUNs while requiring much fewer parameters. -->

# Architecture

<div align=center>
<img src="./figures/LADE_DUN_fig.png" >
</div>

# Results Visualization (Real Data)

<div align=center>
<img src="./figures/real_fig.png" >
</div>

# Usage 

## Prepare Dataset:
Follow the [RDLUF_MixS2](https://github.com/ShawnDong98/RDLUF_MixS2),
download cave_1024_28 ([Baidu Disk](https://pan.baidu.com/s/1X_uXxgyO-mslnCTn4ioyNQ), code: `fo0q` | [One Drive](https://bupteducn-my.sharepoint.com/:f:/g/personal/mengziyi_bupt_edu_cn/EmNAsycFKNNNgHfV9Kib4osB7OD4OSu-Gu6Qnyy5PweG0A?e=5NrM6S)), CAVE_512_28 ([Baidu Disk](https://pan.baidu.com/s/1ue26weBAbn61a7hyT9CDkg), code: `ixoe` | [One Drive](https://mailstsinghuaeducn-my.sharepoint.com/:f:/g/personal/lin-j21_mails_tsinghua_edu_cn/EjhS1U_F7I1PjjjtjKNtUF8BJdsqZ6BSMag_grUfzsTABA?e=sOpwm4)), KAIST_CVPR2021 ([Baidu Disk](https://pan.baidu.com/s/1LfPqGe0R_tuQjCXC_fALZA), code: `5mmn` | [One Drive](https://mailstsinghuaeducn-my.sharepoint.com/:f:/g/personal/lin-j21_mails_tsinghua_edu_cn/EkA4B4GU8AdDu0ZkKXdewPwBd64adYGsMPB8PNCuYnpGlA?e=VFb3xP)), TSA_simu_data ([Baidu Disk](https://pan.baidu.com/s/1LI9tMaSprtxT8PiAG1oETA), code: `efu8` | [One Drive](https://1drv.ms/u/s!Au_cHqZBKiu2gYFDwE-7z1fzeWCRDA?e=ofvwrD)), TSA_real_data ([Baidu Disk](https://pan.baidu.com/s/1RoOb1CKsUPFu0r01tRi5Bg), code: `eaqe` | [One Drive](https://1drv.ms/u/s!Au_cHqZBKiu2gYFTpCwLdTi_eSw6ww?e=uiEToT)), and then modify the data paths in the `option.py`.


## Pretrained weights and Results

Download pretrained weights and results at ([Onedrive](https://westlakeu-my.sharepoint.com/:f:/g/personal/wuzongliang_westlake_edu_cn/EikQ7Wr9ToNCp6YktuRzDwkBZnyvcB3Hb4meuhmH3YhKXg?e=k7W36q)).

## Environment
```
python==3.10
torch==2.0.1
scikit-image==0.21.0
scikit-learn==1.5.1
numpy==1.24.4
scipy==1.11.2
pyiqa==0.1.7
matplotlib==3.7.2
Pillow==10.0.0
lpips==0.1.4
```
## Simulation Experiement:

See the `readme.md` in the [./train_code_syn](./train_code_syn).

## Real Experiement:

See the `readme.md` in the [./train_code_real](./train_code_real).


## Acknowledgements

Our code is based on following codes, thanks for their generous open source:

- [https://github.com/ShawnDong98/RDLUF_MixS2](https://github.com/ShawnDong98/RDLUF_MixS2)
- [https://github.com/caiyuanhao1998/MST](https://github.com/caiyuanhao1998/MST)
- [https://github.com/mengziyi64/TSA-Net](https://github.com/mengziyi64/TSA-Net)
- [https://github.com/Zj-BinXia/DiffIR](https://github.com/Zj-BinXia/DiffIR)




## Citation

If this code helps you, please consider citing our works:

```shell
@article{wu2023latent,
  title={Latent diffusion prior enhanced deep unfolding for spectral image reconstruction},
  author={Wu, Zongliang and Lu, Ruiying and Fu, Ying and Yuan, Xin},
  journal={arXiv preprint arXiv:2311.14280},
  year={2023}
}
```
