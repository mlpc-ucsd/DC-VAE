# Dual Contradistinctive Generative AutoEncoder (CVPR 2021)

[**Project**](http://dcvae.gauravparmar.com/) | 
[**Paper**](https://arxiv.org/abs/2011.10063) | 
[**Usage**](#usage) |
[**Citation**](#citation) 

<p align="center">
  <img src="https://gauravparmar.com/projects/dcvae/resources/dcvae_results.png"  width="800" />
</p>

A generative autoencoder model with dual contradistinctive losses to improve generative autoencoder that performs simultaneous inference (reconstruction) and synthesis (sampling). Our model, named dual contradistinctive generative autoencoder (DC-VAE), integrates an instance-level discriminative loss (maintaining the instance-level fidelity for the reconstruction/synthesis) with a set-level adversarial loss (encouraging the set-level fidelity for there construction/synthesis), both being contradistinctive. 

---

## Usage

- Clone the repository
  ```
  git clone https://github.com/mlpc-ucsd/DC-VAE
  cd DC-VAE
  ```

- Setup the conda environment 
  ```
  conda env create -f dcvae_env.yml
  conda activate dcvae_env
  ```

- Train the network on CIFAR-10
  ```
  python train.py
  ```

---

## Citation
If you find this project useful for your research, please cite the following work.

```
@InProceedings{Parmar_2021_CVPR,
    author    = {Parmar, Gaurav and Li, Dacheng and Lee, Kwonjoon and Tu, Zhuowen},
    title     = {Dual Contradistinctive Generative Autoencoder},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {823-832}
}
```

---

## Credits
We found the following libraries helpful in our research. 

 - [FID](https://github.com/mseitzer/pytorch-fid/) - computing the FID score
 - [IS](https://github.com/openai/improved-gan/tree/master/inception_score) - computing the Inception Score. 
 - [AutoGAN](https://github.com/TAMU-VITA/AutoGAN) - model architecture for the low resolution experiments experiments
 - [ProGAN](https://github.com/rosinality/progressive-gan-pytorch) - model architecture for the high resolution experiments.

---

## Acknowledgements
This work is funded by NSF IIS- 1717431 and NSF IIS-1618477. We thank Qualcomm Inc. for an award support. The work was performed when G. Parmar and D. Li were with UC San Diego.

---
