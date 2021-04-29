import os, sys, time, pdb
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

path = "/workspace/gparmar/github_gaparmar/DC-VAE/notebooks"
if path not in sys.path:
    sys.path.append(path)

from models.models_32 import *
from utils.misc import *
from utils.eval import *
from utils.finetuning import *
from utils.fid_score import calculate_fid_given_paths


def reparameterize(mu, logvar):
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    return mu + eps*std

def f_recon(img, netE, netD, zdim, C):
    bs, imsize = img.shape[0], img.shape[2]
    mu_logvar = netE(img).view(bs,-1)
    mu = mu_logvar[:,0:zdim]
    logvar = mu_logvar[:,zdim:]
    z = reparameterize(mu, logvar)
    if C["use_mu"]: recon = netD(z).view(bs,3,imsize,imsize)
    else: recon = netD(mu).view(bs,3,imsize,imsize)
    
    return recon, mu, logvar

def compute_fid_recon(netE, netD, dl, C, zdim=128, buf_dir=None):
    if buf_dir is None:
        buf_dir = f"tmp/{time.time()}"
    if not os.path.exists(buf_dir):
        os.makedirs(buf_dir)
    fid_stat = "/mnt/cube/gparmar/fid_stats_cifar10_train.npz"
    num_eval_imgs = 50000
    if C["eval_mode"]: 
        netE.eval()
        netD.eval()
    else:
        netE.train()
        netD.train()
    total = 0
    while total <= num_eval_imgs:
        for iter_idx, (imgs, _) in enumerate(tqdm(dl)):
            with torch.no_grad():
                imgs = imgs.cuda()
                curr_bs = imgs.shape[0]
                rec, _, _ = f_recon(imgs, netE, netD, zdim, C)
                out_imgs = rec.mul_(127.5).add_(127.5).clamp(0.0,255.0)
                out_imgs = out_imgs.permute(0,2,3,1).to("cpu",torch.uint8).numpy()
                for img_idx, img in enumerate(out_imgs):
                    file_name = os.path.join(buf_dir, f"{total}_{iter_idx}_b{img_idx}.png")
                    imsave(file_name, img)
                total += curr_bs
                if total >= num_eval_imgs:
                    break
    fid_score = calculate_fid_given_paths([buf_dir, fid_stat], C["fid_bs"])
    os.system(f"rm -r {buf_dir}")
    return fid_score


"""
Meta parameters for FID eval
"""
C = {}
C["eval_mode"] = False
C["use_avg_net"] = True
C["use_mu"] = False
C["generator_bs"] = 16
C["fid_bs"] = 32


ds = torchvision.datasets.CIFAR10("./dataset/", train=False, download=False,
                transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize(
                                    (0.5, 0.5, 0.5), 
                                    (0.5, 0.5, 0.5)),
                           ]))


name = "local_patch_b1_finallocal_h2_kld1e-06_k1024_lamLocal0.125"
load_epoch=680
laod_dir = os.path.join("saved_models", f"cifar10-{name}")
encoder, decoder, decoder_avg, dual_encoder, dual_encoder_M = load_all_models(laod_dir, epoch=load_epoch)

L_gen_bs = [8, 16, 32, 64, 128, 256]
for x in L_gen_bs:
    C["fid_bs"] = x
    test_loader = torch.utils.data.DataLoader(ds, batch_size=C["generator_bs"], 
                        shuffle=True, pin_memory=True, drop_last=True,
                        num_workers=4)
    
    if C["use_avg_net"]:
        fid = compute_fid_recon(encoder, decoder_avg, test_loader, C)
    else:
        compute_fid_recon(encoder, decoder, test_loader, C)
    print(C)
    print(fid,"\n\n")