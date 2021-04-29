import time
import torch
import numpy as np
from tqdm import tqdm
from imageio import imsave
from .fid_score import calculate_fid_given_paths
# from .inception_score import inception_score
from .misc import *

def compute_fid_recon(netE, netD, dl, zdim, eval_bs=128, buf_dir=None):
    if buf_dir is None:
        buf_dir = f"tmp/{time.time()}"
    if not os.path.exists(buf_dir):
        os.makedirs(buf_dir)
    fid_stat = "./utils/fid_stats_cifar10_train.npz"
    num_eval_imgs = 50000
    netE.eval()
    netD.eval()
    total = 0
    while total <= num_eval_imgs:
        for iter_idx, (imgs, _) in enumerate(tqdm(dl)):
            curr_bs = imgs.shape[0]
            rec, _, _ = f_recon(imgs, netE, netD, zdim, mode="train", clipping=False)
            out_imgs = rec.mul_(127.5).add_(127.5).clamp(0.0,255.0)
            out_imgs = out_imgs.permute(0,2,3,1).to("cpu",torch.uint8).numpy()
            for img_idx, img in enumerate(out_imgs):
                file_name = os.path.join(buf_dir, f"{total}_{iter_idx}_b{img_idx}.png")
                imsave(file_name, img)
            total += curr_bs
            if total >= num_eval_imgs:
                break
    fid_score = calculate_fid_given_paths([buf_dir, fid_stat], eval_bs)
    netE.train()
    netD.train()
    os.system(f"rm -r {buf_dir}")
    return fid_score

def compute_fid_sample(netD, zdim, eval_bs=128, buf_dir=None):
    if buf_dir is None:
        buf_dir = f"tmp/{time.time()}"
    if not os.path.exists(buf_dir):
        os.makedirs(buf_dir)
    fid_stat = "utils/fid_stats_cifar10_train.npz"
    num_eval_imgs = 50000
    netD.eval()
    total = 0
    while total <= num_eval_imgs:
        curr_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (eval_bs, zdim)))
        curr_z = torch.clamp(curr_z, min=-1.0, max=1.0)
        imgs = netD(curr_z)
        out_imgs = imgs.mul_(127.5).add_(127.5).clamp(0.0,255.0)
        out_imgs = out_imgs.permute(0,2,3,1).to("cpu",torch.uint8).numpy()
        for img_idx, img in enumerate(out_imgs):
            file_name = os.path.join(buf_dir, f"iter_{total}_{img_idx}.png")
            imsave(file_name, img)
        total += eval_bs
    
    fid_score = calculate_fid_given_paths([buf_dir, fid_stat], eval_bs)
    netD.train()
    os.system(f"rm -r {buf_dir}")
    return fid_score