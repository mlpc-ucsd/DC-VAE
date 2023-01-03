import os, sys, time, pdb
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from models.models_32 import *
from utils.misc import *
from utils.eval import *
import wandb
wandb.init(project="DC-VAE-replication")

#############################
# Hyperparameters
#############################
seed               = 123
lr                 = 0.0002
beta1              = 0.0
beta2              = 0.9
num_workers        = 2
data_path          = "dataset"

dis_batch_size     = 64
gen_batch_size     = 128
max_epoch          = 800
lambda_kld         = 1e-6
latent_dim         = 256
cont_dim           = 16
cont_k             = 8192
cont_temp          = 0.07
wandb.config.latent_dim = latent_dim
wandb.config.beta1 = beta1
wandb.config.beta2 = beta2
wandb.config.lr = lr

# multi-scale contrastive setting
layers             = ["b1", "final"]

name =("").join(layers)
log_fname = f"logs/cifar10-basic-l2-{name}"
fid_fname = f"logs/FID_cifar10-basic-l2-{name}"
viz_dir = f"viz/cifar10-256-basic{name}"
models_dir = f"saved_models/cifar10-256-basic-l2{name}"
wandb.config.model_dir = models_dir

if not os.path.exists("logs"):
    os.makedirs("logs")
if not os.path.exists(viz_dir):
    os.makedirs(viz_dir)
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
lambda_cont = 1.0/len(layers)
fix_seed(random_seed=seed)

#############################
# Make and initialize the Networks
#############################
encoder = torch.nn.DataParallel(Encoder(latent_dim)).cuda()
decoder = torch.nn.DataParallel(Decoder(latent_dim)).cuda()

encoder.apply(weights_init)
decoder.apply(weights_init)


gen_avg_param = copy_params(decoder)
d_queue, d_queue_ptr = {}, {}
for layer in layers:
    d_queue[layer] = torch.randn(cont_dim, cont_k).cuda()
    d_queue[layer] = F.normalize(d_queue[layer], dim=0)
    d_queue_ptr[layer] = torch.zeros(1, dtype=torch.long)


#############################
# Make the optimizers
#############################
opt_encoder = torch.optim.Adam(filter(lambda p: p.requires_grad, 
                                        encoder.parameters()),
                                lr, (beta1, beta2))
opt_decoder = torch.optim.Adam(filter(lambda p: p.requires_grad, 
                                        decoder.parameters()),
                                lr, (beta1, beta2))

#############################
# Make the dataloaders
#############################
ds = torchvision.datasets.CIFAR10(data_path, train=True, download=True,
                transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize(
                                    (0.5, 0.5, 0.5), 
                                    (0.5, 0.5, 0.5)),
                           ]))
train_loader = torch.utils.data.DataLoader(ds, batch_size=dis_batch_size, 
                        shuffle=True, pin_memory=True, drop_last=True,
                        num_workers=num_workers)
ds = torchvision.datasets.CIFAR10(data_path, train=False, download=False,
                transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize(
                                    (0.5, 0.5, 0.5), 
                                    (0.5, 0.5, 0.5)),
                           ]))
test_loader = torch.utils.data.DataLoader(ds, batch_size=dis_batch_size, 
                        shuffle=True, pin_memory=True, drop_last=True,
                        num_workers=num_workers)

global_steps = 0
# train loop
for epoch in tqdm(range(max_epoch), desc='total progress'):
    encoder.train()
    decoder.train()
    #dual_encoder.train()
    #autoencoder.train()
    for iter_idx, (imgs, _) in enumerate(tqdm(train_loader)):
        curr_bs = imgs.shape[0]
        curr_log = f"{epoch}:{iter_idx}\t"
        real_imgs = imgs.type(torch.cuda.FloatTensor)
        z = torch.cuda.FloatTensor(np.random.normal(0, 1, (imgs.shape[0], latent_dim)))

        # encoder decoder training
        opt_decoder.zero_grad()
        opt_encoder.zero_grad()
        bs, imsize = imgs.shape[0], imgs.shape[2]
        mu_logvar = encoder(imgs).view(bs, -1)
        mu = mu_logvar[:,0:latent_dim]
        logvar = mu_logvar[:,latent_dim:]
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        z0 = mu + eps*std
        z0 = torch.clamp(z0, min=-1.0, max=1.0)
        x_recon = decoder(z0).view(bs,3,imsize,imsize)
        recon_loss = torch.mean((x_recon - real_imgs)**2)        
        #recon_loss = torch.mean(torch.abs(x_recon - real_imgs))
        wandb.log({"recon_loss": recon_loss})
        recon_loss.backward()
        opt_decoder.step()
        opt_encoder.step()


    # no need to modify from here on
        if global_steps%250 == 0:
            print_and_save(curr_log, log_fname)
            viz_img = real_imgs[0:8].view(8,3,32,32)
            viz_rec = x_recon[0:8].view(8,3,32,32)
            out = torch.cat((viz_img, viz_rec), dim=0)
            fname = os.path.join(viz_dir, f"{global_steps}_recon_l2_256.png")
            disp_images(out, fname, 8, norm="0.5")
        
        
        global_steps += 1
    
    if epoch % 5 == 0:
        decoder.eval()
        encoder.eval()

        backup_param = copy_params(decoder)
        load_params(decoder, gen_avg_param)
        # metrics are computed by moving average
        fid_sample = compute_fid_sample(decoder, latent_dim)
        fid_recon = compute_fid_recon(encoder, decoder, test_loader, latent_dim)
        wandb.log({"fid_sample": fid_sample})
        wandb.log({"fid_recon": fid_recon})
        S = f"epoch:{epoch} sample:{fid_sample} recon:{fid_recon}"
        print_and_save(S, fid_fname)
        
        # save checkpoints
        torch.save(encoder.state_dict(), os.path.join(models_dir, f"{epoch}_encoder.sd"))
        torch.save(decoder.state_dict(), os.path.join(models_dir, f"{epoch}_decoder_avg.sd"))
        load_params(decoder, backup_param)
        torch.save(decoder.state_dict(), os.path.join(models_dir, f"{epoch}_decoder.sd"))

        fid_sample_1 = compute_fid_sample(decoder, latent_dim)
        fid_recon_1 = compute_fid_recon(encoder, decoder, test_loader, latent_dim)
        wandb.log({"fid_sample_actual": fid_sample_1})
        wandb.log({"fid_recon_actual": fid_recon_1})
        encoder.train()
        decoder.train()