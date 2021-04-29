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
latent_dim         = 128
cont_dim           = 16
cont_k             = 8192
cont_temp          = 0.07

# multi-scale contrastive setting
layers             = ["b1", "final"]

name =("").join(layers)
log_fname = f"logs/cifar10-{name}"
fid_fname = f"logs/FID_cifar10-{name}"
viz_dir = f"viz/cifar10-{name}"
models_dir = f"saved_models/cifar10-{name}"
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
dual_encoder = torch.nn.DataParallel(DualEncoder(cont_dim)).cuda()
encoder.apply(weights_init)
decoder.apply(weights_init)
dual_encoder.apply(weights_init)
dual_encoder_M = torch.nn.DataParallel(DualEncoder(cont_dim)).cuda()
for p, p_momentum in zip(dual_encoder.parameters(), dual_encoder_M.parameters()):
    p_momentum.data.copy_(p.data)
    p_momentum.requires_grad = False
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
shared_params = list(dual_encoder.module.block1.parameters()) + \
                list(dual_encoder.module.block2.parameters()) + \
                list(dual_encoder.module.block3.parameters()) + \
                list(dual_encoder.module.block4.parameters()) + \
                list(dual_encoder.module.l5.parameters())
opt_shared = torch.optim.Adam(filter(lambda p: p.requires_grad, 
                                        shared_params),
                                lr, (beta1, beta2))
opt_disc_head = torch.optim.Adam(filter(lambda p: p.requires_grad, 
                    dual_encoder.module.head_disc.parameters()),
                lr, (beta1, beta2))
cont_params = list(dual_encoder.module.head_b1.parameters()) + \
                list(dual_encoder.module.head_b2.parameters()) + \
                list(dual_encoder.module.head_b3.parameters()) + \
                list(dual_encoder.module.head_b4.parameters())
opt_cont_head = torch.optim.Adam(filter(lambda p: p.requires_grad, cont_params),
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
    dual_encoder.train()
    for iter_idx, (imgs, _) in enumerate(tqdm(train_loader)):
        curr_bs = imgs.shape[0]
        curr_log = f"{epoch}:{iter_idx}\t"
        real_imgs = imgs.type(torch.cuda.FloatTensor)
        z = torch.cuda.FloatTensor(np.random.normal(0, 1, (imgs.shape[0], latent_dim)))
        # ---------------------
        #  Train Discriminator
        # ---------------------
        opt_shared.zero_grad()
        opt_disc_head.zero_grad()
        real_validity = dual_encoder(real_imgs, mode="dis")
        fake_imgs = decoder(z).detach()
        fake_validity = dual_encoder(fake_imgs, mode="dis")
        rec, mu, logvar = f_recon(real_imgs, encoder, decoder, latent_dim)
        rec_validity = dual_encoder(rec, mode="dis")
        # cal loss
        d_loss = torch.mean(nn.ReLU(inplace=True)(1.0 - real_validity)) + \
                 torch.mean(nn.ReLU(inplace=True)(1.0 + fake_validity))*0.5 + \
                 torch.mean(nn.ReLU(inplace=True)(1.0 + rec_validity))*0.5
        d_loss.backward()
        curr_log += f"d:{d_loss.item():.2f}\t"
        opt_shared.step()
        opt_disc_head.step()

        # -----------------
        #  Train Generator
        # -----------------
        if global_steps % 5 == 0:
            opt_decoder.zero_grad()
            opt_encoder.zero_grad()
            gen_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (gen_batch_size, latent_dim)))
            gen_imgs = decoder(gen_z)
            fake_validity = dual_encoder(gen_imgs, mode="dis")
            rec, mu, logvar = f_recon(real_imgs, encoder, decoder, latent_dim)
            rec_validity = dual_encoder(rec, mode="dis")
            # cal loss
            g_loss = -(torch.mean(fake_validity)*0.5 + torch.mean(rec_validity)*0.5)
            kld = (-0.5 * torch.sum(1+logvar-mu.pow(2)-logvar.exp()))*lambda_kld
            (g_loss+kld).backward()
            opt_decoder.step()
            opt_encoder.step()
            curr_log += f"g:{g_loss.item():.2f}\t"

            # contrastive
            opt_encoder.zero_grad()
            opt_decoder.zero_grad()
            opt_shared.zero_grad()
            opt_cont_head.zero_grad()
            rec, mu, logvar = f_recon(real_imgs, encoder, decoder, latent_dim)
            im_k = real_imgs
            im_q = rec
            with torch.no_grad():
                # update momentum encoder
                for p, p_mom in zip(dual_encoder.parameters(), dual_encoder_M.parameters()):
                    p_mom.data = (p_mom.data*0.999) + (p.data*(1.0-0.999))
                d_k = dual_encoder_M(im_k, mode="cont")
                for l in layers:
                    d_k[l] = F.normalize(d_k[l], dim=1)
            total_cont = torch.tensor(0.0).cuda()
            d_q = dual_encoder(im_q, mode="cont")
            for l in layers:
                q = F.normalize(d_q[l], dim=1)
                k = d_k[l]
                queue = d_queue[l]
                l_pos = torch.einsum("nc,nc->n", [k,q]).unsqueeze(-1)
                l_neg = torch.einsum('nc,ck->nk', [q,queue.detach()])
                logits = torch.cat([l_pos, l_neg], dim=1) / cont_temp#0.07
                labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
                cont_loss = nn.CrossEntropyLoss()(logits, labels) * lambda_cont
                total_cont += cont_loss
                acc1, acc5 = accuracy(logits, labels, topk=(1, 5))
                curr_log += f"cont{l}:{cont_loss.item():.1f}\t"
                curr_log += f"acc1{l}:{acc1.item():.1f}\t"
                curr_log += f"acc5{l}:{acc5.item():.1f}\t"
            kld = (-0.5 * torch.sum(1+logvar-mu.pow(2)-logvar.exp()))*lambda_kld
            (total_cont+kld).backward()
            opt_encoder.step()
            opt_decoder.step()
            opt_shared.step()
            opt_cont_head.step()

            for l in layers:
                ptr = int(d_queue_ptr[l])
                d_queue[l][:, ptr:(ptr+curr_bs)] = d_k[l].transpose(0,1)
                ptr = (ptr+curr_bs)%cont_k # move the pointer ahead
                d_queue_ptr[l][0] = ptr
            with torch.no_grad():
                rec_pix = torch.nn.MSELoss()(im_q, im_k).mean()
            # moving average weight
            for p, avg_p in zip(decoder.parameters(), gen_avg_param):
                avg_p.mul_(0.999).add_(0.001, p.data)
        
        if global_steps%250 == 0:
            print_and_save(curr_log, log_fname)
            viz_img = im_k[0:8].view(8,3,32,32)
            viz_rec = im_q[0:].view(curr_bs,3,32,32)
            out = torch.cat((viz_img, viz_rec), dim=0)
            fname = os.path.join(viz_dir, f"{global_steps}_recon.png")
            disp_images(out, fname, 8, norm="0.5")
            fname = os.path.join(viz_dir, f"{global_steps}_sample.png")
            disp_images(fake_imgs.view(-1,3,32,32), fname, 8, norm="0.5")
        
        global_steps += 1
    
    if epoch % 5 == 0:
        decoder.eval()
        encoder.eval()
        backup_param = copy_params(decoder)
        load_params(decoder, gen_avg_param)
        fid_sample = compute_fid_sample(decoder, latent_dim)
        fid_recon = compute_fid_recon(encoder, decoder, test_loader, latent_dim)
        S = f"epoch:{epoch} sample:{fid_sample} recon:{fid_recon}"
        print_and_save(S, fid_fname)
        # save checkpoints
        torch.save(encoder.state_dict(), os.path.join(models_dir, f"{epoch}_encoder.sd"))
        torch.save(decoder.state_dict(), os.path.join(models_dir, f"{epoch}_decoder_avg.sd"))
        load_params(decoder, backup_param)
        torch.save(decoder.state_dict(), os.path.join(models_dir, f"{epoch}_decoder.sd"))
        torch.save(dual_encoder.state_dict(), os.path.join(models_dir, f"{epoch}_dual_encoder.sd"))
        torch.save(dual_encoder_M.state_dict(), os.path.join(models_dir, f"{epoch}_dual_encoder_M.sd"))
        torch.save(opt_encoder.state_dict(), os.path.join(models_dir, f"{epoch}_opt_encoder.sd"))
        torch.save(opt_decoder.state_dict(), os.path.join(models_dir, f"{epoch}_opt_decoder.sd"))
        torch.save(opt_shared.state_dict(), os.path.join(models_dir, f"{epoch}_opt_shared.sd"))
        torch.save(opt_cont_head.state_dict(), os.path.join(models_dir, f"{epoch}_opt_cont_head.sd"))
        torch.save(opt_disc_head.state_dict(), os.path.join(models_dir, f"{epoch}_opt_disc_head.sd"))
        for layer in layers:
            torch.save(d_queue[layer], os.path.join(models_dir, f"{epoch}_{layer}_queue.sd"))
            torch.save(d_queue_ptr[layer], os.path.join(models_dir, f"{epoch}_{layer}_queueptr.sd"))
        encoder.train()
        decoder.train()