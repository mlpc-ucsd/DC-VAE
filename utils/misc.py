import os, sys, pdb, time
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torchvision

def reparameterize(mu, logvar):
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    return mu + eps*std

# weight init
def weights_init(m):
    init_type="xavier_uniform"
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.xavier_uniform(m.weight.data, 1.)
    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

def print_and_save(s, fname):
    print(s)
    with open(fname,"a") as f:
        f.write(s + "\n")

def f_recon(img, netE, netD, zdim, mode="train", clipping=False):
    bs, imsize = img.shape[0], img.shape[2]
    mu_logvar = netE(img).view(bs,-1)
    mu = mu_logvar[:,0:zdim]
    logvar = mu_logvar[:,zdim:]
    z = reparameterize(mu, logvar)
    
    if mode=="train":
        if clipping: z = torch.clamp(z, min=-1.0, max=1.0)
        recon = netD(z).view(bs,3,imsize,imsize)
    else:
        if clipping: mu = torch.clamp(mu, min=-1.0, max=1.0)
        recon = netD(mu).view(bs,3,imsize,imsize)
    return recon, mu, logvar

def f_recon_with_z(img, netE, netD, zdim, mode="train", clipping=False):
    bs, imsize = img.shape[0], img.shape[2]
    mu_logvar = netE(img).view(bs,-1)
    mu = mu_logvar[:,0:zdim]
    logvar = mu_logvar[:,zdim:]
    z = reparameterize(mu, logvar)
    
    if mode=="train":
        if clipping: z = torch.clamp(z, min=-1.0, max=1.0)
        recon = netD(z).view(bs,3,imsize,imsize)
    else:
        if clipping: mu = torch.clamp(mu, min=-1.0, max=1.0)
        recon = netD(mu).view(bs,3,imsize,imsize)
    return recon, mu, logvar, z



class NormalLogProb(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, loc, scale, z):
    var = torch.pow(scale, 2)
    return -0.5 * torch.log(2 * np.pi * var) - torch.pow(z - loc, 2) / (2 * var)

class BernoulliLogProb(nn.Module):
  def __init__(self):
    super().__init__()
    self.bce_with_logits = nn.BCEWithLogitsLoss(reduction='none')

  def forward(self, logits, target):
    # bernoulli log prob is equivalent to negative binary cross entropy
    return -self.bce_with_logits(logits, target)

log_q_z = NormalLogProb()
log_p_z = NormalLogProb()
log_p_x = BernoulliLogProb()

latent_size = 128
p_z_loc = torch.zeros(latent_size).cuda()
p_z_scale = torch.ones(latent_size).cuda()

def get_many_z(img, netE, zdim, mode="train", clipping=False, n_samples=1):
    bs, imsize = img.shape[0], img.shape[2]
    mu_logvar = netE(img).view(bs,-1)
    mu = mu_logvar[:,0:zdim].unsqueeze(1)
    logvar = mu_logvar[:,zdim:].unsqueeze(1)
    #z = reparameterize(mu, logvar)
    
    std = torch.exp(0.5*logvar)
    #eps = torch.randn_like(std)
    eps = torch.randn((mu.shape[0], n_samples, mu.shape[-1]), device=mu.device)
    
    z = mu + eps*std
    log_qz = log_q_z(mu, std, z).sum(-1, keepdim=True)
    
    return z, log_qz


def log_p_x_and_z(z, img, netD, zdim, mode="train", clipping=False, n_samples=1):
    bs, imsize = img.shape[0], img.shape[2]
    """
    mu_logvar = netE(img).view(bs,-1)
    mu = mu_logvar[:,0:zdim]
    logvar = mu_logvar[:,zdim:]
    z = reparameterize(mu, logvar)
    """
    
    print ("p_z_loc.shape, p_z_scale.shape, z.shape", p_z_loc.shape, p_z_scale.shape, z.shape)
    log_pz = log_p_z(p_z_loc, p_z_scale, z).sum(-1, keepdim=True)
    #logits = self.generative_network(z)
    if mode=="train":
        if clipping: z = torch.clamp(z, min=-1.0, max=1.0)
        recon = netD(mu)
        #print ("recon.shape", recon.shape)
        recon = recon.view(bs,n_samples,3,imsize,imsize)
    else:
        if clipping: mu = torch.clamp(z, min=-1.0, max=1.0)
        else: mu = z
        recon = netD(mu)
        #print ("recon.shape", recon.shape)
        recon = recon.view(bs,n_samples,3,imsize,imsize)
    print ("MSE", torch.nn.MSELoss()(recon[:, 0], img).sum().item())

    #print ("recon.shape", recon.shape)
    #print ("z.shape", z.shape)
    recon_rescale = recon / 2.0 + 0.5
    #recon_rescale = recon_rescale * 0.4
    recon_rescale = recon_rescale.reshape(bs,n_samples, 3*32*32)
    img_rescale = img / 2.0 + 0.5
    #img_rescale = img_rescale * 0.4
    #print ("img_rescale", img_rescale)
    img_rescale = img_rescale.reshape(-1, 3*32*32)

    #print ("recon_rescale, img_rescale", recon_rescale.min(), img_rescale.min())
    # unsqueeze sample dimension
    recon_rescale, img_rescale = torch.broadcast_tensors(recon_rescale, img_rescale.unsqueeze(1))
    log_px = log_p_x(recon_rescale, img_rescale).sum(-1, keepdim=True)
    ##log_px = log_p_x(img_rescale, recon_rescale).sum(-1, keepdim=True)


    print ("log_pz.shape", log_pz.shape, log_px.shape)
    print ("log_pz.mean()", log_pz.mean(), log_px.mean())    # log_pz.mean() tensor(-221.8115, device='cuda:0') tensor(-2234.6194, device='cuda:0')
    return log_pz + log_px
    #return log_px


def fix_seed(random_seed=123):
    torch.cuda.manual_seed(random_seed)
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def denorm(img, mean, std):
    img = img.clone().detach()
    # img shape is B, 3,64,64 and detached
    for i in range(3):
        img[:, i,:,:] *= std[i]
        img[:, i,:,:] += mean[i]
    return img

def disp_images(img, fname, nrow, norm="none"):
    if not os.path.exists(os.path.dirname(fname)):
        os.makedirs(os.path.dirname(fname))
    bs = img.shape[0]
    imsize = img.shape[2]
    nc = img.shape[1]
    if nc==3 and norm=="0.5":
        img = denorm(img,mean=[0.5, 0.5, 0.5], std=[0.5,0.5,0.5])
    elif nc==3 and norm=="none":
        pass
    elif nc==1:
        img = img
    else:
        raise ValueError("image has incorrect channels")
    img = img.view(bs,-1,imsize,imsize).cpu()
    grid =  torchvision.utils.make_grid(img,nrow=nrow)
    torchvision.utils.save_image(grid, fname)

def copy_params(model):
    flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten

def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)

