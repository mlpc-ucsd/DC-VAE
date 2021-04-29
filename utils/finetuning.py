import os, sys, time, pdb
from models.models_32 import *

def load_all_models(P, epoch, latent_dim=128, cont_dim=16):
    encoder = torch.nn.DataParallel(Encoder(latent_dim)).cuda()
    encoder.load_state_dict(torch.load(os.path.join(P, f"{epoch}_encoder.sd")))
    decoder = torch.nn.DataParallel(Decoder(latent_dim)).cuda()
    decoder.load_state_dict(torch.load(os.path.join(P, f"{epoch}_decoder.sd")))
    decoder_avg = torch.nn.DataParallel(Decoder(latent_dim)).cuda()
    decoder_avg.load_state_dict(torch.load(os.path.join(P, f"{epoch}_decoder_avg.sd")))
    dual_encoder = torch.nn.DataParallel(DualEncoder(cont_dim)).cuda()
    dual_encoder.load_state_dict(torch.load(os.path.join(P, f"{epoch}_dual_encoder.sd")))
    dual_encoder_M = torch.nn.DataParallel(DualEncoder(cont_dim)).cuda()
    dual_encoder_M.load_state_dict(torch.load(os.path.join(P, f"{epoch}_dual_encoder.sd")))
    return encoder, decoder, decoder_avg, dual_encoder, dual_encoder_M

def load_all_queues(P, epoch, layers=["final"]):
    d_queue, d_queue_ptr = {}, {}
    for layer in layers:
        d_queue[layer] = torch.load(os.path.join(P, f"{epoch}_{layer}_queue.sd"))
        d_queue_ptr[layer] = torch.zeros(1, dtype=torch.long)
    return d_queue, d_queue_ptr