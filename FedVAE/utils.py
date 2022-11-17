import sys

import torch

sys.path.insert(1, 'invertinggradients')
from invertinggradients.inversefed.reconstruction_algorithms import GradientReconstructor 

def get_gradient(model, input_data, train_mode, ng=None, ng_indices=None):
    
    if train_mode:
        model.train()
    else:
        model.eval()

    model.zero_grad()
    # if use_vb:
    #     target_loss = ce(model(input_data), gt_labels) + model.loss()
    # else:
    #     target_loss = ce(model(input_data), gt_labels)
    recon_batch, mu, var = model(input_data)
    target_loss = model.loss_function(recon_batch, input_data, mu, var, beta=.2)
    gradient = torch.autograd.grad(target_loss, model.parameters(), allow_unused=True)
    gradient = [grad.detach() for grad in gradient]
    # gradient = [grad.detach() + ((torch.randn_like(grad) * ng) if ng is not None else 0) for grad in gradient]
    # gradient = [grad.detach() + (torch.normal(0, ng, grad.shape, device=grad.device) if ng is not None else 0) for grad in gradient]
    if ng is not None:
        ng_indices = list(range(len(gradient))) if ng_indices is None else ng_indices
        for i in ng_indices:
            gradient[i] = gradient[i] + torch.normal(0, ng, gradient[i].shape, device=gradient[i].device)
    return gradient

def gradient_inversion(gradient, model, input_shape, dm, ds, config, reconstruct_dp=False, K=None):
    rec_machine = GradientReconstructor(model, (dm, ds), config)
    output, stats = rec_machine.reconstruct(gradient, input_shape=input_shape, reconstruct_dp=reconstruct_dp, K=K)
    return output
