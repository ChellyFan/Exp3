import sys

import torch

sys.path.insert(1, 'invertinggradients')
from invertinggradients.inversefed.reconstruction_algorithms import GradientReconstructor 

from src.training import CrossEntropy

def get_gradient(model, input_data, gt_labels, use_vb, train_mode, ng=None, ng_indices=None):
    ce = CrossEntropy()
    if train_mode:
        if hasattr(model, 'VB'):
            model.VB.train()
    else:
        model.eval()

    model.zero_grad()
    if use_vb:
        target_loss = ce(model(input_data), gt_labels) + model.loss()
    else:
        target_loss = ce(model(input_data), gt_labels)
    gradient = torch.autograd.grad(target_loss, model.parameters(), allow_unused=True)
    gradient = [grad.detach() for grad in gradient]
    # gradient = [grad.detach() + ((torch.randn_like(grad) * ng) if ng is not None else 0) for grad in gradient]
    # gradient = [grad.detach() + (torch.normal(0, ng, grad.shape, device=grad.device) if ng is not None else 0) for grad in gradient]
    if ng is not None:
        ng_indices = list(range(len(gradient))) if ng_indices is None else ng_indices
        for i in ng_indices:
            gradient[i] = gradient[i] + torch.normal(0, ng, gradient[i].shape, device=gradient[i].device)
    return gradient

def gradient_inversion(gradient, labels, model, data_shape, dm, ds, config, K=None):
    rec_machine = GradientReconstructor(model, (dm, ds), config, num_images=labels.shape[0])
    output, stats = rec_machine.reconstruct(gradient, labels, img_shape=data_shape, K=K)
    return output
