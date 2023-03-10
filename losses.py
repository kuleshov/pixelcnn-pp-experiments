import torch
from math import sqrt

def make_scale_matrix(num_gen, num_orig, device='cpu'):
    # first 'N' entries have '1/N', next 'M' entries have '-1/M'
    s1 =  torch.ones(num_gen, 1, requires_grad=False, device=device)/num_gen
    s2 = -torch.ones(num_orig, 1, requires_grad=False, device=device)/num_orig
    return torch.cat([s1, s2], dim=0)

# before we had: sigma = [2, 5, 10, 20, 40, 80]
def kernelized_energy_loss(x, gen_x, sigma = [0.1, 0.5, 1, 2, 5, 10, 20, 40, 80, 160], reduction='sum', device='cuda'):
    """Computes this kernelized energy loss

    Tensors x and gen_x need to have shape: (batch, chans, dimx, dimy, nsamples)
    For the tensor of real data x, we should have nsamples=1.
    This code will compute the kernelized MMD along the dimension 4 (of size nsamples)
    It will then take the average across all dimensions.
    """
    # concatenation of the generated images and images from the dataset
    # first 'N' rows are the generated ones, next 'M' are from the data
    X = torch.cat([gen_x, x], dim=4) # shape=(b, c, x, y, s+1)
    # dot product between all combinations of rows in 'X'

    # this converts each tensor from (batch, chans, dimx, dimy, nsamples+1)
    # to (batch, dimx, dimy, nsamples+1, chans)
    X = X.permute(0,2,3,4,1)
    d = X.shape[4]

    XX = torch.matmul(X, torch.transpose(X, 3, 4)) # (b, x, y, s+1, s+1)
    # dot product of rows with themselves
    X2 = torch.sum(X * X, dim = 4, keepdim=True) # (b, x, y, s+1, 1)
    # exponent entries of the RBF kernel (without the sigma) for each
    # combination of the rows in 'X'
    # -0.5 * (x^Tx - 2*x^Ty + y^Ty)
    exponent = XX - 0.5 * X2 - 0.5 * torch.transpose(X2, 3, 4) 
    # exponent = exponent / sqrt(d) # (b, x, y, s+1, s+1)
    # exponent = torch.clip(exponent, -1e4, 1e4)
    # scaling constants for each of the rows in 'X'
    s = make_scale_matrix(gen_x.shape[4], x.shape[4], device=device)
    # scaling factors of each of the kernel values, corresponding to the
    # exponent values
    S = torch.matmul(s, torch.transpose(s, 0, 1)) # shape=(nsamples+1, nsamples+1)
    S = S.unsqueeze(0).unsqueeze(1).unsqueeze(2) # shape=(1, 1, 1, s+1, s+1)
    loss = 0
    # for each bandwidth parameter, compute the MMD value and add them all
    for i in range(len(sigma)):
        # kernel values for each combination of the rows in 'X'
        v = 1.0 / sigma[i] * exponent
        kernel_val = torch.exp(v)
        # mean or sum here?
        loss += torch.sum(S * kernel_val, axis=[1,2,3]) # shape=(b, s+1)

    if reduction == 'mean':
        final_loss = torch.mean(loss, axis=1)
    elif reduction == 'sum':
        final_loss = torch.sum(loss, axis=1)
    elif reduction == 'none':
        final_loss = loss
    else:
        raise ValueError()
    # TODO: this is a strange place to take sqrt
    final_loss = torch.sqrt(final_loss) # shape=(batch, )
    return final_loss.mean()

def simple_energy_loss(y_pred1, y_pred2, y_true):
    # WARNING: copy-pasted from tensorflow and did not verify if this works in torch
    y_true = y_true.flatten(1)
    y_pred1 = y_pred1.flatten(1)
    y_pred2 = y_pred2.flatten(1)
    norm1 = torch.linalg.norm(y_pred1 - y_pred2, axis=1)
    norm2 = torch.linalg.norm(y_pred1 - y_true, axis=1)
    energy_score = 0.5*norm1**0.5 - norm2**0.5
    return -torch.mean(energy_score, axis=0)        

def quantile_loss(y_true, y_pred, alpha):
    y_true = y_true.flatten(1)
    y_pred = y_pred.flatten(1)
    alpha = alpha.flatten(1)
    loss_vector = torch.maximum(
      alpha * (y_true - y_pred), 
      (alpha - 1) * (y_true - y_pred)
    )
    return loss_vector.sum(1).mean(0)