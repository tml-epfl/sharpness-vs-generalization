import torch
import torch.nn.functional as F
from datetime import datetime


def process_arg(args, arg):
    if arg in ['gpu', 'eval_sharpness', 'log', 'rewrite']:
        return ''
    if arg == 'adaptive':
        return ''
    if arg != 'model_path':
        return str(getattr(args, arg))
    # return args.model_path.split('/')[-1][:24].replace(' ', '_')
    return ''


def get_path(args, log_folder):
    name = '-'.join([process_arg(args, arg) for arg in list(filter(lambda x: x not in ['adaptive'], vars(args)))])
    name = str(datetime.now())[:-3].replace(' ', '_') + name
    if getattr(args, 'adaptive'):
        name += '-adaptive'
    path = f'{log_folder}/{name}.json'
    return path


def zero_grad(model):
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()
        

def compute_err(batches, model, loss_f=F.cross_entropy, n_batches=-1):
    n_wrong_classified, train_loss_sum, n_ex = 0, 0.0, 0

    with torch.no_grad():
        for i, (X, _, y, _, ln) in enumerate(batches):
            if n_batches != -1 and i > n_batches:  # limit to only n_batches
                break
            X, y = X.cuda(), y.cuda()
            
            # print(X, X.shape)
            output = model(X)
            loss = loss_f(output, y)  

            n_wrong_classified += (output.max(1)[1] != y).sum().item()
            train_loss_sum += loss.item() * y.size(0)
            n_ex += y.size(0)

    err = n_wrong_classified / n_ex
    avg_loss = train_loss_sum / n_ex

    return err, avg_loss


def estimate_loss_err(model, batches, loss_f):
    err = 0
    loss = 0
    
    with torch.no_grad():
        for i_batch, (x, _, y, _, _) in enumerate(batches):
            x, y = x.cuda(), y.cuda()
            curr_y = model(x)
            loss += loss_f(curr_y, y)
            err += (curr_y.max(1)[1] != y).float().mean().item()
            
    return loss.item() / len(batches), err / len(batches)

