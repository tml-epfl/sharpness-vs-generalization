import torch
import utils
import copy
import math
from functools import partial


def zero_init_delta_dict(delta_dict, rho):
    for param in delta_dict:
        delta_dict[param] = torch.zeros_like(param).cuda()

    delta_norm = torch.cat([delta_param.flatten() for delta_param in delta_dict.values()]).norm()
    for param in delta_dict:
        delta_dict[param] *= rho / delta_norm

    return delta_dict


def random_init_on_sphere_delta_dict(delta_dict, rho, **unused_kwargs):
    for param in delta_dict:
        delta_dict[param] = torch.randn_like(param).cuda()

    delta_norm = torch.cat([delta_param.flatten() for delta_param in delta_dict.values()]).norm()
    for param in delta_dict:
        delta_dict[param] *= rho / delta_norm

    return delta_dict


def random_gaussian_dict(delta_dict, rho):
    n_el = 0
    for param_name, p in delta_dict.items():
        delta_dict[param_name] = torch.randn_like(p).cuda()
        n_el += p.numel()

    for param_name in delta_dict.keys():
        delta_dict[param_name] *= rho / (n_el ** .5)

    return delta_dict


def random_init_lw(delta_dict, rho, orig_param_dict, norm='l2', adaptive=False):
    assert norm in ['l2', 'linf'], f'Unknown perturbation model {norm}.'

    for param in delta_dict:
        if norm == 'l2':
            delta_dict[param] = torch.randn_like(delta_dict[param]).cuda()
        elif norm == 'linf':
            delta_dict[param] = (2 * torch.rand_like(delta_dict[param], device='cuda') - 1)

    for param in delta_dict:
        param_norm_curr = orig_param_dict[param].abs() if adaptive else 1
        delta_dict[param] *= rho * param_norm_curr

    return delta_dict


def weight_ascent_step_momentum(
    model, x, y, loss_f, orig_param_dict, delta_dict, prev_delta_dict,
    step_size, rho, momentum=0.75, layer_name_pattern='all', no_grad_norm=False,
    verbose=False, adaptive=False, eot_iter=0, eot_sigma=1., norm='linf'):
    """
    model:              w[k]
    orig_param_dict:    w[0]
    delta_dict:         w[k]-w[0]
    prev_delta_dict:    w[k-1]-w[0]
    1-alpha:            momentum coefficient
    
    -----------------------------------------------
    z[k+1] = P(w[k] + step_size*Grad F(w[k]))
    w[k+1] = P(w[k] + alpha*(z[k+1]-w[k])+(1-alpha)*(w[k]-w[k-1]))

    versions
    - old           -> L2-bound on all parameters, layer-wise rescaling
    - lw_l2_indep   -> L2-bound on each layer of rho * norm of the layer (if adaptive)
    """
    
    delta_dict_backup = {param: delta_dict[param].clone() for param in delta_dict}       # copy of perturbation dictionary  (w[k]-w[0])
    # curr_params = {name_p: p.clone() for name_p, p in model.named_parameters()}

    utils.zero_grad(model)
    output = model(x)
    # print(output.data)
    obj = loss_f(output, y) 
    obj.backward()
    # # Average gradients in a neighborhood of current model.
    # for _ in range(eot_iter):
    #     for n, p in model.named_parameters():
    #         p.data = curr_params[n] + torch.randn_like(p) * eot_sigma
    #     obj = loss_f(model(x), y) / (eot_iter + 1)
    #     obj.backward()
    
    with torch.no_grad():
        # Gradient ascent step, calculating perturbations
        if norm == 'l2':
            grad_norm = sum([param.grad.norm()**2 for _, param in model.named_parameters()])**0.5
            for _, param in model.named_parameters():
                delta_dict[param] += (
                    step_size / (grad_norm + 1e-12) * param.grad * (
                        1 if not adaptive else orig_param_dict[param].abs()))

        elif norm == 'linf':
            for _, param in model.named_parameters():
                grad_sign_curr = param.grad.sign()
                delta_dict[param] += (step_size * grad_sign_curr * (
                    1 if not adaptive else orig_param_dict[param].abs()))
        
        else:
            raise ValueError('wrong norm')
    
    utils.zero_grad(model)

    with torch.no_grad():
        # Projection step I, rescaling perturbations
        if norm == 'l2':  # Project onto L2-ball of radius rho (* ||w|| if adaptive)
            def weighted_norm(delta_dict):
                return sum([((delta_dict[param] / (orig_param_dict[param].abs() if adaptive else 1))**2).sum() for param in delta_dict])**0.5
            
            if not adaptive:  # standard projection on the sphere
                delta_norm = weighted_norm(delta_dict)
                if delta_norm > rho:
                    for param in delta_dict:
                        delta_dict[param] *= rho / delta_norm
            else:  # projection on the ellipsoid
                lmbd = 0.1  # weighted_norm(delta_dict_tmp) / 2 / rho - 0.5
                max_lmbd_limit = 10.0
                min_lmbd, max_lmbd = 0.0, max_lmbd_limit
                delta_dict_tmp = {param: delta_dict[param].clone() for param in delta_dict} 
                
                curr_norm = new_norm = weighted_norm(delta_dict_tmp)
                if curr_norm > rho:
                    while (new_norm - rho).abs() > 10**-5:
                        curr_norm = new_norm
                        for param in delta_dict:
                            c = 1/orig_param_dict[param].abs() if adaptive else 1
                            delta_dict_tmp[param] = delta_dict[param] / (1 + 2*lmbd*c**2)
                        new_norm = weighted_norm(delta_dict_tmp)
                        if new_norm > rho:  # if the norm still exceeds rho, increase lmbd and set a new min_lmbd
                            lmbd, min_lmbd = (lmbd + max_lmbd) / 2, lmbd
                        else:
                            lmbd, max_lmbd = (min_lmbd + lmbd) / 2, lmbd
                        if (max_lmbd_limit - max_lmbd) < 10**-2: 
                            max_lmbd_limit, max_lmbd = max_lmbd_limit*2, max_lmbd*2
                        # print(lmbd, weighted_norm(delta_dict_tmp))
                delta_dict = {param: delta_dict_tmp[param].clone() for param in delta_dict_tmp} 

        elif norm == 'linf':
            # Project onto Linf-ball of radius rho (* |w| if adaptive)
            for param in delta_dict:
                param_curr = orig_param_dict[param].abs() if adaptive else torch.ones_like(orig_param_dict[param])
                delta_dict[param] = torch.max(
                    torch.min(delta_dict[param], param_curr * rho), -1. * param_curr * rho)

        else:
            raise ValueError('wrong norm')

        # Average perturbations (apply momentum)
        for param_name, param in model.named_parameters():
            delta_dict[param] = (
                momentum * delta_dict[param] + (1 - momentum) * prev_delta_dict[param])
        
        # Applying perturbations
        for param in model.parameters():
            param.data = orig_param_dict[param] + delta_dict[param]

        for param_name, param in model.named_parameters():
            prev_delta_dict[param] = delta_dict_backup[param]
        
    return delta_dict, prev_delta_dict


def eval_APGD_sharpness(
    model, batches, loss_f, train_err, train_loss, rho=0.01,
    step_size_mult=1, n_iters=200, layer_name_pattern='all',
    n_restarts=1, min_update_ratio=0.75, rand_init=True,
    no_grad_norm=False, verbose=False, return_output=False,
    adaptive=False, version='default', norm='linf', **kwargs,
    ):
    """Computes worst-case sharpness for every batch independently, and returns
    the average values.
    """

    assert n_restarts == 1 or rand_init, 'Restarts need random init.'
    del train_err
    del train_loss
    gradient_step_kwargs = kwargs.get('gradient_step_kwargs', {})

    init_fn = partial(random_init_lw, norm=norm, adaptive=adaptive),

    def get_loss_and_err(model, loss_fn, x, y):
        """Compute loss and class. error on a single batch."""
        with torch.no_grad():
            output = model(x)
            loss = loss_fn(output, y)
            err = (output.max(1)[1] != y).float().mean()
        return loss.cpu().item(), err.cpu().item()

    orig_model_state_dict = copy.deepcopy(model.state_dict())
    orig_param_dict = {param: param.clone() for param in model.parameters()}
    
    n_batches, delta_norm = 0, 0.
    avg_loss, avg_err, avg_init_loss, avg_init_err = 0., 0., 0., 0.
    output = ""
    
    if version == 'default':
        p = [0, 0.22]
        w = [0, math.ceil(n_iters * 0.22)]
        
        while w[-1] < n_iters and w[-1] != w[-2]:
            p.append(p[-1] + max(p[-1] - p[-2] - 0.03, 0.06))
            w.append(math.ceil(p[-1] * n_iters))

        w = w[1:]  # No check needed at the first iteration.
        print(w)
        step_size_scaler = .5
    else:
        raise ValueError(f'Unknown version {version}')
    
    for i_batch, (x, _, y, _, _) in enumerate(batches):
        x, y = x.cuda(), y.cuda()

        # Loss and err on the unperturbed model.
        init_loss, init_err = get_loss_and_err(model, loss_f, x, y)

        # Accumulate over batches.
        avg_init_loss += init_loss
        avg_init_err += init_err

        worst_loss_over_restarts = init_loss
        worst_err_over_restarts = init_err
        worst_delta_norm_over_restarts = 0.

        for restart in range(n_restarts):

            if rand_init:
                delta_dict = {param: torch.zeros_like(param) for param in model.parameters()}  
                delta_dict = init_fn(delta_dict, rho, orig_param_dict=orig_param_dict)
                for param in model.parameters():
                    param.data += delta_dict[param]
            else:
                delta_dict = {param: torch.zeros_like(param) for param in model.parameters()}
                
            prev_delta_dict = {param: delta_dict[param].clone() for param in delta_dict}
            worst_model_dict = copy.deepcopy(model.state_dict())

            prev_worst_loss, worst_loss = init_loss, init_loss
            worst_err = init_err
            step_size, prev_step_size = 2 * rho * step_size_mult, 2 * rho * step_size_mult
            prev_cp = 0
            num_of_updates = 0
            
            for i in range(n_iters):
                
                delta_dict, prev_delta_dict = weight_ascent_step_momentum(
                    model, x, y, loss_f, orig_param_dict, delta_dict, prev_delta_dict,
                    step_size, rho, momentum=0.75, layer_name_pattern=layer_name_pattern,
                    no_grad_norm=no_grad_norm, verbose=False, adaptive=adaptive,
                    norm=norm, **gradient_step_kwargs)
                
                with torch.no_grad():
                    curr_loss, curr_err = get_loss_and_err(model, loss_f, x, y)
                    delta_norm_total = torch.cat([delta_param.flatten() for delta_param in delta_dict.values()]).norm().item()    
                        
                    if curr_loss > worst_loss:
                        worst_loss = curr_loss
                        worst_err = curr_err
                        worst_model_dict = copy.deepcopy(model.state_dict())
                        worst_delta_norm = delta_norm_total
                        num_of_updates += 1
                
                    if i in w:
                        cond1 = num_of_updates < (min_update_ratio * (i - prev_cp))
                        cond2 = (prev_step_size == step_size) and (prev_worst_loss == worst_loss)
                        prev_step_size, prev_worst_loss, prev_cp = step_size, worst_loss, i
                        num_of_updates = 0
                        
                        if cond1 or cond2:
                            print('Reducing step size.')
                            step_size *= step_size_scaler
                            model.load_state_dict(worst_model_dict)
                
                str_to_log = '[batch={} restart={} iter={}] Sharpness: obj={:.4f}, err={:.2%}, delta_norm={:.5f} (step={:.5f})'.format(
                    i_batch + 1, restart + 1, i + 1, curr_loss - init_loss, curr_err - init_err, delta_norm_total, step_size)
                if verbose:
                    print(str_to_log)
                output += str_to_log + '\n'
                            
            # Keep the best values over restarts.
            if worst_loss > worst_loss_over_restarts:
                worst_loss_over_restarts = worst_loss
                worst_err_over_restarts = worst_err
                worst_delta_norm_over_restarts = worst_delta_norm

            # Reload the unperturbed model for the next restart or batch.
            model.load_state_dict(orig_model_state_dict)

            if verbose:
                print('')

        # Accumulate over batches.
        n_batches += 1
        avg_loss += worst_loss_over_restarts
        avg_err += worst_err_over_restarts
        delta_norm = max(delta_norm, worst_delta_norm_over_restarts)

        if verbose:
            print('')

    vals = (
        (avg_loss - avg_init_loss) / n_batches,
        (avg_err - avg_init_err) / n_batches,
        delta_norm,
    )
    if return_output:
        vals += (output,)
    
    return vals


def eval_average_sharpness(
    model,
    batches,
    loss_f,
    n_iters=100,
    rho=1.,
    verbose=False,
    adaptive=False,
    return_output=True,
    norm='l2'):
    """Average case sharpness with Gaussian noise ~ (0, rho)."""

    def get_loss_and_err(model, loss_fn, x, y):
        """Compute loss and class. error on a single batch."""
        with torch.no_grad():
            output = model(x)
            loss = loss_fn(output, y)
            err = (output.max(1)[1] != y).float().mean()
        return loss.cpu().item(), err.cpu().item()

    orig_param_dict = {param_name: p.clone() for param_name, p in model.named_parameters()} # {param: param.clone() for param in model.parameters()}
    # orig_norm = torch.cat([p.flatten() for p in orig_param_dict.values()]).norm()
    
    orig_norm = 0
    n_el = 0
    for p in orig_param_dict.values():
        orig_norm += p.flatten().norm() ** 2. * p.numel()
        n_el += p.numel()
    orig_norm = (orig_norm / n_el) ** .5
    noisy_model = copy.deepcopy(model)

    delta_dict = {param_name: torch.zeros_like(param) for param_name, param in model.named_parameters()}
    print('Named params:', len(delta_dict))
    print('Params:', len([None for _ in model.parameters()]))
    print('rho:', rho, 'samples:', n_iters)
    
    n_batches, avg_loss, avg_err, avg_init_loss, avg_init_err = 0, 0., 0., 0., 0.
    output = ''

    with torch.no_grad():
        for i_batch, (x, _, y, _, _) in enumerate(batches):
            x, y = x.cuda(), y.cuda()

            # Loss and err on the unperturbed model.
            init_loss, init_err = get_loss_and_err(model, loss_f, x, y)
            avg_init_loss += init_loss
            avg_init_err += init_err

            batch_loss, batch_err = 0., 0.

            for i in range(n_iters):
                delta_dict = random_init_lw(delta_dict, rho, orig_param_dict, norm=norm, adaptive=adaptive)
                for (param_name, delta), (_, param) in zip(delta_dict.items(), noisy_model.named_parameters()):
                    param.data = orig_param_dict[param_name] + delta_dict[param_name]

                curr_loss, curr_err = get_loss_and_err(noisy_model, loss_f, x, y)
                batch_loss += curr_loss
                batch_err += curr_err

            n_batches += 1
            avg_loss += (batch_loss / n_iters)
            avg_err += (batch_err / n_iters)

            str_to_log = f'[batch={i_batch + 1}] obj={batch_loss / n_iters - init_loss}' + \
                f' err={batch_err / n_iters - init_err}'
            if verbose:
                print(str_to_log)
            output += str_to_log + '\n'

    vals = (
        (avg_loss - avg_init_loss) / n_batches,
        (avg_err - avg_init_err) / n_batches,
        0.,
    )
    if return_output:
        vals += (output,)
    
    return vals

