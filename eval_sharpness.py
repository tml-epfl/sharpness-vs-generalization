import argparse
import os
import numpy as np
import torch
import time
import data
import models
import utils
import json
import sharpness
import torch.nn.functional as F


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--dataset', default='cifar10', type=str)
    parser.add_argument('--model', default='resnet18', type=str)
    parser.add_argument('--model_path', type=str, help='model path')
    parser.add_argument('--n_eval', default=10000, type=int, help='#examples to evaluate on error')
    parser.add_argument('--bs', default=256, type=int, help='batch size for error computation')
    parser.add_argument('--n_eval_sharpness', default=1024, type=int, help='#examples to evaluate on sharpness')
    parser.add_argument('--bs_sharpness', default=128, type=int, help='batch size for sharpness experiments')
    parser.add_argument('--rho', default=0.1, type=float, help='L2 radius for sharpness')
    parser.add_argument('--step_size_mult', default=1.0, type=float, help='step size multiplier for sharpness')
    parser.add_argument('--n_iters', default=20, type=int, help='number of iterations for sharpness')
    parser.add_argument('--n_restarts', default=1, type=int, help='number of restarts for sharpness')
    parser.add_argument('--model_width', default=64, type=int, help='model width (# conv filters on the first layer for ResNets)')
    parser.add_argument('--sharpness_on_test_set', action='store_true', help='compute sharpness on the test set')
    parser.add_argument('--sharpness_rand_init', action='store_true', help='random initialization')
    parser.add_argument('--merge_bn_stats', action='store_true', help='merge BN means and variances to its learnable parameters')
    parser.add_argument('--no_grad_norm', action='store_true', help='no gradient normalization in APGD')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--algorithm', default='m_apgd_linf', choices=['avg_l2', 'avg_linf', 'm_apgd_l2', 'm_apgd_linf'], type=str)
    parser.add_argument('--log_folder', default='logs_eval', type=str)
    parser.add_argument('--adaptive', action='store_true')
    parser.add_argument('--normalize_logits', action='store_true')
    parser.add_argument('--data_augm_sharpness', action='store_true')    
    return parser.parse_args()


start_time = time.time()
args = get_args()

n_cls = 10 if args.dataset != 'cifar100' else 100
sharpness_split = 'test' if args.sharpness_on_test_set else 'train'
assert args.n_eval_sharpness % args.bs_sharpness == 0, 'args.n_eval should be divisible by args.bs_sharpness'
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
np.set_printoptions(precision=4, suppress=True)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

loss_f = lambda logits, y: F.cross_entropy(logits, y, reduction='mean')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = models.get_model(args.model, n_cls, False, data.shapes_dict[args.dataset], args.model_width, normalize_logits=args.normalize_logits).to(device).eval()
model_dict = torch.load('{}'.format(args.model_path))['last']
model.load_state_dict({k: v for k, v in model_dict.items()})  
model = models.LogitNormalizationWrapper(model, normalize_logits=args.normalize_logits)

eval_train_batches = data.get_loaders(args.dataset, args.n_eval, args.bs, split='train', shuffle=False,
                                      data_augm=False, drop_last=False)
eval_test_batches = data.get_loaders(args.dataset, args.n_eval, args.bs, split='test', shuffle=False,
                                     data_augm=False, drop_last=False)
eval_test_corruptions_batches = data.get_loaders(args.dataset + 'c', args.n_eval, args.bs, split='test', shuffle=False,
                                     data_augm=False, drop_last=False)
train_err, train_loss = utils.compute_err(eval_train_batches, model)
test_err, test_loss = utils.compute_err(eval_test_batches, model)
test_err_corrupt, test_loss_corrupt = utils.compute_err(eval_test_corruptions_batches, model)
print('[train] err={:.2%} loss={:.5f}, [test] err={:.2%}, loss={:.4f}, [test corrupted] err={:.2%}, loss={:.4f}'.format(train_err, train_loss, test_err, test_loss, test_err_corrupt, test_loss_corrupt))


batches_sharpness = data.get_loaders(args.dataset, args.n_eval_sharpness, args.bs_sharpness, split=sharpness_split, shuffle=False,
                                        data_augm=args.data_augm_sharpness, drop_last=False, randaug=args.data_augm_sharpness)
    
if args.algorithm == 'm_apgd_l2':
    sharpness_obj, sharpness_err, _, output = sharpness.eval_APGD_sharpness(
        model, batches_sharpness, loss_f, train_err, train_loss, 
        rho=args.rho, n_iters=args.n_iters, n_restarts=args.n_restarts, step_size_mult=args.step_size_mult,
        rand_init=args.sharpness_rand_init, no_grad_norm=args.no_grad_norm,
        verbose=True, return_output=True, adaptive=args.adaptive, version='default', norm='l2')

if args.algorithm == 'm_apgd_linf':
    sharpness_obj, sharpness_err, _, output = sharpness.eval_APGD_sharpness(
        model, batches_sharpness, loss_f, train_err, train_loss, 
        rho=args.rho, n_iters=args.n_iters, n_restarts=args.n_restarts, step_size_mult=args.step_size_mult,
        rand_init=args.sharpness_rand_init, no_grad_norm=args.no_grad_norm,
        verbose=True, return_output=True, adaptive=args.adaptive, version='default', norm='linf')

if args.algorithm == 'avg_l2':
    sharpness_obj, sharpness_err, _, output = sharpness.eval_average_sharpness(
        model, batches_sharpness, loss_f, rho=args.rho, n_iters=args.n_iters, return_output=True, adaptive=args.adaptive, norm='l2')

if args.algorithm == 'avg_linf':
    sharpness_obj, sharpness_err, _, output = sharpness.eval_average_sharpness(
        model, batches_sharpness, loss_f, rho=args.rho, n_iters=args.n_iters, return_output=True, adaptive=args.adaptive, norm='linf')

print('sharpness: obj={:.5f}, err={:.2%}'.format(sharpness_obj, sharpness_err)) 


### Save all the arguments, train_err, train_loss,test_err, test_loss, sharpness_obj, sharpness_err, sharpness_gradp_norm
checkpoint = dict([(arg, getattr(args, arg)) for arg in vars(args)])
# checkpoint['output'] = output
checkpoint['train_err'] = train_err
checkpoint['train_loss'] = train_loss
checkpoint['test_err'] = test_err
checkpoint['test_loss'] = test_loss
checkpoint['test_err_corrupt'] = test_err_corrupt
checkpoint['test_loss_corrupt'] = test_loss_corrupt
checkpoint['sharpness_obj'] = sharpness_obj 
checkpoint['sharpness_err'] = sharpness_err
checkpoint['time'] = (time.time() - start_time) / 60

path = utils.get_path(args, args.log_folder)
if not os.path.exists(args.log_folder):
    os.makedirs(args.log_folder)
with open(path, 'w') as outfile:
    json.dump(checkpoint, outfile)

print('Done in {:.2f}m'.format((time.time() - start_time) / 60))

