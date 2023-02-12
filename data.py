import torch
import torch.utils.data as td
import numpy as np
from types import SimpleNamespace
from torchvision import datasets, transforms
from robustbench.data import load_cifar10c, load_cifar100c
from randomaug import RandAugment


class DatasetWithLabelNoise(torch.utils.data.Dataset):
    def __init__(self, data, split, transform):
        self.data = data
        self.split = split
        self.transform = transform

    def __getitem__(self, index):
        x = self.data.data[index]
        x1 = self.transform(x) if self.transform is not None else x
        if self.split == 'train':
            x2 = self.transform(x) if self.transform is not None else x
        else:  # to save a bit of computations
            x2 = x1
        y = self.data.targets[index]
        y_correct = self.data.targets_correct[index]
        label_noise = self.data.label_noise[index]
        return x1, x2, y, y_correct, label_noise

    def __len__(self):
        return len(self.data.targets)


def dataset_cifar10c(*args, **kwargs):
    x, y = load_cifar10c(data_dir=args[0], n_examples=15000, severity=5, shuffle=True)
    data_obj = SimpleNamespace()
    data_obj.data, data_obj.targets = np.array(x.permute([0, 2, 3, 1])*255).astype(np.uint8), y.tolist()
    return data_obj


def dataset_cifar100c(*args, **kwargs):
    x, y = load_cifar100c(data_dir=args[0], n_examples=15000, severity=5, shuffle=True)
    data_obj = SimpleNamespace()
    data_obj.data, data_obj.targets = np.array(x.permute([0, 2, 3, 1])*255).astype(np.uint8), y.tolist()
    return data_obj


def get_loaders(dataset, n_ex, batch_size, split, shuffle, data_augm, val_indices=None, p_label_noise=0.0,
                noise_type='sym', drop_last=False, normalization=True, randaug=False):
    dir_ = '/tmlscratch/andriush/data'  
    # dir_ = '/tmldata1/andriush/data'
    dataset_f = datasets_dict[dataset]
    batch_size = n_ex if n_ex < batch_size and n_ex != -1 else batch_size
    num_workers_train, num_workers_val, num_workers_test = 4, 4, 4

    base_transforms = [transforms.ToPILImage()] if dataset != 'gaussians_binary' else []
    data_augm_transforms = []
    if randaug:
        data_augm_transforms.append(RandAugment(2, 14))
    data_augm_transforms.append(transforms.RandomCrop(32, padding=4))
    if dataset not in ['mnist', 'svhn']:
        data_augm_transforms.append(transforms.RandomHorizontalFlip())
    transform_list = base_transforms + data_augm_transforms if data_augm else base_transforms
    transform_list += [transforms.ToTensor()]
    
    if normalization and 'cifar10' in dataset:
        transform_list.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
    elif normalization and 'cifar100' in dataset:
        transform_list.append(transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)))

    transform = transforms.Compose(transform_list)

    if dataset == 'cifar10_horse_car':
        cl1, cl2 = 7, 1  # 7=horse, 1=car
    elif dataset == 'cifar10_dog_cat':
        cl1, cl2 = 5, 3  # 5=dog, 3=cat
    if split in ['train', 'val']:
        if dataset != 'svhn':
            data = dataset_f(dir_, train=True, transform=transform, download=True)
        else:
            data = dataset_f(dir_, split='train', transform=transform, download=True)
            data.data = data.data.transpose([0, 2, 3, 1])
            data.targets = data.labels
        data.targets = np.array(data.targets)
        n_cls = max(data.targets) + 1

        if dataset in ['cifar10_horse_car', 'cifar10_dog_cat']:
            data.targets = np.array(data.targets)
            idx = (data.targets == cl1) + (data.targets == cl2)
            data.data, data.targets = data.data[idx], data.targets[idx]
            data.targets[data.targets == cl1], data.targets[data.targets == cl2] = 0, 1
            n_cls = 2
        n_ex = len(data.targets) if n_ex == -1 else n_ex
        if '_gs' in dataset:
            data.data = data.data.mean(3).astype(np.uint8)

        if val_indices is not None:
            assert len(val_indices) < len(data.targets), '#val has to be < total #train pts'
            val_indices_mask = np.zeros(len(data.targets), dtype=bool)
            val_indices_mask[val_indices] = True
            if split == 'train':
                data.data, data.targets = data.data[~val_indices_mask], data.targets[~val_indices_mask]
            else:
                data.data, data.targets = data.data[val_indices_mask], data.targets[val_indices_mask]
        data.data, data.targets = data.data[:n_ex], data.targets[:n_ex]  # so the #pts can be in [n_ex-n_eval, n_ex]
        # e.g., when frac_train=1.0, for training set, n_ex=50k while data.data.shape[0]=45k bc of val set
        if n_ex > data.data.shape[0]:
            n_ex = data.data.shape[0]

        data.label_noise = np.zeros(n_ex, dtype=bool)
        data.targets_correct = data.targets.copy()
        if p_label_noise > 0.0:
            print('Split: {}, number of examples: {}, noisy examples: {}'.format(split, n_ex, int(n_ex*p_label_noise)))
            print('Dataset shape: x is {}, y is {}'.format(data.data.shape, data.targets.shape))
            assert n_ex == data.data.shape[0]  # there was a mistake previously here leading to a larger noise level

            # gen random indices
            indices = np.random.permutation(np.arange(len(data.targets)))[:int(n_ex*p_label_noise)]
            for index in indices:
                lst_classes = list(range(n_cls))
                cls_int = data.targets[index] if type(data.targets[index]) is int else data.targets[index].item()
                lst_classes.remove(cls_int)
                data.targets[index] = np.random.choice(lst_classes)
            data.label_noise[indices] = True
        print(data.data.shape)
        data = DatasetWithLabelNoise(data, split, transform if dataset != 'gaussians_binary' else None)
        loader = torch.utils.data.DataLoader(
            dataset=data, batch_size=batch_size, shuffle=shuffle, pin_memory=True,
            num_workers=num_workers_train if split == 'train' else num_workers_val, drop_last=drop_last)

    elif split == 'test':
        if dataset != 'svhn':
            data = dataset_f(dir_, train=False, transform=transform, download=True)
        else:
            data = dataset_f(dir_, split='test', transform=transform, download=True)
            data.data = data.data.transpose([0, 2, 3, 1])
            data.targets = data.labels
        n_ex = len(data) if n_ex == -1 else n_ex

        if dataset in ['cifar10_horse_car', 'cifar10_dog_cat']:
            data.targets = np.array(data.targets)
            idx = (data.targets == cl1) + (data.targets == cl2)
            data.data, data.targets = data.data[idx], data.targets[idx]
            data.targets[data.targets == cl1], data.targets[data.targets == cl2] = 0, 1
            data.targets = list(data.targets)  # to reduce memory consumption
        if '_gs' in dataset:
            data.data = data.data.mean(3).astype(np.uint8)
        data.data, data.targets = data.data[:n_ex], data.targets[:n_ex]
        data.targets_correct = data.targets.copy()

        data.label_noise = np.zeros(n_ex)
        data = DatasetWithLabelNoise(data, split, transform if dataset != 'gaussians_binary' else None)
        loader = torch.utils.data.DataLoader(dataset=data, batch_size=batch_size, shuffle=shuffle, pin_memory=True,
                                             num_workers=num_workers_test, drop_last=drop_last)

    else:
        raise ValueError('wrong split')

    return loader


def create_loader(x, y, ln, n_ex, batch_size, shuffle, drop_last):
    if n_ex > 0:
        x, y, ln = x[:n_ex], y[:n_ex], ln[:n_ex]
    data = td.TensorDataset(x, y, ln)
    loader = torch.utils.data.DataLoader(dataset=data, batch_size=batch_size, shuffle=shuffle, pin_memory=False,
                                         num_workers=2, drop_last=drop_last)
    return loader


shapes_dict = {'mnist': (60000, 1, 28, 28),
               'mnist_binary': (13007, 1, 28, 28),
               'svhn': (73257, 3, 32, 32),
               'cifar10': (50000, 3, 32, 32),
               'cifar10_horse_car': (10000, 3, 32, 32),
               'cifar10_dog_cat': (10000, 3, 32, 32),
               'cifar100': (50000, 3, 32, 32),
               }
np.random.seed(0)
datasets_dict = {'mnist': datasets.MNIST,
                 'mnist_binary': datasets.MNIST,
                 'svhn': datasets.SVHN,
                 'cifar10': datasets.CIFAR10,
                 'cifar10_horse_car': datasets.CIFAR10,
                 'cifar10_dog_cat': datasets.CIFAR10,
                 'cifar10c': dataset_cifar10c,
                 'cifar10c_binary': dataset_cifar10c,
                 'cifar100': datasets.CIFAR100,
                 'cifar100c': dataset_cifar100c
                 }
classes_dict = {'cifar10': {0: 'airplane',
                            1: 'automobile',
                            2: 'bird',
                            3: 'cat',
                            4: 'deer',
                            5: 'dog',
                            6: 'frog',
                            7: 'horse',
                            8: 'ship',
                            9: 'truck',
                            }
                }

