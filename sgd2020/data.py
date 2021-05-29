import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
from os.path import join

data_list = {'mnist': 'MNIST', 'kmnist': 'KMNIST', 'fashion_mnist': 'FashionMNIST', 'cifar10': 'CIFAR10', 'cifar100': 'CIFAR100','svhn': 'SVHN', 'mnist_c': 'MNISTC'}

def get_dataloader(args):
    num_classes = 10
    if args.dataset == 'cifar100':
        num_classes = 100
    # Transformations
    RC   = transforms.RandomCrop(32, padding=4)
    RHF  = transforms.RandomHorizontalFlip()
    RVF  = transforms.RandomVerticalFlip()
    NRM  = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    TT   = transforms.ToTensor()
    # TPIL = transforms.ToPILImage()
    # Transforms object for trainset with augmentation
    # transform_with_aug = transforms.Compose([RC, TT, NRM])
    transform_with_aug = transforms.Compose([transforms.Resize(32), TT, NRM])
    # Transforms object for testset with NO augmentation
    transform_no_aug   = transforms.Compose([transforms.Resize(32), TT, NRM])
    all_transforms = transforms.Compose([transforms.Resize(32), transforms.ToTensor()])

    if args.ratio == 0.0:
        if args.dataset == 'mnist_c':
            data_class = MNISTC
        else:
            data_class = getattr(datasets, data_list[args.dataset])
    else:
        data_class = label_randomize(args)
    
    if args.dataset == 'mnist_c':
        all_transforms_c = transforms.Compose([transforms.ToPILImage(), transforms.Resize(32), transforms.ToTensor()])
        train_data = data_class(root=join(args.root, 'mnist_corrupted'), corrupt_type=args.corrupted_type, train=True, transform=all_transforms_c)
        test_data = data_class(root=join(args.root, 'mnist_corrupted'), corrupt_type=args.corrupted_type, train=False, transform=all_transforms_c)
    elif args.dataset == 'svhn':
        train_data = data_class(root=join(args.root, f'{args.dataset}'), split='train', download=True, transform=all_transforms)
        test_data = data_class(root=join(args.root, f'{args.dataset}'), split='test', download=True, transform=all_transforms)
    elif 'mnist' in args.dataset:
        train_data = data_class(root=join(args.root, f'{args.dataset}'), train=True, download=True, transform=all_transforms)
        test_data = data_class(root=join(args.root, f'{args.dataset}'), train=False, transform=all_transforms)
    elif 'cifar' in args.dataset:
        train_data = data_class(root=join(args.root, f'{args.dataset}'), train=True, download=True, transform=transform_with_aug)
        test_data = data_class(root=join(args.root, f'{args.dataset}'), train=False, transform=transform_no_aug)
    else:
        raise NotImplementedError(f'{args.dataset} not implemented!!')
 
    train_ids = torch.randperm(len(train_data))
    train_set = torch.utils.data.Subset(train_data, train_ids[:int(args.subset*len(train_data))])
    # test_ids = torch.randperm(len(test_data))
    # test_set = torch.utils.data.Subset(test_data, test_ids[:int(args.subset*len(test_data))])
    test_set = test_data

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    
    return train_loader, test_loader, num_classes

class MNISTC(Dataset):
    """D corrupted MNIST dataset."""
    def __init__(self, root, subsample=1, transform=None, corrupt_type='fog', train=True):
        """
        Parameters
        ----------
        subsample : int
            Only load every |subsample| number of images.
        """
        if train == True:
            self.data = np.load(join(root, corrupt_type, 'train_images.npy'))[::subsample]
            self.targets = np.load(join(root, corrupt_type, 'train_labels.npy'))[::subsample]
        else:
            self.data = np.load(join(root, corrupt_type, 'test_images.npy'))[::subsample]
            self.targets = np.load(join(root, corrupt_type, 'test_labels.npy'))[::subsample]
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.targets[idx]
        if self.transform:
            sample = self.transform(sample)
        # print(sample[0])
        return sample, label

def label_randomize(args):
    if args.dataset == 'mnist_c':
        CLS = MNISTC
    else:
        CLS = getattr(datasets, data_list[args.dataset])
    class RandomDataSet(CLS):
        """support for randomly corrupt labels.
        Params
        ------
        corrupt_prob: float
            Default 0.0. The probability of a label being replaced with
            random label.
        num_classes: int
            Default 10. The number of classes in the dataset.
        """
        def __init__(self, corrupt_prob=args.ratio, num_classes=args.num_classes, **kwargs):
            super(RandomDataSet, self).__init__(**kwargs)
            self.n_classes = num_classes
            if corrupt_prob > 0:
                self.corrupt_labels(corrupt_prob)

        def corrupt_labels(self, corrupt_prob):
            if args.dataset == 'svhn':
                labels = np.array(self.labels)
            else:
                labels = np.array(self.targets)
            np.random.seed(12345)
            mask = np.random.rand(len(labels)) <= corrupt_prob
            rnd_labels = np.random.choice(self.n_classes, mask.sum())
            labels[mask] = rnd_labels
            # we need to explicitly cast the labels from npy.int64 to
            # builtin int type, otherwise pytorch will fail...
            labels = [int(x) for x in labels]

            if args.dataset == 'svhn':
                self.labels = labels
            else:
                self.targets = labels
    return RandomDataSet