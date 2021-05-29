import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
from datetime import datetime
import os
from os.path import join
import random
from tqdm import tqdm
import json
from collections import OrderedDict
import multiprocessing as mp

import models
from utils import Logger, model_test
from data import get_dataloader

from hessian_eigenthings import compute_hessian_eigenthings

torch.autograd.set_detect_anomaly(True)

def main():
    parser = argparse.ArgumentParser(description='Geometrical Generalization Bound Based on FBM')
    parser.add_argument('--arch', default='fc', type=str)
    parser.add_argument('--depth', default=3, type=int)
    parser.add_argument('--width', default=64, type=int)
    parser.add_argument('--scale', default=8, type=int)
    parser.add_argument('--dataset', default='cifar10', type=str)
    parser.add_argument('--num_channels', default=3, type=int)
    parser.add_argument('--momentum', default=0.0, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=0.0, type=float)
    parser.add_argument('--num_classes', default=10, type=int)
    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--save_dir', default='./runs', type=str)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--subset', type=float, default=1.0, help='percentage of label to be randomized')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--root', default='../data', type=str)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--gpu_id', default=0, type=int)
    parser.add_argument('--verbose', default=0, type=int)
    parser.add_argument('--log_interval', default=500, type=int)
    parser.add_argument('--eigen', default=0, type=int, help='compute the eigenvalues of the Hessian matrix')
    parser.add_argument('--num_eigen', default=1, type=int, help='top num_eigen eigenvalues of the Hessian matrix')
    parser.add_argument('--distance', default=1, type=int, help='compute the distance from initialization')
    parser.add_argument('--ratio', default=0, type=float, help='fraction of randomized labels')
    parser.add_argument('--epochs', default=100, type=int, help='number of training epochs')
    parser.add_argument('--stopping_criterion', default=0.01, type=float, help='early stopping')
    parser.add_argument('--repeat', default=1, type=int, help='number of times to estimate H')
    args = parser.parse_args()
    # print(json.dumps(args.__dict__, indent=4, sort_keys=True))
    print(args.save_dir)
    # args.seed = datetime.now().microsecond

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    with open(join(args.save_dir, 'config.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=4)
    args.device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    loss_function = nn.functional.cross_entropy
    train_loader, test_loader, num_classes = get_dataloader(args)
    args.num_classes = num_classes

    if not hasattr(models, args.arch):
        raise NotImplementedError(f'{args.arch} is not implemented now!!')
    else:
        model_class = getattr(models, args.arch)
    if args.arch == 'fc':
        model = model_class(input_dim=32*32*args.num_channels, num_classes=args.num_classes, width=args.width, depth=args.depth).to(args.device)
    elif args.arch == 'alexnet':
        model = model_class(num_classes=args.num_classes, ch=args.scale).to(args.device)
    else:
        model = model_class(num_classes=args.num_classes).to(args.device)
        
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    logger = Logger()
    params = OrderedDict(model.named_parameters())
    if args.distance:
        dist = []
        init_weight = torch.cat([params[key].data.flatten() for key in params])
    pbar = tqdm(range(args.epochs), total=args.epochs, ascii=True)
    for i in pbar:
        model.train()
        for _, (data, target) in enumerate(train_loader):
            data, target = data.to(args.device), target.to(args.device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_function(output, target)
            loss.backward()
            optimizer.step()
        train_acc, train_loss, _ = model_test(model, train_loader, args)
        test_acc, test_loss, _ = model_test(model, test_loader, args)
        logger.basic_update(train_loss, test_loss, train_acc, test_acc)
        pbar.set_description(f'Epoch {i+1}, train_loss: {train_loss:.4f}, test_loss: {test_loss:.4f}, train_acc: {train_acc:.4f}, test_acc: {test_acc:.4f}')
        if args.distance:
            weight = torch.cat([params[key].data.flatten() for key in params])
            dist.append(torch.norm(weight-init_weight).item())
            torch.save(dist, join(args.save_dir, 'dist.pt'))
        if train_loss < args.stopping_criterion:
            break
    torch.save(model.state_dict(), join(args.save_dir, 'model.pt'))
    if args.eigen:
        eigenvals, _ = compute_hessian_eigenthings(model, train_loader, loss_function, args.num_eigen)
        logger.largest_eigenvalue = eigenvals
    torch.save(logger, join(args.save_dir, 'logger.pt'))

if __name__ == '__main__':
    main()