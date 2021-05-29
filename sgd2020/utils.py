import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from os.path import join


class Logger():
    def __init__(self):
        self.train_loss = np.array([])
        self.test_loss = np.array([])
        self.train_acc = np.array([])
        self.test_acc = np.array([])
        self.hurst = -1.0
        self.largest_eigenvalue = -1.0
    def basic_update(self, train_loss, test_loss, train_acc, test_acc):
        self.train_loss = np.append(self.train_loss, train_loss)
        self.test_loss = np.append(self.test_loss, test_loss)
        self.train_acc = np.append(self.train_acc, train_acc)
        self.test_acc = np.append(self.test_acc, test_acc)
    def get_best_acc(self):
        if len(self.train_acc) >= 1:
            return max(self.train_acc), max(self.test_acc)
        else:
            raise RuntimeError('No values logged until now!!')
    def get_current_acc(self):
        if len(self.train_acc) >= 1:
            return self.train_acc[-1], self.test_acc[-1]
        else:
            raise RuntimeError('No values logged until now!!')

def cycle_loader(data_loader):
    while 1:
        for _, data in enumerate(data_loader):
            yield data

def hist(data=None, bins=20, color='green', cmap='flag'):
    _, _, patches = plt.hist(data, bins, color=color)
    cm = plt.cm.get_cmap(cmap)
    for i, p in enumerate(patches):
        plt.setp(p, 'facecolor', cm(i/bins))
    plt.show()

def get_grads(model): 
    # wrt data at the current step
    res = []
    for p in model.parameters():
        if p.requires_grad:
            res.append(p.grad.data.flatten())
    grad_flat = torch.cat(res)
    return grad_flat

def get_param(model): 
    # wrt data at the current step
    res = []
    for p in model.parameters():
        if p.requires_grad:
            res.append(p.data.flatten())
    grad_flat = torch.cat(res)
    return grad_flat, len(grad_flat)

def model_test(model, eval_loader, args):
    model.eval()
    loss_function = nn.CrossEntropyLoss(reduction='none')
    test_loss = 0.0
    correct = 0.0
    loss_vec = []
    with torch.no_grad():
        for data, target in eval_loader:
            data, target = data.to(args.device), target.to(args.device)
            output = model(data)
            loss = loss_function(output, target)
            loss_vec.append(loss.data.cpu().numpy())
            test_loss += loss.sum()
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum()
    loss_vec = np.concatenate(loss_vec)
    return correct.item() / len(eval_loader.dataset), test_loss.item() / len(eval_loader.dataset), loss_vec

