import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from torchvision import datasets, transforms

import numpy as np
import random

def fix_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    

def batchnorm_callibration(model, train_loader, n_callibration_batches = 200000//512, 
                           layer_name = None, device="cuda:0"):
    '''
    Update batchnorm statistics for layers after layer_name

    Parameters:

    model                   -   Pytorch model
    train_loader            -   Training dataset dataloader, Pytorch Dataloader
    n_callibration_batches  -   Number of batchnorm callibration iterations, int
    layer_name              -   Name of layer after which to update BN statistics, string or None
                                (if None updates statistics for all BN layers)
    device                  -   Device to store the model, string
    '''
    
    # switch batchnorms into the mode, in which its statistics are updated
    model.to(device).train() 

    if layer_name is not None:
        #freeze batchnorms before replaced layer
        for lname, l in model.named_modules():

            if lname == layer_name:
                break
            else:
                if (isinstance(l, nn.BatchNorm2d)):
                    l.eval()

    with torch.no_grad():            

        for i, (data, _) in enumerate(train_loader):
            _ = model(data.to(device))

            if i > n_callibration_batches:
                break
            
        del data
        torch.cuda.empty_cache()
        
    model.eval()
    return model


def get_layer_by_name(model, mname):
    '''
    Extract layer using layer name
    '''
    module = model
    mname_list = mname.split('.')
    for mname in mname_list:
        module = module._modules[mname]

    return module


def replace_layer_by_name(model, mname, new_layer):
    '''
    Replace layer using layer name
    '''
    module = model
    mname_list = mname.split('.')
    for mname in mname_list[:-1]:
        module = module._modules[mname]
    module._modules[mname_list[-1]] = new_layer
    
def calc_accuracy(target, output, top_k = (1, 5)):
    '''
    Calculate top k accuracy for classification task
    '''
    
    with torch.no_grad():
        max_k = max(top_k)
        res = []

        _, pred = output.topk(max_k, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))


        for k in top_k:
            res.append(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu())
    
    return np.array(res)

def get_validation_scores(model, val_loader, top_k = (1, 5), val_iterations = None, device = "cpu"):
    """
    Computes the top k classification accuracy for the specified values of k
    
    args:
    
    model           - pytorch NN model,
    val_folder_path - path to validation folder,
    top_n           - tuple of top n for validation
    
    output:
    accuracy scores (tuple)
    """
    
    loader = val_loader
    
    counter = 0
    cum_sum = np.array([0] * len(top_k))
    model.eval()
    if val_iterations is None:
        total = len(loader)
    else:
        total = np.min((len(loader), val_iterations))
    
    for i, (input, target) in enumerate(loader):
        with torch.no_grad():
            counter += target.size(0)
            
            input = input.to(device)
            target = target.to(device)
            output = model(input)

            res = calc_accuracy(target, output, top_k = top_k)
            for i in range(len(res)):
                cum_sum[i] += res[i]
                
            if counter == val_iterations:
                break
        
    return cum_sum / counter

def train(model, device, train_loader, optimizer, epoch, log_interval, verbose):
    
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if verbose and batch_idx % log_interval == 0:
            print(f'Train Epoch: {epoch} '
                  f'[{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]'
                  f'\tLoss: {loss.item():.6f}')


def get_cifar100_dataloader(dataset_path, batch_size, num_workers, download=False):

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(dataset_path, train=True, download=download,
                          transform=transforms.Compose([
                              transforms.RandomCrop(32, padding=4),
                              transforms.RandomHorizontalFlip(),
                              transforms.ToTensor(),
                              transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                          ])),
        batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(dataset_path, train=False, download=download,
                          transform=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                          ])),
        batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader