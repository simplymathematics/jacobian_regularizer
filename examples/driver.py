'''
    Copyright (c) Facebook, Inc. and its affiliates.

    This source code is licensed under the MIT license found in the
    LICENSE file in the root directory of this source tree.
    
    Example script training a simple MLP on MNIST
    demonstrating the PyTorch implementation of
    Jacobian regularization described in [1].

    [1] Judy Hoffman, Daniel A. Roberts, and Sho Yaida,
        "Robust Learning with Jacobian Regularization," 2019.
        [arxiv:1908.02729](https://arxiv.org/abs/1908.02729)
'''
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from jacobian import JacobianReg
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import pandas as pd
from pathlib import Path
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.shape) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class AddAdversarialNoise(object):
    def __init__(self, classifier, eps=0.1):
        self.eps = eps
        self.classifier = classifier
    
    def __call__(self, tensor):
        adv_crafter = FastGradientMethod(self.classifier, eps=self.eps)
        adv_tensor = adv_crafter.generate(tensor)
        return adv_tensor
    
    

class MLP(nn.Module):
    '''
    Simple MLP to demonstrate Jacobian regularization.
    '''
    def __init__(self, in_channel=1, im_size=28, n_classes=10, 
                 fc_channel1=200, fc_channel2=200):
        super(MLP, self).__init__()
        
        # Parameter setup
        compression=in_channel*im_size*im_size
        self.compression=compression
        
        # Structure
        self.fc1 = nn.Linear(compression, fc_channel1)
        self.fc2 = nn.Linear(fc_channel1, fc_channel2)
        self.fc3 = nn.Linear(fc_channel2, n_classes)
        
        # Initialization protocol
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
    
    def forward(self, x):
        x = x.view(-1, self.compression)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def eval(device, model, loader, criterion, lambda_JR):
    '''
    Evaluate a model on a dataset for Jacobian regularization

    Arguments:
        device (torch.device): specifies cpu or gpu training
        model (nn.Module): the neural network to evaluate
        loader (DataLoader): a loader for the dataset to eval
        criterion (nn.Module): the supervised loss function
        lambda_JR (float): the Jacobian regularization weight

    Returns:
        correct (int): the number correct
        total (int): the total number of examples
        loss_super (float): the supervised loss
        loss_JR (float): the Jacobian regularization loss
        loss (float): the total combined loss
    '''

    correct = 0
    total = 0 
    loss_super_avg = 0 
    loss_JR_avg = 0 
    loss_avg = 0

    # for eval, let's compute the jacobian exactly
    # so n, the number of projections, is set to -1.
    reg_full = JacobianReg(n=-1) 
    for data, targets in loader:
        data = data.to(device)
        data.requires_grad = True # this is essential!
        targets = targets.to(device)
        output = model.forward(data)
        _, predicted = torch.max(output, 1)
        correct += (predicted == targets).sum().item()
        total += targets.size(0)
        loss_super = criterion(output, targets) # supervised loss
        loss_JR = reg_full(data, output) # Jacobian regularization
        loss = loss_super + lambda_JR*loss_JR # full loss
        loss_super_avg += loss_super.item()*targets.size(0)
        loss_JR_avg += loss_JR.item()*targets.size(0)
        loss_avg += loss.item()*targets.size(0)
    loss_super_avg /= total
    loss_JR_avg /= total
    loss_avg /= total
    return correct, total, loss_super, loss_JR, loss

def main(seed = 1, batch_size = 1024, epochs = 20, lambda_JR = .1, n_proj = -1, learning_rate = 1e-3, dataset="mnist", noise = 0, test_noise = 0, adversarial=False):
    '''
    Train MNIST with Jacobian regularization.
    '''
    # number of projections, default is n_proj=1
    # should be greater than 0 and less than sqrt(# of classes)
    # by default, it is set n_proj=-1 to compute the full jacobian
    # which is computationally inefficient
    # setup devices
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.manual_seed(seed)
    else:
        device = torch.device("cpu")

    # load MNIST trainset and testset
    if dataset == "mnist":
        mnist_mean = (0.1307,)
        mnist_std = (0.3081,)
        im_size = 28
        in_channel = 1
        n_classes = 10
        train_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mnist_mean, mnist_std), AddGaussianNoise(0, noise)]
        )
        test_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mnist_mean, mnist_std), AddGaussianNoise(0, test_noise)]
        )
        adv_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mnist_mean, mnist_std), AddAdversarialNoise(0, .03)]
        )
        trainset = datasets.MNIST(root='./data', train=True, 
            download=True, transform=train_transform
        )
        testset = datasets.MNIST(root='./data', train=False, 
            download=True, transform=test_transform
        )
        advset = datasets.MNIST(root='./data', train=False, download=True, transform=adv_transform)
        model = MLP(im_size=im_size, n_classes=n_classes, in_channel=in_channel, fc_channel1=200, fc_channel2=200).to(device)
    elif dataset == "cifar10":
        cifar10_mean = (0.4914, 0.4822, 0.4465)
        cifar10_std = (0.2470, 0.2434, 0.2616)
        in_channel = 3
        im_size = 32
        n_classes = 10
        train_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(cifar10_mean, cifar10_std), AddGaussianNoise(0, noise)]
        )
        test_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(cifar10_mean, cifar10_std), AddGaussianNoise(0, test_noise)]
        )
        
        trainset = datasets.CIFAR10(root='./data', train=True, 
            download=True, transform=train_transform
        )
        testset = datasets.CIFAR10(root='./data', train=False, 
            download=True, transform=test_transform
        )
                
        # initialize the model
        model = MLP(im_size=im_size, n_classes=n_classes, in_channel=in_channel, fc_channel1=500, fc_channel2=500).to(device)
        adv_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(cifar10_mean, cifar10_std), AddAdversarialNoise(model, .03)]
        )
        advset = datasets.CIFAR10(root='./data', train=False,
            download=True, transform=adv_transform
        )
    elif dataset == "generated":
        # raise NotImplementedError("generated data not implemented")
        n_classes = 10
        in_channel = 1
        im_size = 10
        n_features=im_size*im_size*in_channel
        n_informative = int(round(n_features*0.95))
        X,y = make_classification(n_samples=1000, n_features=n_features, n_redundant=0, n_informative=n_informative, n_clusters_per_class=1, random_state=seed, class_sep = 10, n_classes =  n_classes)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
        X_train = X_train.reshape(-1, in_channel, im_size, im_size)
        X_test = X_test.reshape(-1, in_channel, im_size, im_size)
        X_train =  torch.from_numpy(X_train).float()
        X_test =  torch.from_numpy(X_test).float()
        gen_mean = torch.mean(X_train, axis=0)
        gen_std = torch.std(X_train, axis=0)
        train_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(gen_mean, gen_std)]
        )
        test_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(gen_mean, gen_std)]
        )
        trainset = torch.utils.data.TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long(), transform=train_transform)
        testset = torch.utils.data.TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long(), transform=test_transform)
    else:   
        raise ValueError("data must be either 'mnist', 'cifar10', or 'generated'")
    trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True
        )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=True
    )
        
    advloader = torch.utils.data.DataLoader(advset, batch_size=batch_size, shuffle=True)

    # initialize the loss and regularization
    criterion = nn.CrossEntropyLoss()
    reg = JacobianReg(n=n_proj) # if n_proj = 1, the argument is unnecessary

    # initialize the optimizer
    optimizer = optim.SGD(model.parameters(), 
        lr=learning_rate, momentum=0.9, weight_decay=5e-4
    )

    # eval on testset before any training
    correct_i, total, loss_super_i, loss_JR_i, loss_i = eval(
        device, model, testloader, criterion, lambda_JR
    )

    # train
    for epoch in range(epochs):
        print('Training epoch %d.' % (epoch + 1) )
        running_loss_super = 0.0
        running_loss_JR = 0.0
        for idx, (data, target) in enumerate(trainloader):        

            data, target = data.to(device), target.to(device)
            data.requires_grad = True # this is essential!

            optimizer.zero_grad()

            output = model(data) # forward pass

            loss_super = criterion(output, target) # supervised loss
            loss_JR = reg(data, output)   # Jacobian regularization
            loss = loss_super + lambda_JR*loss_JR # full loss

            loss.backward() # computes gradients

            optimizer.step()

            # print running statistics
            running_loss_super += loss_super.item()
            running_loss_JR += loss_JR.item()
            if idx % 100 == 99:    # print every 100 mini-batches
                print('[%d, %5d] supervised loss: %.9f, Jacobian loss: %.9f' %
                        (
                            epoch + 1, 
                            idx + 1, 
                            running_loss_super / 100,  
                            running_loss_JR / 100, 
                        )
                )
                running_loss_super = 0.0
                running_loss_JR = 0.0

    # eval on testset after training
    correct_f, total, loss_super_f, loss_JR_f, loss_f = eval(
        device, model, testloader, criterion, lambda_JR
    )

    # print results
    print(f'\n Test set results for {dataset} dataset with Lambda={lambda_JR}:')
    print('Before training:')
    print('\taccuracy: %d/%d=%.9f' % (correct_i, total, correct_i/total))
    print('\tsupervised loss: %.9f' % loss_super_i)
    print('\tJacobian loss: %.9f' % loss_JR_i)
    print('\ttotal loss: %.9f' % loss_i)

    print('\nAfter %d epochs of training:' % epochs)
    print('\taccuracy: %d/%d=%.9f' % (correct_f, total, correct_f/total))
    print('\tsupervised loss: %.9f' % loss_super_f)
    print('\tJacobian loss: %.9f' % loss_JR_f)
    print('\ttotal loss: %.9f' % loss_f)
    results = {
        "accuracy": correct_f/total,
        "supervised loss": loss_super_f.item(),
        "Jacobian loss": loss_JR_f.item(),
        "total loss": loss_f.item(),
        "lambda_JR": lambda_JR,
        "dataset": dataset,
        "n_proj": n_proj,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "noise": noise,
        "test_noise": test_noise,
    }
    if adversarial is True:
        clip_values = (0, 255)
        model = PyTorchClassifier(model, clip_values=clip_values, loss=criterion, optimizer=optimizer, input_shape=(in_channel, im_size, im_size), nb_classes=n_classes)
        attack = FastGradientMethod(model, eps=0.03)
        advs = []
        for test_img in testloader:
            x_test_adv = attack.generate(x=test_img)
            advs.append(x_test_adv)
        # eval on testset before any training
        correct_adv, total, loss_super_adv, loss_JR_adv, loss_adv = eval(
            device, model, x_test_adv, criterion, lambda_JR
        )
        results.update({
            "adversarial accuracy": correct_adv/total,
            "adversarial supervised loss": loss_super_adv.item(),
            "adversarial Jacobian loss": loss_JR_adv.item(),
            "adversarial total loss": loss_adv.item(),
        })
    return results

if __name__ == '__main__':
    for dataset in [ "cifar10", "mnist"]:
        print("*"*80)
        print("Evaluating on %s" % dataset)
    ## Jacobian regularization parameter testing
        for lambda_JR  in [0.0, 1e-9, 1e-8, .0000001, .000001, .00001,  .0001, .001, .01, 0.1, 1, 10, 100]:
            filename = Path(dataset) / "scales.csv"
            if filename.exists():
                df = pd.read_csv(filename)
            else:
                df = pd.DataFrame(columns=["accuracy", "supervised loss", "Jacobian loss", "total loss", "lambda_JR"])
            if lambda_JR in df['lambda_JR'].values:
                print("Skipping lambda_JR=%f" % lambda_JR)
            else:
                print("Running lambda_JR=%f" % lambda_JR)
                results = main(lambda_JR=lambda_JR, epochs=20, dataset=dataset)
                df = df.append(results, ignore_index=True)
                df.sort_values(by="lambda_JR", inplace=True)
                df.to_csv(filename, index=False)
    ## Training noise testing
        for noise in [0.0, 1e-9, 1e-8, .0000001, .000001, .00001,  .0001, .001, .01, 0.1, 1, 10, 100]:
            filename = Path(dataset) / "noise.csv"
            if filename.exists():
                df = pd.read_csv(filename)
            else:
                df = pd.DataFrame()
            if "noise" in df.columns and noise in df['noise'].values:
                print("Skipping noise=%f" % noise)
            else:
                print("Running noise=%f" % noise)
                results = main(noise=noise, epochs=20, dataset=dataset, lambda_JR=0.0)
                df = df.append(results, ignore_index=True)
                df.sort_values(by="noise", inplace=True)
                df.to_csv(filename, index=False)
    ## Testing noise testing
        for noise in [0.0, 1e-9, 1e-8, .0000001, .000001, .00001,  .0001, .001, .01, 0.1, 1, 10, 100]:
            filename = Path(dataset) / "test_noise.csv"
            if filename.exists():
                df = pd.read_csv(filename)
            else:
                df = pd.DataFrame()
            if "test_noise" in df.columns and noise in df['test_noise'].values:
                print("Skipping noise=%f" % noise)
            else:
                print("Running test noise=%f" % noise)
                results = main(test_noise=noise, epochs=20, dataset=dataset, lambda_JR=1e-7)
                df = df.append(results, ignore_index=True)
                df.sort_values(by="test_noise", inplace=True)
                df.to_csv(filename, index=False)
    ## Testing adversarial noise
        filename = Path(dataset) / "adversarial.csv"
        if filename.exists():
            df = pd.read_csv(filename)
        else:
            df = pd.DataFrame()
        if "adversarial_accuracy" in df.columns:
            print("Adversaries already tested")
        else:
            results = main(epochs=1, dataset=dataset, lambda_JR=1e-7, adversarial=True)
            df = df.append(results, ignore_index=True)
            df.sort_values(by="adversarial", inplace=True)
            df.to_csv(filename, index=False)
    
        
        
            
            
                
                
            