import argparse
import csv
import os
import random
import sys
from datetime import datetime

import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import trange

import flops_benchmark
from clr import CyclicLR
from data import get_loaders
from logger import CsvLogger
from model import MobileNet2
from run import train, test, save_checkpoint, find_bounds_clr

claimed_acc_top1 = {224: {1.4: 0.75, 1.3: 0.744, 1.0: 0.718, 0.75: 0.698, 0.5: 0.654, 0.35: 0.603},
                    192: {1.0: 0.707, 0.75: 0.687, 0.5: 0.639, 0.35: 0.582},
                    160: {1.0: 0.688, 0.75: 0.664, 0.5: 0.610, 0.35: 0.557},
                    128: {1.0: 0.653, 0.75: 0.632, 0.5: 0.577, 0.35: 0.508},
                    96: {1.0: 0.603, 0.75: 0.588, 0.5: 0.512, 0.35: 0.455},
                    }
claimed_acc_top5 = {224: {1.4: 0.925, 1.3: 0.921, 1.0: 0.910, 0.75: 0.896, 0.5: 0.864, 0.35: 0.829},
                    192: {1.0: 0.901, 0.75: 0.889, 0.5: 0.854, 0.35: 0.812},
                    160: {1.0: 0.890, 0.75: 0.873, 0.5: 0.832, 0.35: 0.791},
                    128: {1.0: 0.869, 0.75: 0.855, 0.5: 0.808, 0.35: 0.750},
                    96: {1.0: 0.832, 0.75: 0.816, 0.5: 0.758, 0.35: 0.704},
                    }


def train_network(start_epoch, epochs, scheduler, model, train_loader, val_loader, optimizer, criterion, device, dtype,
                  batch_size, log_interval, csv_logger, save_path, claimed_acc1, claimed_acc5, best_test):
    for epoch in trange(start_epoch, epochs + 1):
        if not isinstance(scheduler, CyclicLR):
            scheduler.step()
        train_loss, train_accuracy1, train_accuracy5, = train(model, train_loader, epoch, optimizer, criterion, device,
                                                              dtype, batch_size, log_interval, scheduler)
        test_loss, test_accuracy1, test_accuracy5 = test(model, val_loader, criterion, device, dtype)
        csv_logger.write({'epoch': epoch + 1, 'val_error1': 1 - test_accuracy1, 'val_error5': 1 - test_accuracy5,
                          'val_loss': test_loss, 'train_error1': 1 - train_accuracy1,
                          'train_error5': 1 - train_accuracy5, 'train_loss': train_loss})
        save_checkpoint({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'best_prec1': best_test,
                         'optimizer': optimizer.state_dict()}, test_accuracy1 > best_test, filepath=save_path)

        csv_logger.plot_progress(claimed_acc1=claimed_acc1, claimed_acc5=claimed_acc5)

        if test_accuracy1 > best_test:
            best_test = test_accuracy1

    csv_logger.write_text('Best accuracy is {:.2f}% top-1'.format(best_test * 100.))
    

def main():
	
    seed = random.randint(1, 10000)
    random.seed(seed)
    torch.manual_seed(seed)
    
    torch.cuda.manual_seed_all(seed)

    time_stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    results_dir = '/tmp'
    save = time_stamp
    save_path = os.path.join(results_dir, save)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    gpus = 2#[int(i) for i in gpus.split(',')]
    device = 'cuda:0' #+ str(args.gpus[0])
    cudnn.benchmark = True
    dtype = torch.float64

    input_size = 224
    scaling = 1.0
    batch_size = 20
    workers = 4
    learning_rate = 0.02
    momentum = 0.9
    decay = 0.00004
    max_lr = 1
    min_lr = 0.00001
    start_epoch = 0
    epochs = 400
    epochs_per_step = 20
    log_interval = 100
    mode = 'triangular2'
    evaluate = 'false'
    dataroot = "data"


    model = MobileNet2(input_size=input_size, scale=scaling)
    num_parameters = sum([l.nelement() for l in model.parameters()])
    #print(model)


    """print('number of parameters: {}'.format(num_parameters))
    print('FLOPs: {}'.format(
        flops_benchmark.count_flops(MobileNet2,
                                    batch_size // len(gpus) if gpus is not None else batch_size,
                                    device, dtype, input_size, 3, scaling)))"""

    train_loader, val_loader = get_loaders(dataroot, batch_size, batch_size, input_size, workers)

    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    model = torch.nn.DataParallel(model)
        
    model.to(device=device, dtype=dtype)
    criterion.to(device=device, dtype=dtype)

    optimizer = torch.optim.SGD(model.parameters(), learning_rate, momentum=momentum, weight_decay=decay,
                                nesterov=True)
    find_bounds_clr(model, train_loader, optimizer, criterion, device, dtype, min_lr=min_lr,
                        max_lr=max_lr, step_size=epochs_per_step * len(train_loader), mode=mode,
                        save_path=save_path)
    scheduler = CyclicLR(optimizer, base_lr=min_lr, max_lr=max_lr,
                             step_size=epochs_per_step * len(train_loader), mode=mode)
    
    best_test = 0

 
    if evaluate == 'true':
        loss, top1, top5 = test(model, val_loader, criterion, device, dtype)  # TODO
        return

    data = []

    csv_logger = CsvLogger(filepath=save_path, data=data)
    #csv_logger.save_params(sys.argv, args)

    claimed_acc1 = None
    claimed_acc5 = None
    if input_size in claimed_acc_top1:
        if scaling in claimed_acc_top1[input_size]:
            claimed_acc1 = claimed_acc_top1[input_size][scaling]
            claimed_acc5 = claimed_acc_top5[input_size][scaling]
            csv_logger.write_text(
                'Claimed accuracies are: {:.2f}% top-1, {:.2f}% top-5'.format(claimed_acc1 * 100., claimed_acc5 * 100.))
            
            
    train_network(start_epoch, epochs, scheduler, model, train_loader, val_loader, optimizer, criterion,
                  device, dtype, batch_size, log_interval, csv_logger, './data', claimed_acc1, claimed_acc5,
                  best_test)
    
    return 1


