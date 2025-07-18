import torch
import torch.nn.functional as F
from torch.autograd import Variable
from utils import AverageMeter
import numpy as np



def train_Nirvana_oe(net, criterion, optimizer, scheduler, trainloader, trainloader_oe, epoch=None, **options):
    net.train()
    losses = AverageMeter()
    torch.cuda.empty_cache()
    loss_all = 0
    trainloader_oe.dataset.offset = np.random.randint(len(trainloader_oe.dataset))
    
    for batch_idx, (in_set, out_set) in enumerate(zip(trainloader, trainloader_oe)):
        inputs_inout = torch.cat((in_set[0], out_set[0]), 0)
        targets_in = in_set[1]
        if options['use_gpu']:
            inputs_inout, targets_in = inputs_inout.cuda(), targets_in.cuda()
        
        optimizer.zero_grad()
        
        x, y = net(inputs_inout, True)
        intraclass_loss, triplet_loss, outlier_triplet_loss = criterion(
            targets_in, 
            x[:len(in_set[0])], 
            x[len(in_set[0]):], 
            ramp=options['ramp_activate']
        )
        total_loss = intraclass_loss + triplet_loss + outlier_triplet_loss
        total_loss.backward()
        optimizer.step()
        scheduler.step()
        
        losses.update(total_loss.item(), targets_in.size(0))

        if (batch_idx+1) % options['print_freq'] == 0:
            print("LR: {} - Batch {}/{}\t Loss {:.6f} ({:.6f})" \
                  .format(optimizer.param_groups[0]["lr"], batch_idx+1, len(trainloader), losses.val, losses.avg))
        
        loss_all += losses.avg
    print("Epoch loss: {}".format(loss_all))
    return loss_all

def train_Nirvana_oe_reg(net, criterion, optimizer, scheduler, trainloader, trainloader_oe=None, epoch=None, **options):
    net.train()
    losses = AverageMeter()
    torch.cuda.empty_cache()
    loss_all = 0
    trainloader_oe.dataset.offset = np.random.randint(len(trainloader_oe.dataset))

    
    for batch_idx, (in_set, out_set) in enumerate(zip(trainloader, trainloader_oe)):
        inputs_inout = torch.cat((in_set[0], out_set[0]), 0)
        targets_in = in_set[1]
        if options['use_gpu']:
            inputs_inout, targets_in = inputs_inout.cuda(), targets_in.cuda()
        
        optimizer.zero_grad()
        
        x, y = net(inputs_inout, True)
        intraclass_loss, triplet_loss, outlier_triplet_loss, uncertainty_loss = criterion(
            targets_in, 
            x[:len(in_set[0])], 
            x[len(in_set[0]):], 
            ramp=options['ramp_activate']
        )
        total_loss = intraclass_loss + triplet_loss + outlier_triplet_loss + uncertainty_loss

        # Add L1 regularization if specified
        if options.get('l1_weight', 0) > 0:
            l1_penalty = sum(p.abs().sum() for p in net.parameters())
            total_loss = total_loss + options['l1_weight'] * l1_penalty
            
        total_loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()
        
        losses.update(total_loss.item(), targets_in.size(0))

        if (batch_idx+1) % options['print_freq'] == 0:
            print("LR: {} - Batch {}/{}\t Loss {:.6f} ({:.6f})" \
                  .format(optimizer.param_groups[0]["lr"], batch_idx+1, 
                         len(trainloader), losses.val, losses.avg))
        
        loss_all += losses.avg
    print("Epoch loss: {}".format(loss_all))
    return loss_all

def train_ddfm_oe(net, criterion, optimizer, optimizer_center, scheduler,trainloader, trainloader_oe, epoch=None, **options):
    net.train()
    losses = AverageMeter()

    torch.cuda.empty_cache()
    
    loss_all = 0
    trainloader_oe.dataset.offset = np.random.randint(len(trainloader_oe.dataset))
    for batch_idx, (in_set, out_set) in enumerate(zip(trainloader,trainloader_oe)):
        inputs_inout = torch.cat((in_set[0],out_set[0]),0)
        targets_in = in_set[1]
        if options['use_gpu']:
            inputs_inout, targets_in = inputs_inout.cuda(), targets_in.cuda()
        
        with torch.set_grad_enabled(True):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            x, y = net(inputs_inout, True)
            intraclass_loss, triplet_loss, outlier_triplet_loss = criterion(targets_in, x[:len(in_set[0])], x[len(in_set[0]):], ramp=options['ramp_activate'])
            total_loss = intraclass_loss + triplet_loss + outlier_triplet_loss
            # total_loss = criterion(x[:len(in_set[0])], targets_in)

            total_loss.backward()
            optimizer.step()
            optimizer_center.step()
            scheduler.step()
        losses.update(total_loss.item(), targets_in.size(0))

        if (batch_idx+1) % options['print_freq'] == 0:
            print("LR: {} - Batch {}/{}\t Loss {:.6f} ({:.6f})" \
                  .format(optimizer.param_groups[0]["lr"],batch_idx+1, len(trainloader), losses.val, losses.avg))
        
        loss_all += losses.avg
    print("Epoch loss: {}".format(loss_all))
    return loss_all

def train(net, criterion, optimizer, scheduler, trainloader, epoch=None, **options):
    net.train()
    losses = AverageMeter()

    torch.cuda.empty_cache()
    
    loss_all = 0
    for batch_idx, (data, labels) in enumerate(trainloader):
        if options['use_gpu']:
            data, labels = data.cuda(), labels.cuda()
        
        with torch.set_grad_enabled(True):
            optimizer.zero_grad()
            x, y = net(data, True)
            loss = criterion(y, labels)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
        
        losses.update(loss.item(), labels.size(0))

        if (batch_idx+1) % options['print_freq'] == 0:
            print("Batch {}/{}\t Loss {:.6f} ({:.6f})" \
                  .format(batch_idx+1, len(trainloader), losses.val, losses.avg))
        
        loss_all += losses.avg
    print("Epoch loss: {}".format(loss_all))
    return loss_all
