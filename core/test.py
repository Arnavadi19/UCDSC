import os
import os.path as osp
import numpy as np

import torch
from torch.autograd import Variable
import torch.nn.functional as F

from core import evaluation
from modules.dchs import get_l2_pred, get_l2_pred_b9
from modules.NirvanaLoss import get_l2_pred_nosubcenter, cosine_similarity, euc_cos

def test_nirvana_oe(net, criterion, testloader, epoch=None, **options):
    net.eval()
    correct, total = 0, 0
    results = dict()
    torch.cuda.empty_cache()

    # _pred_k, _pred_u, _labels = [], [], []
    # _pred_k_b9, _pred_u_b9 = [], []
    with torch.no_grad():
        for data, labels in testloader:
            if options['use_gpu']:
                data, labels = data.cuda(), labels.cuda()
            
            with torch.set_grad_enabled(False):
                x, y = net(data, True)
                predictions = get_l2_pred_nosubcenter(x,criterion.centers, labels)
                # preds_l2_norm = cosine_similarity(x,criterion.centers, labels)
                # dist_euc_cos = euc_cos(x,criterion.centers, labels)
                
                # predictions, logits, logits_b9 = get_l2_pred_b9(x,criterion.centers,return_logits=True)
           
                total += labels.size(0)
                correct += (predictions == labels.data).sum()
            
                # _pred_k.append(logits.data.cpu().numpy())
                # _pred_k_b9.append(logits_b9.data.cpu().numpy())
                # _labels.append(labels.data.cpu().numpy())


    # Accuracy
    acc = float(correct) * 100. / float(total)
    results['ACC'] = acc
    print('Test-Acc: {:.5f}'.format(acc))
    return results


def test_ddfm_b9(net, criterion, testloader, outloader, epoch=None, **options):
    net.eval()
    correct, total = 0, 0

    torch.cuda.empty_cache()

    _pred_k, _pred_u, _labels = [], [], []
    _pred_k_b9, _pred_u_b9 = [], []
    with torch.no_grad():
        for data, labels in testloader:
            if options['use_gpu']:
                data, labels = data.cuda(), labels.cuda()
            
            with torch.set_grad_enabled(False):
                x, y = net(data, True)
                # x = net(data, True)
                predictions, logits, logits_b9 = get_l2_pred_b9(x,criterion.centers,return_logits=True)
           
                total += labels.size(0)
                correct += (predictions == labels.data).sum()
            
                _pred_k.append(logits.data.cpu().numpy())
                _pred_k_b9.append(logits_b9.data.cpu().numpy())
                _labels.append(labels.data.cpu().numpy())

        for batch_idx, (data, labels) in enumerate(outloader):
            if options['use_gpu']:
                data, labels = data.cuda(), labels.cuda()
            
            with torch.set_grad_enabled(False):
                x, y = net(data, True)
                # x = net(data, True)
                _, logits, logits_b9 = get_l2_pred_b9(x,criterion.centers,return_logits=True)
                _pred_u.append(logits.data.cpu().numpy())
                _pred_u_b9.append(logits_b9.data.cpu().numpy())

    # Accuracy
    acc = float(correct) * 100. / float(total)
    print('Acc: {:.5f}'.format(acc))

    _pred_k = np.concatenate(_pred_k, 0)
    _pred_u = np.concatenate(_pred_u, 0)
    _labels = np.concatenate(_labels, 0)
    _pred_k_b9 = np.concatenate(_pred_k_b9, 0)
    _pred_u_b9 = np.concatenate(_pred_u_b9, 0)
    
    # Out-of-Distribution detction evaluation
    x1, x2 = np.max(_pred_k, axis=1), np.max(_pred_u, axis=1)
    results = evaluation.metric_ood(x1, x2)['Bas']
    
    # Out-of-Distribution detction evaluation
    x1_b9, x2_b9 = np.max(_pred_k_b9, axis=1), np.max(_pred_u_b9, axis=1)
    results_b9 = evaluation.metric_ood(x1_b9, x2_b9)['Bas']

    # OSCR
    _oscr_socre = evaluation.compute_oscr(_pred_k, _pred_u, _labels)

    results['ACC'] = acc
    results['OSCR'] = _oscr_socre * 100.

    # OSCR
    _oscr_socre_b9 = evaluation.compute_oscr(_pred_k_b9, _pred_u_b9, _labels)

    results_b9['ACC'] = acc
    results_b9['OSCR'] = _oscr_socre_b9 * 100.

    return results, results_b9

def test_ddfm(net, criterion, testloader, outloader, epoch=None, **options):
    net.eval()
    correct, total = 0, 0

    torch.cuda.empty_cache()

    _pred_k, _pred_u, _labels = [], [], []

    with torch.no_grad():
        for data, labels in testloader:
            if options['use_gpu']:
                data, labels = data.cuda(), labels.cuda()
            
            with torch.set_grad_enabled(False):
                x, y = net(data, True)
                predictions, logits = get_l2_pred(x,criterion.centers,return_logits=True)
           
                total += labels.size(0)
                correct += (predictions == labels.data).sum()
            
                _pred_k.append(logits.data.cpu().numpy())
                _labels.append(labels.data.cpu().numpy())

        for batch_idx, (data, labels) in enumerate(outloader):
            if options['use_gpu']:
                data, labels = data.cuda(), labels.cuda()
            
            with torch.set_grad_enabled(False):
                x, y = net(data, True)
                _, logits = get_l2_pred(x,criterion.centers,return_logits=True)
                _pred_u.append(logits.data.cpu().numpy())

    # Accuracy
    acc = float(correct) * 100. / float(total)
    print('Acc: {:.5f}'.format(acc))

    _pred_k = np.concatenate(_pred_k, 0)
    _pred_u = np.concatenate(_pred_u, 0)
    _labels = np.concatenate(_labels, 0)
    
    # Out-of-Distribution detction evaluation
    x1, x2 = np.max(_pred_k, axis=1), np.max(_pred_u, axis=1)
    results = evaluation.metric_ood(x1, x2)['Bas']
    
    # OSCR
    _oscr_socre = evaluation.compute_oscr(_pred_k, _pred_u, _labels)

    results['ACC'] = acc
    results['OSCR'] = _oscr_socre * 100.

    return results

def test(net, criterion, testloader, epoch=None, **options):
    net.eval()
    correct, total = 0, 0
    results = dict()
    torch.cuda.empty_cache()

    _pred_k, _pred_u, _labels = [], [], []

    with torch.no_grad():
        for data, labels in testloader:
            if options['use_gpu']:
                data, labels = data.cuda(), labels.cuda()
            
            with torch.set_grad_enabled(False):
                x, y = net(data, True)
                # logits, _ = criterion(x, y)
                predictions = y.data.max(1)[1]
                total += labels.size(0)
                correct += (predictions == labels.data).sum()
            
                _pred_k.append(y.data.cpu().numpy())
                _labels.append(labels.data.cpu().numpy())

        # for batch_idx, (data, labels) in enumerate(outloader):
        #     if options['use_gpu']:
        #         data, labels = data.cuda(), labels.cuda()
            
        #     with torch.set_grad_enabled(False):
        #         x, y = net(data, True)
        #         # x, y = net(data, return_feature=True)
        #         logits, _ = criterion(x, y)
        #         _pred_u.append(logits.data.cpu().numpy())

    # Accuracy
    acc = float(correct) * 100. / float(total)
    print('Test Acc: {:.5f}'.format(acc))

    # _pred_k = np.concatenate(_pred_k, 0)
    # _pred_u = np.concatenate(_pred_u, 0)
    # _labels = np.concatenate(_labels, 0)
    
    # # Out-of-Distribution detction evaluation
    # x1, x2 = np.max(_pred_k, axis=1), np.max(_pred_u, axis=1)
    # results = evaluation.metric_ood(x1, x2)['Bas']
    
    # # OSCR
    # _oscr_socre = evaluation.compute_oscr(_pred_k, _pred_u, _labels)

    results['ACC'] = acc
    # results['OSCR'] = _oscr_socre * 100.

    return results
