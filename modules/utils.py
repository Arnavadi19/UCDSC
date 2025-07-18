import datetime, time
import torch
import numpy as np
import modules.utils_torchvision as utils
from collections import defaultdict
from torchvision import transforms
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.models as models

def train_one_epoch(model, centerloss, optimizer, data_loader, device, epoch, args):
    model.train(), centerloss.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('img/s', utils.SmoothedValue(window_size=10, fmt='{value:.1f}'))
    header = 'Epoch: [{}]'.format(epoch)
    all_features, all_labels = [], []
    for image, target in metric_logger.log_every(data_loader, args.print_freq, header):
        start_time = time.time()
        image, target = image.to(device), target.to(device)
        
        if args.distributed:
            centers = centerloss.module.centers
        else:
            centers = centerloss.centers
        
        feats = model(image)
        loss = centerloss(feats,target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        all_features.append(feats.data.cpu().numpy())
        all_labels.append(target.data.cpu().numpy())
        acc1 = accuracy_l2_nosubcenter(feats,centers,target)
        batch_size = image.shape[0]
        metric_logger.update(loss=loss.item(),lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['img/s'].update(batch_size / (time.time() - start_time))
    
    metric_logger.synchronize_between_processes()
    # all_features = np.concatenate(all_features, 0)
    # all_labels = np.concatenate(all_labels, 0)
    print(' *Train Acc@1 {top1.global_avg:.3f} '
          .format(top1=metric_logger.acc1))
    return metric_logger.acc1.global_avg, metric_logger.loss.global_avg


def train_one_uhs(model, centerloss, optimizer, optimizer_center, data_loader, device, epoch, args):
    model.train(), centerloss.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('img/s', utils.SmoothedValue(window_size=10, fmt='{value:.1f}'))
    header = 'Epoch: [{}]'.format(epoch)
    all_features, all_labels = [], []
    for image, target in metric_logger.log_every(data_loader, args.print_freq, header):
        start_time = time.time()
        image, target = image.to(device), target.to(device)
        
        if args.distributed:
            centers = centerloss.module.centers
        else:
            centers = centerloss.centers
        
        feats, _ = model(image)
        intraclass_loss, triplet_loss, uniform_loss = centerloss(feats, target)
        loss = 2.5 * intraclass_loss + 2.5 * triplet_loss + 0.001 * uniform_loss
        optimizer_center.zero_grad()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer_center.step()
        all_features.append(feats.data.cpu().numpy())
        all_labels.append(target.data.cpu().numpy())
        acc1 = accuracy_l2_nosubcenter(feats,centers,target)
        batch_size = image.shape[0]
        metric_logger.update(loss=loss.item(),lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['img/s'].update(batch_size / (time.time() - start_time))
    
    metric_logger.synchronize_between_processes()
    # all_features = np.concatenate(all_features, 0)
    # all_labels = np.concatenate(all_labels, 0)
    print(' *Train Acc@1 {top1.global_avg:.3f} '
          .format(top1=metric_logger.acc1))
    return metric_logger.acc1.global_avg, metric_logger.loss.global_avg   


def evaluate_majority_voting(model, centerloss, data_loader, device, epoch, args):
    model.eval(), centerloss.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    if args.distributed:
        centers = centerloss.module.centers
    else:
        centers = centerloss.centers
    header = 'Test:'
    all_preds_l2,all_preds_l2_norm, all_labels, all_feats, all_dist_euc_cos = [], [], [], [], []
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, args.print_freq, header):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            feats, _ = model(image)
            preds_l2 = get_l2_pred_nosubcenter(feats,centers, target)
            preds_l2_norm = cosine_similarity(feats,centers, target)
            dist_euc_cos = euc_cos(feats,centers, target)
            all_dist_euc_cos.append(dist_euc_cos.cpu())
            all_preds_l2.append(preds_l2.cpu())
            all_preds_l2_norm.append(preds_l2_norm.cpu())
            all_labels.append(target.cpu())
            all_feats.append(feats.cpu())

    preds_l2_norm = torch.cat(all_preds_l2_norm, 0)
    preds_l2 = torch.cat(all_preds_l2, 0)
    preds_euc_cos = torch.cat(all_dist_euc_cos, 0)
    labels = torch.cat(all_labels, 0)
    test_acc_l2 = (float((labels == preds_l2).sum())/len(labels))*100.0
    test_acc_l2_norm = (float((labels == preds_l2_norm).sum())/len(labels))*100.0
    test_acc_dist_cos = (float((labels == preds_euc_cos).sum())/len(labels))*100.0
    print('Test Acc@1_L2 %.3f'%test_acc_l2)
    print('Test Acc@1_COSINE %.3f'%test_acc_l2_norm)
    print('Test Acc@EUC_COS %.3f'%test_acc_dist_cos)
    # all_feats_tsne = TSNE(n_components=2).fit_transform(torch.cat(all_feats, 0))
    # plot_features(all_feats_tsne, labels.data.numpy())
    return test_acc_l2,test_acc_l2_norm

def evaluate_ijba_maj_vot(model, criterion_center, train_transform, device, epoch, args):
    model.eval(),criterion_center.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    data_dict = defaultdict(list)
    with torch.no_grad():
        for i in range(1,11):
            all_features, all_labels = [], []
            csv_name = 'gallery%d'%i
            folder_name='split%d'%i
            dataset = IJBA(folder_path ='./data/ijba/IJBA-Aligned-112',meta_path='./data/ijba/META/%s/%s.csv'%(folder_name,csv_name),transform=train_transform)
            data_loader = torch.utils.data.DataLoader(dataset, batch_size=512,shuffle=False, num_workers=8, pin_memory=True)
            for inputs, labels, path in metric_logger.log_every(data_loader, args.print_freq, "Extract_Feats_%s"%csv_name):
                inputs = inputs.cuda()
                features, _ = model(inputs)
                all_features.append(features.data.cpu())
                all_labels.append(labels.cpu())
            
            data_dict[csv_name] = [torch.cat(all_features, 0), torch.cat(all_labels, 0)]
        
        for i in range(1,11):
            all_features, all_labels = [], []
            csv_name = 'probe%d'%i
            folder_name='split%d'%i
            dataset = IJBA(folder_path ='./data/ijba/IJBA-Aligned-112',meta_path='./data/ijba/META/%s/%s.csv'%(folder_name,csv_name),transform=train_transform)
            data_loader = torch.utils.data.DataLoader(dataset, batch_size=512,shuffle=False, num_workers=8, pin_memory=True)
            for inputs, labels, path in metric_logger.log_every(data_loader, args.print_freq, "Extract_Feats_%s"%csv_name):
                inputs = inputs.cuda()
                features, _ = model(inputs)
                all_features.append(features.data.cpu())
                all_labels.append(labels.cpu())
            
            data_dict[csv_name] = [torch.cat(all_features, 0), torch.cat(all_labels, 0)]
    set_scores = []
    for i in range(1,11):

        gallery_features = data_dict['gallery%d'%i][0]
        gallery_subj_ids = data_dict['gallery%d'%i][1][:,1]
        gallery_temp_ids = data_dict['gallery%d'%i][1][:,0]

        gallery_uniq_subj_ids = np.unique(gallery_subj_ids)
        assert np.unique(gallery_subj_ids).size ==  np.unique(gallery_temp_ids).size
        
        gallery_subj_means = np.zeros((gallery_uniq_subj_ids.shape[0],args.feat_dim))
        for j, subj_id in enumerate(gallery_uniq_subj_ids):
            gallery_subj_means[j,:] = np.mean(gallery_features[gallery_subj_ids==subj_id].numpy(),0)
    

        probe_features = data_dict['probe%d'%i][0]
        probe_subj_ids = data_dict['probe%d'%i][1][:,1]
 
        # preds = ultraslow_l2_pred(torch.from_numpy(probe_features).double(), torch.from_numpy(gallery_subj_means).double())
        # preds = get_l2_pred_nosubcenter(torch.from_numpy(probe_features).cpu().double(), torch.from_numpy(gallery_subj_means).cpu().double())
        
        preds = get_l2_pred_nosubcenter_new(probe_features.double(), torch.from_numpy(gallery_subj_means).double())
        num_set = []
        for cls_ in np.unique(gallery_uniq_subj_ids):
            if(preds[probe_subj_ids==cls_].nelement()):
                res=np.argmax(np.bincount(preds[probe_subj_ids==cls_]))
                num_set.append(gallery_uniq_subj_ids[res]==cls_)
        print('Test Probe%d-Gallery%d Set_Acc %.5f'%(i,i,(np.count_nonzero(num_set)/float(len(num_set)))*100.0))
        print('Test Probe%d-Gallery%d Acc@1 %.5f'%(i,i,((probe_subj_ids.numpy() == gallery_uniq_subj_ids[preds]).sum()/len(probe_subj_ids))*100.0))
        set_scores.append((np.count_nonzero(num_set)/float(len(num_set)))*100.0)
    mean_score = np.mean(set_scores)
    print("Averaged set score: %.3f"%mean_score)
    return mean_score

def accuracy_l2_nosubcenter(features,centers,targets):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        # features are expected in (batch_size,feat_dim)
        # centers are expected in shape (num_classes,num_subcenters,feat_dim)
        batch_size = targets.size(0)
        num_classes, feat_dim = centers.shape
        num_centers = num_classes
        
        serialized_centers = centers.view(-1,feat_dim)
        assert num_centers == serialized_centers.size(0)

        # distmat = torch.pow(features, 2).sum(dim=1, keepdim=True).expand(batch_size, num_centers) + \
        #           torch.pow(serialized_centers, 2).sum(dim=1, keepdim=True).expand(num_centers, batch_size).t()
        # distmat.addmm_(features, serialized_centers.t(),beta=1,alpha=-2)
        # distmat in shape (batch_size,num_centers)
        distmat = torch.cdist(features, serialized_centers, p=2)
        pred = distmat.argmin(1)
        correct = pred.eq(targets)
        correct_k = correct.flatten().sum(dtype=torch.float32)
        return correct_k * (100.0 / batch_size)    
    
def get_l2_pred_nosubcenter_new(features,centers,normalize = True):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        # features are expected in (batch_size,feat_dim)
        # centers are expected in shape (num_classes,num_subcenters,feat_dim)
        batch_size = features.size(0)
        num_classes, feat_dim = centers.shape
        num_centers = num_classes
        
        serialized_centers = centers.view(-1,feat_dim)
        assert num_centers == serialized_centers.size(0)
        
        if normalize:
            feat_norms = torch.norm(features,dim=1)
            features = features/feat_norms.unsqueeze(1)
            centers_norm = torch.norm(serialized_centers,dim=1)
            serialized_centers = serialized_centers/centers_norm.unsqueeze(1)
        
        distmat = torch.pow(features, 2).sum(dim=1, keepdim=True).expand(batch_size, num_centers) + \
                  torch.pow(serialized_centers, 2).sum(dim=1, keepdim=True).expand(num_centers, batch_size).t()
        distmat.addmm_(features, serialized_centers.t(),beta=1,alpha=-2)
        pred = distmat.argmin(1)

        return pred
    
def get_l2_pred_nosubcenter(features,centers, target):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        # features are expected in (batch_size,feat_dim)
        # centers are expected in shape (num_classes,num_subcenters,feat_dim)
        batch_size = features.size(0)
        num_classes, feat_dim = centers.shape
        num_centers = num_classes
        
        serialized_centers = centers.view(-1,feat_dim)
        assert num_centers == serialized_centers.size(0)

        distmat1 = torch.pow(features, 2).sum(dim=1, keepdim=True).expand(batch_size, num_centers) + \
                  torch.pow(serialized_centers, 2).sum(dim=1, keepdim=True).expand(num_centers, batch_size).t()
        distmat1.addmm_(features, serialized_centers.t(),beta=1,alpha=-2)
        # distmat in shape (batch_size,num_centers)
        distmat = torch.cdist(features, serialized_centers, p=2)
        pred = distmat.argmin(1)

        return pred

def cosine_similarity(features, centers, target):
    with torch.no_grad():
        # features are expected in (batch_size,feat_dim)
        # centers are expected in shape (num_classes,num_subcenters,feat_dim)
        batch_size = features.size(0)
        num_classes, feat_dim = centers.shape
        num_centers = num_classes
        serialized_centers = centers.view(-1,feat_dim)
        assert num_centers == serialized_centers.size(0)
        
        pred = torch.empty(batch_size, device=features.device)
        for i in range(batch_size):
            pred[i] = nn.functional.cosine_similarity(features[i].reshape(1,-1), serialized_centers).argmax()
    return pred

def euc_cos(features,centers, target):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        # features are expected in (batch_size,feat_dim)
        # centers are expected in shape (num_classes,num_subcenters,feat_dim)
        batch_size = features.size(0)
        num_classes, feat_dim = centers.shape
        num_centers = num_classes
        
        serialized_centers = centers.view(-1,feat_dim)
        assert num_centers == serialized_centers.size(0)
    
        # distmat = torch.pow(features, 2).sum(dim=1, keepdim=True).expand(batch_size, num_centers) + \
        #           torch.pow(serialized_centers, 2).sum(dim=1, keepdim=True).expand(num_centers, batch_size).t()
        # distmat.addmm_(features, serialized_centers.t(),beta=1,alpha=-2)
        # distmat in shape (batch_size,num_centers)
        disteuc = torch.cdist(features, serialized_centers, p=2)
        # pred = distmat.argmin(1)   
        distcos = torch.empty(batch_size, num_classes, device=features.device)
        for i in range(batch_size):
            distcos[i] = nn.functional.cosine_similarity(features[i].reshape(1,-1), serialized_centers)
    return ((1/(2+distcos))*disteuc).argmin(1)

def dataset_preperation(args):
    if args.dataset_name == 'CIFAR10' or args.dataset_name == 'CIFAR100':
        dataset = eval('datasets' + '.'+ args.dataset_name)
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                        std=[0.2023, 0.1994, 0.2010]) 
        data_loader =  torch.utils.data.DataLoader(
           dataset(root='./data', train=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]), download=True),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)

        data_loader_val = torch.utils.data.DataLoader(
            dataset(root='./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
    elif args.dataset_name == 'MNIST':
        dataset = eval('datasets' + '.'+ args.dataset_name)
        normalize = transforms.Normalize((0.1307,), (0.3081,)) 
        data_loader =  torch.utils.data.DataLoader(
            dataset(root='./data', train=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]), download=True),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)

        data_loader_val = torch.utils.data.DataLoader(
            dataset(root='./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
    elif args.dataset_name == 'VGGFace2':
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                        std=[0.5, 0.5, 0.5]) 
        data_loader =  torch.utils.data.DataLoader(
            datasets.ImageFolder(root='/home/mlcv/CevikalpPy/deep_simplex_classifier/data/faces_ms1m_v3_12k', transform=transforms.Compose([
                transforms.Resize((112,112)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)

        data_loader_val = torch.utils.data.DataLoader(
            datasets.ImageFolder(root='/home/mlcv/CevikalpPy/deep_simplex_classifier/data/vgg_faces_img_2049', transform=transforms.Compose([
                transforms.Resize((112,112)),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
    return data_loader, data_loader_val

def FindCenters(k, E=1):
    """
    Calculates "k+1" equidistant points in R^{k}.
    Args:
        k (int) dimension of the space
        E (float) expand factor 
    Returns: 
        Centers (np.array) equidistant positions in R^{k}, shape (k+1 x k)
    """
    
    Centers = np.empty((k+1, k), dtype=np.float32)
    CC = np.empty((k,k), dtype=np.float32)
    Unit_Vector = np.identity(k)
    c = -((1+np.sqrt(k+1))/np.power(k, 3/2))
    CC.fill(c)
    d = np.sqrt((k+1)/k)
    DU = d*Unit_Vector 
    Centers[0,:].fill(1/np.sqrt(k))
    Centers[1:,:] = CC + DU
    
    # Calculate and Check Distances
    # Distances = np.empty((k+1,k), dtype=np.float32)
    # for k, rows in enumerate(Centers):
    #     Distances[k,:] = np.linalg.norm(rows - np.delete(Centers, k, axis=0), axis=1)
    # # print("Distances:",Distances)    
    # assert np.allclose(np.random.choice(Distances.flatten(), size=1), Distances, rtol=1e-05, atol=1e-08, equal_nan=False), "Distances are not equal" 
    return Centers*E