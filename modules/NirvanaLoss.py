import torch
import torch.nn as nn
import numpy as np

from .utils import FindCenters

       

class center_loss_nirvana(nn.Module):
    """Center loss with subcenters added.
    
    Args:
        num_classes (int): number of classes.
        num_subcenters (int): number of subcenters per class
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=3, feat_dim=2, precalc_centers=None, device=None, Expand=1):
        super(center_loss_nirvana, self).__init__()
        self.num_classes = num_classes
        self.num_centers = self.num_classes
        self.feat_dim = feat_dim
        self.device=device
        self.E = Expand 
        self.loss =  nn.MSELoss()
        #self.interclass_margin = interclass_margin
        if(precalc_centers):
            precalculated_centers = FindCenters(self.feat_dim, self.E)
            precalculated_centers = precalculated_centers[:self.num_classes,:]
        with torch.no_grad():
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim,device=self.device,requires_grad=False))
            if(precalc_centers):
                # precalculated_centers*=10
                self.centers.copy_(torch.from_numpy(precalculated_centers))
                print('Centers loaded.')

    def forward(self, x, labels,  nearest=True, epoch=0):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
            nearest (bool): assign subcenter according to labels or nearest
        """
        batch_size = x.size(0)
        
        centers_batch = self.centers.index_select(0,labels.long())
        
        
        # inlier_distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
        #           torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        # inlier_distmat.addmm_(x, self.centers.t(),beta=1,alpha=-2)
        # intraclass_distances = inlier_distmat.gather(dim=1,index=labels.unsqueeze(1)).squeeze()
        # intraclass_loss = intraclass_distances.sum()/(batch_size*self.feat_dim*2.0)
      
        
        intraclass_loss = self.loss(x, centers_batch)
        return intraclass_loss
    
class NirvanaHinge(nn.Module):
    """Center loss with subcenters added.
   
    Args:
        num_classes (int): number of classes.
        num_subcenters (int): number of subcenters per class
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=3, feat_dim=2, precalc_centers=None, device=None, Expand=1):
        super(NirvanaHinge, self).__init__()
        self.num_classes = num_classes
        self.num_centers = self.num_classes
        self.feat_dim = feat_dim
        self.device=device
        self.E = Expand
        self.loss =  nn.MSELoss()
        #self.interclass_margin = interclass_margin
        if(precalc_centers):
            precalculated_centers = FindCenters(self.feat_dim, self.E)
            precalculated_centers = precalculated_centers[:self.num_classes,:]
        with torch.no_grad():
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim,device=self.device,requires_grad=False))
            if(precalc_centers):
                # precalculated_centers*=10
                self.centers.copy_(torch.from_numpy(precalculated_centers))
                print('Centers loaded.')
            self.margin = torch.div(torch.norm(self.centers[0].detach()-self.centers[1].detach()),10).item()
            #self.margin=2.0


    def forward(self, x,labels,  nearest=True, epoch=0):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
            nearest (bool): assign subcenter according to labels or nearest
        """
        batch_size = x.size(0)
        centers_batch = self.centers.view(-1,self.feat_dim).index_select(0, labels.long())
       
       
        intraclass_dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(x.size(0),x.size(0)) + \
                                torch.pow(centers_batch, 2).sum(dim=1, keepdim=True).expand(x.size(0), x.size(0)).t()
        intraclass_dist.addmm_(x, centers_batch.t(),beta=1,alpha=-2)
        NirvanaHinge = torch.div(torch.sum(torch.max(torch.zeros(1).to(intraclass_dist.device), torch.sub(torch.diag(intraclass_dist), self.margin))), (batch_size*2*2))
       
        # intraclass_loss = self.loss(x, centers_batch)
        return NirvanaHinge
    
class nirvana_mics_loss(nn.Module):
    def __init__(self, num_classes=3, feat_dim=2, precalc_centers=None, device=None, Expand=1):
        super(nirvana_mics_loss, self).__init__()
        self.Lambda = 0.01
        self.num_classes = num_classes
        self.num_centers = self.num_classes
        self.feat_dim = feat_dim
        self.device=device
        self.E = Expand
        self.loss =  nn.MSELoss()
        #self.interclass_margin = interclass_margin
        if(precalc_centers):
            precalculated_centers = FindCenters(self.feat_dim, self.E)
            precalculated_centers = precalculated_centers[:self.num_classes,:]
        with torch.no_grad():
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim,device=self.device,requires_grad=False))
            if(precalc_centers):
                # precalculated_centers*=10
                self.centers.copy_(torch.from_numpy(precalculated_centers))
                print('Centers loaded.')
   
    def forward(self, x,labels,  nearest=True, epoch=0):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
            nearest (bool): assign subcenter according to labels or nearest
        """
        batch_size = x.size(0)
        centers_batch = self.centers.view(-1,self.feat_dim).index_select(0, labels.long())
        # distvec = torch.cdist(x, centers_batch, p=2)
        distmat = torch.cdist(x, self.centers)

        classes = torch.arange(self.num_classes,device=self.device).long()
        classes.expand(batch_size, self.num_classes)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        labels.eq(classes.expand(batch_size, self.num_classes))
        mask = labels.eq(classes.expand(batch_size, self.num_classes))
        exp_loss = torch.sum(torch.log(torch.add(torch.sum(torch.exp(torch.sub(distmat.masked_select(mask).reshape(-1,1), distmat)), dim=1), 1)))
        # exp_loss2 = []
        # torch.zeros((batch_size,10), dtype=bool)
        # for i in range(batch_size):
        #     exp_loss2.append(torch.log(torch.exp(distmat[i,labels[i]] - distmat[i,:]).sum() + 1))
        intraclass_loss = self.loss(x, centers_batch)
        # print('Intra:', intraclass_loss.item(), 'exp_loss:', self.Lambda*torch.stack(exp_loss).sum().item())
        total_loss = torch.add(intraclass_loss, exp_loss, alpha=self.Lambda)
        return total_loss    

class nirvana_hypersphere(nn.Module):
    """Center loss with subcenters added.
   
    Args:
        num_classes (int): number of classes.
        num_subcenters (int): number of subcenters per class
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=3, feat_dim=2, precalc_centers=None, device=None, Expand=1):
        super(nirvana_hypersphere, self).__init__()
        self.num_classes = num_classes
        self.num_centers = self.num_classes
        self.feat_dim = feat_dim
        self.device=device
        self.E = Expand
        self.mse =  nn.MSELoss()
        #self.interclass_margin = interclass_margin
        with torch.no_grad():
            # self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim,device=self.device,requires_grad=False))
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim), requires_grad=True)
            precalculated_centers = torch.mul(torch.nn.functional.normalize(self.centers),self.E)
            if(precalc_centers):
                # precalculated_centers*=10
                self.centers.copy_(precalculated_centers)
                print('Centers loaded.')
        #self.margin = torch.mul(torch.mul(self.num_classes-1, self.E), 2).item()
        self.margin = 34
    def forward(self, x, labels,  nearest=True, epoch=0):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
            nearest (bool): assign subcenter according to labels or nearest
        """
        batch_size = x.size(0)
        # x = torch.mul(nn.functional.normalize(x), self.E)
       # centers_norms = torch.norm(self.centers,dim=1)
        # print(centers_norms)
        with torch.no_grad():
             self.centers.copy_(torch.mul(torch.nn.functional.normalize(self.centers),self.E))
        centers_batch = self.centers.view(-1,self.feat_dim).index_select(0, labels.long())
        #distcent = torch.sum(torch.max(torch.zeros(1).to(self.centers.device), torch.sub(self.margin, torch.sum(torch.cdist(self.centers, self.centers), dim=1))))
        #interclass_loss = torch.div(distcent, self.num_classes*self.num_classes)
       
        
        centers_distance = 1/(torch.pow(torch.cdist(self.centers, self.centers),2)+1) 
        centers_distance.fill_diagonal_(0)
        uniform_loss = torch.sum(centers_distance)
       
       # uniform_loss = torch.div(1,dist_centers)
        
        intraclass_loss = self.mse(x, centers_batch)
        # total_loss = torch.add(intraclass_loss, intraclass_loss, alpha=0.5)  
        xnormalized = torch.mul(nn.functional.normalize(x), self.E)
        angle_loss_matrix = -(1.0/(batch_size*self.E*self.E))*torch.matmul(xnormalized,self.centers.t())
        angle_mask = 2.0*torch.nn.functional.one_hot(labels.long(),num_classes=self.num_classes)-torch.ones(batch_size,self.num_classes).to("cuda:0")
        masked_loss = angle_loss_matrix*angle_mask
        angle_margin =0.8
        angle_loss = (1/batch_size)*torch.max(torch.zeros(1).to(self.device),torch.sub(angle_margin,masked_loss)).sum()

       
                
   
        inlier_distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        inlier_distmat.addmm_(x, self.centers.t(),beta=1,alpha=-2)
        intraclass_distances = inlier_distmat.gather(dim=1,index=labels.unsqueeze(1)).squeeze()
        intraclass_loss_b9 = intraclass_distances.sum()/(batch_size*self.feat_dim*2.0)
        
        centers_dist_inter = (intraclass_distances.repeat(self.num_centers,1).t() - inlier_distmat)
        # outlierdaki her bir ornegi her sınıf için de kullan
        mask = torch.logical_not(torch.nn.functional.one_hot(labels.long(),num_classes=self.num_classes))
        #batch_size'a bölmeyi deneyebilirsin. (1/mask.count_nonzero())*
        interclass_loss_triplet = (1/(self.num_centers*batch_size*2.0))*((self.margin+centers_dist_inter).clamp(min=0)*mask).sum()
                
        
        return intraclass_loss_b9, interclass_loss_triplet, uniform_loss, angle_loss
        
class cross_entropy_nirvana(nn.Module):
        """Center loss with subcenters added.
        
        Args:
            num_classes (int): number of classes.
            num_subcenters (int): number of subcenters per class
            feat_dim (int): feature dimension.
        """
        def __init__(self, num_classes=3, feat_dim=2, precalc_centers=None, device=None, Expand=1):
            super(cross_entropy_nirvana, self).__init__()
            self.num_classes = num_classes
            self.num_centers = self.num_classes
            self.feat_dim = feat_dim
            self.device=device
            self.E = Expand 
            self.loss =  nn.MSELoss()
            #self.interclass_margin = interclass_margin
            if(precalc_centers):
                precalculated_centers = FindCenters(self.feat_dim, self.E)
                precalculated_centers = precalculated_centers[:self.num_classes,:]
            with torch.no_grad():
                self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim,device=self.device,requires_grad=False))
                if(precalc_centers):
                    # precalculated_centers*=10
                    self.centers.copy_(torch.from_numpy(precalculated_centers))
                    print('Centers loaded.')

        def forward(self, x, labels,  nearest=True, epoch=0):
            """
            Args:
                x: feature matrix with shape (batch_size, feat_dim).
                labels: ground truth labels with shape (batch_size).
                nearest (bool): assign subcenter according to labels or nearest
            """
            batch_size = x.size(0)
            
            centers_batch = self.centers.index_select(0,labels.long())
            
           # pr1=torch.sum(x*centers_batch,dim=1)
            pr1=torch.matmul(x,self.centers.T)
            pr2=torch.sum(torch.exp(torch.matmul(x,self.centers.T)),dim=1)
            predictions=torch.exp(pr1)/pr2.reshape(-1,1)
           
            y=labels
            y = y.reshape(batch_size, 1)
          
            num_classes = self.num_classes 
            one_hot_target = (y == torch.arange(num_classes).reshape(1, num_classes).cuda()).float()
            
            
            ce_nirvana = - torch.mean(torch.log(predictions) * one_hot_target) 
            return ce_nirvana         
            
           

   
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

        # distmat = torch.pow(features, 2).sum(dim=1, keepdim=True).expand(batch_size, num_centers) + \
        #           torch.pow(serialized_centers, 2).sum(dim=1, keepdim=True).expand(num_centers, batch_size).t()
        # distmat.addmm_(features, serialized_centers.t(),beta=1,alpha=-2)
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


