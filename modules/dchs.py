import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

        
class NirvanaOpenset_loss(nn.Module):
    """
    Args:
        num_classes (int): number of classes.
        num_subcenters (int): number of subcenters per class
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=10, feat_dim=128, precalc_centers=None, margin=48.0, Expand=200, use_uncertainty_reg=True, uncertainty_weight = 5.0):
        super(NirvanaOpenset_loss, self).__init__()
        self.num_classes = num_classes
        self.num_centers = self.num_classes
        self.feat_dim = feat_dim
        self.margin = margin
        self.E = Expand
        self.use_uncertainty_reg = use_uncertainty_reg
        self.uncertainty_weight = uncertainty_weight

        if(precalc_centers):
            precalculated_centers = FindCenters(self.feat_dim, self.E)
            precalculated_centers = precalculated_centers[:self.num_classes,:]

        with torch.no_grad():
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim, requires_grad=False))
            if(precalc_centers):
                self.centers.copy_(torch.from_numpy(precalculated_centers))
                print('Centers loaded.')

        
    def compute_uncertainty_penalty(self, features, labels):
        """
        Compute uncertainty penalty: U = min_distance / avg_distance
        """
        batch_size = features.size(0)
        
        distances = torch.cdist(features, self.centers)  # [batch_size, num_centers]
        
        min_distances, min_indices = torch.min(distances, dim=1)  # [batch_size]
        
        # Calculate average distance to all OTHER centers (excluding the closest one)
        uncertainty_penalties = []
        for i in range(batch_size):
            sample_distances = distances[i]  # [num_centers]
            closest_idx = min_indices[i]
            
            mask = torch.ones(self.num_centers, dtype=torch.bool, device=features.device)
            mask[closest_idx] = False
            
            avg_other_distance = sample_distances[mask].mean()
            
            # Uncertainty ratio: min_distance / avg_distance_to_others
            uncertainty_ratio = min_distances[i] / (avg_other_distance + 1e-8)
            uncertainty_penalties.append(uncertainty_ratio)
        
        uncertainty_penalties = torch.stack(uncertainty_penalties)
        return uncertainty_penalties.mean()
        
    def forward(self, labels, x, x_out, ramp=False):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        if labels.max() >= self.num_classes or labels.min() < 0:
            raise ValueError("Labels out of valid range for NirvanaOpenset_loss.")
        
        # Convert all tensors to the same dtype
        dtype = x.dtype
        device = x.device
        
        # Convert centers to match x's dtype
        centers = self.centers.to(dtype=dtype, device=device)
        batch_size = x.size(0)

        uncertainty_loss = torch.tensor(0.0, dtype=dtype, device=device)

        # Calculate initial distance components, ensuring same dtype
        x_squared = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes)
        centers_squared = torch.pow(centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        
        # Create distance matrix with controlled dtype
        inlier_distmat = (x_squared + centers_squared).to(dtype=dtype)
        
        # Perform matrix multiplication with tensors of the same dtype
        inlier_distmat.addmm_(x, centers.t(), beta=1, alpha=-2)
        
        # Continue with remaining calculations
        intraclass_distances = inlier_distmat.gather(dim=1, index=labels.unsqueeze(1)).squeeze()
        intraclass_loss = intraclass_distances.sum() / (batch_size * self.feat_dim * 2.0)
        
        centers_dist_inter = (intraclass_distances.repeat(self.num_centers, 1).t() - inlier_distmat)
        mask = torch.logical_not(torch.nn.functional.one_hot(labels.long(), num_classes=self.num_classes))
        interclass_loss_triplet = (1 / (self.num_centers * batch_size * 2.0)) * ((self.margin + centers_dist_inter).clamp(min=0) * mask).sum()
        
        if self.use_uncertainty_reg:
            uncertainty_penalty = self.compute_uncertainty_penalty(x, labels)
            uncertainty_loss += self.uncertainty_weight * uncertainty_penalty
            
        if x_out is not None:
            # Make sure outlier features have same dtype as inlier features
            x_out = x_out.to(dtype=dtype)
            batch_size_out = x_out.size(0)
            
            # Create outlier distance matrix with controlled dtype
            x_out_squared = torch.pow(x_out, 2).sum(dim=1, keepdim=True).expand(batch_size_out, self.num_classes)
            centers_squared_out = torch.pow(centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size_out).t()
            outlier_distmat = (x_out_squared + centers_squared_out).to(dtype=dtype)
            
            # Matrix multiplication with matching dtypes
            outlier_distmat.addmm_(x_out, centers.t(), beta=1, alpha=-2)
            
            outlier_corresponding_multi_distances = outlier_distmat.index_select(1, labels.long())
        
            if ramp:
                hinge_part = (self.margin + (intraclass_distances - outlier_corresponding_multi_distances)).clamp(min=0.).clamp(max=60.)
                outlier_triplet_multi_loss = (1 / (batch_size * batch_size_out * 2.0)) * hinge_part.sum()
            else:
                outlier_triplet_multi_loss = (1 / (batch_size * batch_size_out * 2.0)) * ((self.margin + (intraclass_distances - outlier_corresponding_multi_distances)).clamp(min=0)).sum()
        
            return intraclass_loss, interclass_loss_triplet, outlier_triplet_multi_loss, uncertainty_loss
        else:
            return intraclass_loss, interclass_loss_triplet, None, uncertainty_loss


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
    Distances = np.empty((k+1,k), dtype=np.float32)
    for k, rows in enumerate(Centers):
        Distances[k,:] = np.linalg.norm(rows - np.delete(Centers, k, axis=0), axis=1)
    # print("Distances:",Distances)    
    assert np.allclose(np.random.choice(Distances.flatten(), size=1), Distances, rtol=1e-05, atol=1e-08, equal_nan=False), "Distances are not equal" 
    return Centers*E

        

def get_l2_pred(features,centers, return_logits=False):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        # features are expected in (batch_size,feat_dim)
        # centers are expected in shape (num_classes,num_subcenters,feat_dim)
        batch_size = features.size(0)
        num_classes, feat_dim = centers.shape
        num_centers = num_classes
        
        serialized_centers = centers.view(-1,feat_dim)
        assert num_centers == serialized_centers.size(0)

        distmat = torch.pow(features, 2).sum(dim=1, keepdim=True).expand(batch_size, num_centers) + \
                  torch.pow(serialized_centers, 2).sum(dim=1, keepdim=True).expand(num_centers, batch_size).t()
        distmat.addmm_(features, serialized_centers.t(),beta=1,alpha=-2)
        # distmat in shape (batch_size,num_centers)
        pred = distmat.argmin(1)
        if return_logits:
            logits = 1/(1+distmat)
            return pred, logits
        else:
            return pred

def get_l2_pred_b9(features,centers, return_logits=False):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        # features are expected in (batch_size,feat_dim)
        # centers are expected in shape (num_classes,num_subcenters,feat_dim)
        batch_size = features.size(0)
        num_classes, feat_dim = centers.shape
        num_centers = num_classes
        
        serialized_centers = centers.view(-1,feat_dim)
        assert num_centers == serialized_centers.size(0)

        distmat = torch.pow(features, 2).sum(dim=1, keepdim=True).expand(batch_size, num_centers) + \
                  torch.pow(serialized_centers, 2).sum(dim=1, keepdim=True).expand(num_centers, batch_size).t()
        distmat.addmm_(features, serialized_centers.t(),beta=1,alpha=-2)
        # distmat in shape (batch_size,num_centers)
        pred = distmat.argmin(1)
        if return_logits:
            logits_b9 = 1/(1+F.normalize(distmat,p=2))
            logits = 1/(1+distmat)
            return pred, logits, logits_b9
        else:
            return pred

def accuracy_l2(features,centers,targets):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        # features are expected in (batch_size,feat_dim)
        # centers are expected in shape (num_classes,num_subcenters,feat_dim)
        batch_size = targets.size(0)
        num_classes, feat_dim = centers.shape
        num_centers = num_classes
        
        serialized_centers = centers.view(-1,feat_dim)
        assert num_centers == serialized_centers.size(0)

        distmat = torch.pow(features, 2).sum(dim=1, keepdim=True).expand(batch_size, num_centers) + \
                  torch.pow(serialized_centers, 2).sum(dim=1, keepdim=True).expand(num_centers, batch_size).t()
        distmat.addmm_(features, serialized_centers.t(),beta=1,alpha=-2)
        # distmat in shape (batch_size,num_centers)
        pred = distmat.argmin(1)
        correct = pred.eq(targets)

        correct_k = correct.flatten().sum(dtype=torch.float32)
        return correct_k * (100.0 / batch_size)  


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


