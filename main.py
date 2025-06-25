# InfoNCE
import torch
import torch.nn.functional as F

def info_nce(z, temperature=0.1):
    """
    L_i = -log(div(exp(sim(z_i, z_j)/t), sigma_k=1^2N exp(sim(z_i,z_k)/t))
    
    z: Tensor of shape [2N, D]    [A1, A2, B1, B2, ...]
    """
    # Normalize each representation vector
    z = F.normalize(z, dim=1)

    # Compute cosine similarity matrix (2N x 2N)
    sim = torch.matmul(z, z.T)  # cosine similarity since z is normalized

    # Remove similarity with itself (set diagonal to very small number)
    batch_size = z.size(0)
    mask = torch.eye(batch_size, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, -9e15)

    # Scale by temperature
    sim = sim / temperature

    # Construct labels: for index i, positive is i ^ 1 (bitwise xor)   [1 0 3 2 ...]
    labels = torch.arange(batch_size, device=z.device) ^ 1

    # Apply cross-entropy loss
    loss = F.cross_entropy(sim, labels)
    return loss




# SupCon
def contrastiveLoss(self,feature_implicit, feature_explicit, positive_position, mask_all, mask_positive, cuda = True, t=0.1):
      # reference: Facilitating Contrastive Learning of Discourse Relational Senses by Exploiting the Hierarchy of Sense Relations
  
      sim_implicit, sim_explicit, sim_implicit_explicit = self.pair_cosine_similarity(feature_implicit, feature_explicit)
      sim_implicit = torch.exp(sim_implicit / t)
      sim_explicit = torch.exp(sim_explicit / t)
      sim_implicit_explicit = torch.exp(sim_implicit_explicit / t)

      positive_count = positive_position.sum(1)
      negative_position = (~(positive_position.bool())).float()
      dis = torch.div((sim_implicit * (mask_positive - self.args.con1* torch.eye(sim_implicit.size(0)).float().cuda()) + sim_implicit_explicit * mask_positive), ( \
          (sim_implicit * (mask_all - self.args.con1* torch.eye(sim_implicit.size(0)).float().cuda())).sum(1, keepdim=True).repeat([1, sim_implicit.size(0)]) \
           + (sim_implicit_explicit * mask_all).sum(1,keepdim=True).repeat([1, sim_implicit_explicit.size(0)]))) + negative_position
      
      dis_explicit = torch.div(
      (sim_explicit * (mask_positive - self.args.con1 * torch.eye(sim_explicit.size(0)).float().cuda()) + sim_implicit_explicit.T * mask_positive), ( \
              (sim_explicit * (mask_all - self.args.con1* torch.eye(sim_implicit.size(0)).float().cuda())).sum(1, keepdim=True).repeat([1, sim_explicit.size(0)])  \
              + (sim_implicit_explicit * mask_all).sum(0).unsqueeze(1).repeat([1, sim_implicit_explicit.size(0)]))) + negative_position 
      
      loss = (torch.log(dis).sum(1) + torch.log(dis_explicit).sum(1)) / positive_count
      return -loss.mean()





