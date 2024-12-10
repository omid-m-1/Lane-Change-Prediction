import torch
import torch.nn as nn
import torch.nn.functional as F
import models
import params

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        if alpha is None:
            alpha = torch.ones(12)*0.5
        self.alpha=alpha
        self.gamma=gamma
        self.reduction=reduction
    def forward(self, inputs, targets):
        log_probs=F.log_softmax(inputs,dim=-1)
        probs=torch.exp(log_probs)
        target_logp=log_probs.gather(1,targets.unsqueeze(1)).squeeze(1)
        target_p=probs.gather(1,targets.unsqueeze(1)).squeeze(1)
        alpha_t=self.alpha[targets].to(inputs.device)
        loss_val=-alpha_t*(1 - target_p)**self.gamma*target_logp
        if self.reduction=='mean':
            return loss_val.mean()
        elif self.reduction=='sum':
            return loss_val.sum()
        return loss_val

MODELS={
    'CNNTRANS':{
        'name':'CNNTRANS',
        'ref':models.CNNTransformerModel,
        'disc':'CNN+Transformer with 5 pipelines,6 layers each,adaptive pool,12 classes,TTLC',
        'tag':'',
        'hyperparams':{
            'task':params.CLASSIFICATION,
            'curriculum loss':False,
            'curriculum seq':False,
            'curriculum virtual':False,
        },
        'optimizer':torch.optim.Adam,
        'lc loss function':FocalLoss,
        'ttlc loss function':nn.MSELoss,
        'data type':'image',
        'state type':'',
    }
}
