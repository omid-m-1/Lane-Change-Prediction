import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from params import CLASSIFICATION, REGRESSION, DUAL
import os

def perform_training_epoch(p, model, optimizer, scheduler, loader, lc_loss, ttlc_loss, task, curriculum, epoch, device):
    model.train()
    for batch_idx,(data, labels, _, ttlc_flag) in enumerate(loader):
        data=[d.to(device) for d in data]
        labels=labels.to(device)
        ttlc_flag=ttlc_flag.to(device)

        optimizer.zero_grad()

        out=model(data[0]) # (B,25,200,80)
        lc_pred=out['lc_pred']
        ttlc_pred=out['ttlc_pred']

        if task in [CLASSIFICATION,DUAL]:
            lcl=lc_loss(lc_pred,labels)
        else:
            lcl=0

        if task in [REGRESSION,DUAL]:
            # A placeholder TTLC target, in real scenario set actual target_ttlc
            target_ttlc=torch.zeros_like(ttlc_pred)
            ttl=ttlc_loss(ttlc_pred,target_ttlc)
        else:
            ttl=0

        current_loss=lcl+ttl
        current_loss.backward()
        optimizer.step()

    scheduler.step()

def evaluate_phase(p, model, lc_loss, ttlc_loss, task, loader, device):
    model.eval()
    total_loss=0
    all_preds=[]
    all_labels=[]
    total_count=0
    with torch.no_grad():
        for data,labels,_,ttlc_flag in loader:
            data=[d.to(device) for d in data]
            labels=labels.to(device)
            out=model(data[0])
            lc_pred=out['lc_pred']

            if task in [CLASSIFICATION,DUAL]:
                loss_val=lc_loss(lc_pred,labels)
                total_loss+=loss_val.item()*labels.size(0)
                pred_prob=F.softmax(lc_pred,dim=-1).cpu().numpy()
                pred_cls=np.argmax(pred_prob,axis=-1)
                all_preds.append(pred_cls)
            else:
                total_loss+=0
            all_labels.append(labels.cpu().numpy())
            total_count+=labels.size(0)

    if total_count>0 and all_preds:
        all_preds=np.concatenate(all_preds)
        all_labels=np.concatenate(all_labels)
        acc=(all_preds==all_labels).sum()/total_count
    else:
        acc=0
    avg_loss=total_loss/(total_count if total_count>0 else 1)
    return acc, avg_loss

def train_top_func(p, model, optimizer, lc_loss, ttlc_loss, task, curriculum, tr_dataset, val_dataset, device, model_tag=''):
    tr_loader=DataLoader(tr_dataset,batch_size=p.BATCH_SIZE,shuffle=True,drop_last=True,num_workers=2)
    val_loader=DataLoader(val_dataset,batch_size=p.BATCH_SIZE,shuffle=True,drop_last=True,num_workers=2)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,p.LR_DECAY_EPOCH,p.LR_DECAY)
    best_val_loss=float('inf')
    best_val_acc=0
    best_epoch=0
    patience=p.PATIENCE
    model_path=os.path.join(p.MODELS_DIR,f"{p.SELECTED_DATASET}_{model_tag}.pt")

    for ep in range(1,p.NUM_EPOCHS+1):
        perform_training_epoch(p, model, optimizer, scheduler, tr_loader, lc_loss, ttlc_loss, task, curriculum, ep, device)
        val_acc,val_loss=evaluate_phase(p,model,lc_loss,ttlc_loss,task,val_loader,device)
        if ep>p.CL_EPOCH:
            if val_loss<best_val_loss:
                best_val_loss=val_loss
                best_val_acc=val_acc
                best_epoch=ep
                torch.save(model.state_dict(),model_path)
                patience=p.PATIENCE
            else:
                patience-=1
                if patience<=0:
                    break
    return {'EarlyStopping Epoch':best_epoch,'Best Validaction Acc':best_val_acc,'Best Validation Loss':best_val_loss}

def eval_top_func(p, model, lc_loss, ttlc_loss, task, te_dataset, device, model_tag=''):
    te_loader=DataLoader(te_dataset,batch_size=p.BATCH_SIZE,shuffle=True,drop_last=True,num_workers=2)
    model_path=os.path.join(p.MODELS_DIR,f"{p.SELECTED_DATASET}_{model_tag}.pt")
    model.load_state_dict(torch.load(model_path))
    test_acc,test_loss=evaluate_phase(p,model,lc_loss,ttlc_loss,task,te_loader,device)
    return {'Test Acc': test_acc, 'Test Classification Loss': test_loss}
