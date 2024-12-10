import torch
import params
import utils
import dataset as ds
import models_dict as m

def train_model_dict(model_dict, p):
    device = torch.device("cuda:0" if (p.CUDA and torch.cuda.is_available()) else "cpu")
    torch.manual_seed(1)
    if device.type=='cuda':
        torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic=True

    model=model_dict['ref'](p.BATCH_SIZE, device, model_dict['hyperparams'], p).to(device)
    optimizer=model_dict['optimizer'](model.parameters(), lr=p.LR)
    lc_loss_func=model_dict['lc loss function']()
    ttlc_loss_func=model_dict['ttlc loss function']()
    task=model_dict['hyperparams']['task']

    curriculum={'seq':model_dict['hyperparams']['curriculum seq'],
                'loss':model_dict['hyperparams']['curriculum loss'],
                'virtual':model_dict['hyperparams']['curriculum virtual']}

    tr_dataset=ds.LCDataset(p.TRAIN_DATASET_DIR,p.TR_DATA_FILES,data_type=model_dict['data type'],state_type=model_dict['state type'],store_plot_info=False)
    val_dataset=ds.LCDataset(p.TRAIN_DATASET_DIR,p.VAL_DATA_FILES,data_type=model_dict['data type'],state_type=model_dict['state type'],store_plot_info=False)
    te_dataset=ds.LCDataset(p.TEST_DATASET_DIR,p.TE_DATA_FILES,data_type=model_dict['data type'],state_type=model_dict['state type'],store_plot_info=True)

    val_res=utils.train_top_func(p,model,optimizer,lc_loss_func,ttlc_loss_func,task,curriculum,tr_dataset,val_dataset,device,model_dict['tag'])
    te_res=utils.eval_top_func(p,model,lc_loss_func,ttlc_loss_func,task,te_dataset,device,model_dict['tag'])

    log_file=p.TABLES_DIR+p.SELECTED_DATASET+'_'+model_dict['name']+'.csv'
    log_dict=model_dict['hyperparams'].copy()
    log_dict['state type']=model_dict['state type']
    log_dict.update(val_res)
    log_dict.update(te_res)
    columns=', '.join(log_dict.keys())+'\n'
    line=', '.join(str(v) for v in log_dict.values())+'\n'
    mode='a' if torch.os.path.exists(log_file) else 'w'
    if mode=='w':
        line=columns+line
    with open(log_file,mode) as f:
        f.write(line)

if __name__=='__main__':
    p=params.Parameters(SELECTED_MODEL='CNNTRANS',SELECTED_DATASET='HIGHD',UNBALANCED=False,ABLATION=False)
    model_dict=m.MODELS[p.SELECTED_MODEL]
    train_model_dict(model_dict,p)