import torch
import params
import utils
import dataset as ds
import models_dict as m

def test_model_dict(model_dict, p):
    device = torch.device("cuda:0" if (p.CUDA and torch.cuda.is_available()) else "cpu")
    model = model_dict['ref'](p.BATCH_SIZE, device, model_dict['hyperparams'], p).to(device)
    lc_loss_func = model_dict['lc loss function']()
    ttlc_loss_func = model_dict['ttlc loss function']()
    task = model_dict['hyperparams']['task']

    # Load training dataset for normalization (if needed)
    tr_dataset = ds.LCDataset(p.TRAIN_DATASET_DIR, p.TR_DATA_FILES, data_type=model_dict['data type'], state_type=model_dict['state type'], store_plot_info=False)
    te_dataset = ds.LCDataset(p.TEST_DATASET_DIR, p.TE_DATA_FILES, data_type=model_dict['data type'], state_type=model_dict['state type'], store_plot_info=True)

    # Evaluate
    result = utils.eval_top_func(p, model, lc_loss_func, ttlc_loss_func, task, te_dataset, device, model_dict['tag'])
    print("Test Results:", result)

if __name__=='__main__':
    p = params.Parameters(SELECTED_MODEL='CNNTRANS', SELECTED_DATASET='HIGHD', UNBALANCED=False, ABLATION=False)
    model_dict = m.MODELS[p.SELECTED_MODEL]
    test_model_dict(model_dict, p)