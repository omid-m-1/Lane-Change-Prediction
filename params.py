import os
import numpy as np

CLASSIFICATION = 0
REGRESSION = 1
DUAL = 2

class Parameters:
    def __init__(self, SELECTED_MODEL='CNNTRANS', SELECTED_DATASET='HIGHD', UNBALANCED=False, ABLATION=False):
        self.SELECTED_MODEL = SELECTED_MODEL
        self.SELECTED_DATASET = SELECTED_DATASET
        self.UNBALANCED = UNBALANCED
        self.ABLATION = ABLATION

        self.ROBUST_PREDICTOR = True
        self.DATASETS = {
            'HIGHD': {
                'abb_tr_ind': range(1,46),
                'abb_val_ind': range(46,51),
                'abb_te_ind': range(51,56),
                'tr_ind': range(1,51),
                'val_ind': range(51,56),
                'te_ind': range(56,61),
                'image_width': 80,
                'image_height': 200,
            }
        }

        self.IMAGE_HEIGHT = self.DATASETS[self.SELECTED_DATASET]['image_height']
        self.IMAGE_WIDTH = self.DATASETS[self.SELECTED_DATASET]['image_width']

        if self.ABLATION:
            self.TR_DATA_FILES = [f"{i:02d}.h5" for i in self.DATASETS[self.SELECTED_DATASET]['abb_tr_ind']]
            self.VAL_DATA_FILES = [f"{i:02d}.h5" for i in self.DATASETS[self.SELECTED_DATASET]['abb_val_ind']]
            self.TE_DATA_FILES = [f"{i:02d}.h5" for i in self.DATASETS[self.SELECTED_DATASET]['abb_te_ind']]
        else:
            self.TR_DATA_FILES = [f"{i:02d}.h5" for i in self.DATASETS[self.SELECTED_DATASET]['tr_ind']]
            self.VAL_DATA_FILES = [f"{i:02d}.h5" for i in self.DATASETS[self.SELECTED_DATASET]['val_ind']]
            self.TE_DATA_FILES = [f"{i:02d}.h5" for i in self.DATASETS[self.SELECTED_DATASET]['te_ind']]

        self.FPS = 5
        self.SEQ_LEN = 50
        self.IN_SEQ_LEN = 25
        self.ACCEPTED_GAP = 0
        self.THR = 0.34

        self.CUDA = True
        self.BATCH_SIZE = 2  # example batch size
        self.LR = 0.001
        self.LR_DECAY = 1
        self.LR_DECAY_EPOCH = 10
        self.NUM_EPOCHS = 5
        self.PATIENCE = 2
        self.TR_JUMP_STEP = 1

        if self.UNBALANCED:
            self.unblanaced_ext = 'U'
        else:
            self.unblanaced_ext = ''

        self.TRAIN_DATASET_DIR = '/mnt/huge_26TB/IMDB/highD/Processed_highD/RenderedDataset/'
        self.TEST_DATASET_DIR = f'/mnt/huge_26TB/IMDB/highD/Processed_highD/RenderedDataset{self.unblanaced_ext}/'

        self.MAX_PRED_TIME = self.SEQ_LEN - self.IN_SEQ_LEN + 1
        self.cl_step = 5
        self.start_seq_arange = np.arange(self.MAX_PRED_TIME-1,0,-1*self.cl_step)
        self.CL_EPOCH = len(self.start_seq_arange)
        self.START_SEQ_CL = np.concatenate((self.start_seq_arange, np.zeros((self.NUM_EPOCHS-len(self.start_seq_arange)))), axis=0)
        self.END_SEQ_CL = np.ones((self.NUM_EPOCHS))*(self.SEQ_LEN-self.IN_SEQ_LEN+1)
        self.LOSS_RATIO_CL = np.concatenate((np.arange(self.CL_EPOCH)/self.CL_EPOCH, np.ones((self.NUM_EPOCHS-self.CL_EPOCH))), axis=0)
        self.LOSS_RATIO_NoCL = 1

        self.MODELS_DIR = 'models/'
        self.RESULTS_DIR = 'results/'
        self.TABLES_DIR = self.RESULTS_DIR + 'tables/'
        self.FIGS_DIR = self.RESULTS_DIR + 'figures/'
        self.VIS_DIR = self.RESULTS_DIR + 'vis_data/'

        for d in [self.MODELS_DIR,self.RESULTS_DIR,self.TABLES_DIR,self.FIGS_DIR,self.VIS_DIR]:
            if not os.path.exists(d):
                os.makedirs(d, exist_ok=True)

