import torch
from torch.utils.data import Dataset
import os
import numpy as np
import h5py

class LCDataset(Dataset):
    def __init__(self, root_dir, file_list, data_type='image', state_type='', store_plot_info=False):
        super(LCDataset, self).__init__()
        self.root_dir = root_dir
        self.files = [os.path.join(root_dir, f) for f in file_list]
        self.data_type = data_type
        self.state_type = state_type
        self.store_plot_info = store_plot_info

        self.index_limits = []
        running_sum = 0
        for fpath in self.files:
            count = np.load(fpath.replace('.h5','.npy'))
            running_sum += count
            self.index_limits.append(running_sum)
        self.index_limits = np.array(self.index_limits)
        self.total_samples = running_sum

        self.image_only = (data_type=='image')

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        file_idx = np.argmax(self.index_limits>idx)
        if file_idx>0:
            local_idx = idx - self.index_limits[file_idx-1] -1
        else:
            local_idx = idx -1

        with h5py.File(self.files[file_idx],'r') as hf:
            img_data = hf['image_data']
            labels = hf['labels']
            ttlc_avail = hf['ttlc_available']

            lbl = labels[local_idx].astype(np.long)
            ttlc_flag = ttlc_avail[local_idx].astype(np.long)

            if self.store_plot_info:
                frame_data = hf['frame_data']
                tv_data = hf['tv_data']
                tv_id = tv_data[local_idx]
                frames_info = frame_data[local_idx]
                extra = [tv_id, frames_info, self.files[file_idx]]
            else:
                extra = ()

            # Original image data: (25,200,80)
            sample = img_data[local_idx].astype(np.float32)

            # Return as is. Model will reshape to (B,5,5,200,80).
            data_out = [torch.from_numpy(sample)] # shape (25,200,80)

        return data_out, lbl, extra, ttlc_flag
