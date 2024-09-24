import torch.utils.data as tud
import random
import torch
import numpy as np
import scipy.io as sio
# if (opt.mode == '12m' or opt.mode == '121') and patch_size==256:
    
#     input_meas = init_meas(gt, Phi_batch_train, opt.input_setting)
# elif opt.mode == 'm2m' or patch_size<256:
#     mask3d_train_crop = Phi_batch_train = shuffle_crop_mask(Phi_batch_train_orig,  patch_size,opt=opt)
#     input_meas = init_meas(gt, mask3d_train_crop, opt.input_setting)

class dataset(tud.Dataset):
    def __init__(self, opt, CAVE, KAIST):
        super(dataset, self).__init__()
        self.isTrain = opt.isTrain
        self.height = opt.height
        self.width = opt.width
        

        if self.isTrain == True:
            self.num = opt.trainset_num #if opt.debug == False else 2
            
        else:
            self.num = opt.testset_num
        self.CAVE = CAVE
        self.KAIST = KAIST
        self.cave_file_num = len(self.CAVE)
        self.kaist_file_num = len(self.KAIST)
        ## load mask
        data = sio.loadmat(opt.mask_path)
        self.mask_3d_shift = data['mask_3d_shift']

    def __getitem__(self, index):
        if self.isTrain == True:
            # index1 = 0
            d = random.randint(0, 1)
            if d == 0:
                index1 = random.randint(0, self.cave_file_num-1)
                hsi = self.CAVE[index1]
            else:
                index1 = random.randint(0, self.kaist_file_num-1)
                hsi = self.KAIST[index1]
        else:
            index1 = index
            hsi = self.HSI[index1]
        shape = np.shape(hsi)
        ph = random.randint(0, shape[0] - self.height)
        pw = random.randint(0, shape[1] - self.width)
        label = hsi[ph:ph + self.height:1, pw:pw + self.width:1, :]

        if self.height<660:
            ph = random.randint(0, self.mask_3d_shift.shape[0] - self.height)
            pw = random.randint(0, self.mask_3d_shift.shape[1] - self.width - (28 - 1) * 2)
            mask_3d_shift = self.mask_3d_shift[ph:ph + self.height, pw:pw + self.width+(28 - 1) * 2, :]
        else:
            mask_3d_shift = self.mask_3d_shift

        if self.isTrain == True:

            rotTimes = random.randint(0, 3)
            vFlip    = random.randint(0, 1)
            hFlip    = random.randint(0, 1)

            # Random rotation
            for j in range(rotTimes):
                label  =  np.rot90(label)

            # Random vertical Flip
            for j in range(vFlip):
                label = label[:, ::-1, :].copy()

            # Random horizontal Flip
            for j in range(hFlip):
                label = label[::-1, :, :].copy()
        
        temp =  label
        temp_shift = np.zeros((self.height, self.width + (28 - 1) * 2, 28))
        # temp_shift[:, 0:self.width, :] = temp
        
        ###
        # for t in range(28):
        #     temp_shift[:, :, t] = np.roll(temp_shift[:, :, t], 2 * t, axis=1)
        ###
        # temp_shift_out = np.zeros(bs, nC, row, col + (nC - 1) * step)
        for t in range(28):
            temp_shift[:, 2 * t:2 * t + self.width, t] = temp[:,:,t]
        


        meas = np.sum(temp_shift * mask_3d_shift, axis=2)
        # print(f"meas before max: {np.max(meas)} min: {np.min(meas)} mean: {np.mean(meas)}")
        # input = meas / 28 * 2 * 1.2
        input = meas / (meas.max() + 1e-7) * 0.9
        # print(f"meas after: max: {np.max(input)} min: {np.min(input)} mean: {np.mean(input)}")

        QE, bit = 0.4, 2048
        input = np.random.binomial((input * bit / QE).astype(int), QE)
        input = np.float32(input) / np.float32(bit)

        # print(f"input: max: {np.max(input)} min: {np.min(input)} mean: {np.mean(input)}")
        label = torch.FloatTensor(label.copy()).permute(2,0,1)
        input = torch.FloatTensor(input.copy())
        mask_3d_shift = torch.FloatTensor(mask_3d_shift.copy()).permute(2,0,1)
        return input, label, mask_3d_shift

    def __len__(self):
        return self.num