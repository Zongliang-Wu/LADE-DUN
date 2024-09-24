
from utils import *
seed_everything(
    seed = 3407,
    deterministic = True, 
)
from dataset import dataset
import torch.utils.data as tud
import torch
import torch.nn.functional as F

import time
import datetime
from torch.autograd import Variable
import os
from option import opt
from torch_ema import ExponentialMovingAverage

from tqdm import tqdm

import losses
from schedulers import get_cosine_schedule_with_warmup
from pprint import pprint



pprint(opt)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from architecture import *
def prepare_data(path, file_num, height=660):
    HR_HSI = np.zeros((((height,714,file_num))))
    for idx in range(file_num):
        ####  read HrHSI
        path1 = os.path.join(path) + 'scene' + str(idx+1) + '.mat'
        data = sio.loadmat(path1)
        HR_HSI[:,:,idx] = data['meas_real'][:height, :]
        HR_HSI[HR_HSI < 0] = 0.0
        HR_HSI[HR_HSI > 1] = 1.0
    return HR_HSI

def load_mask(path):
    ## load mask
    data = sio.loadmat(path)
    mask_3d_shift = data['mask_3d_shift']
    mask_3d_shift_s = np.sum(mask_3d_shift ** 2, axis=2, keepdims=False)
    mask_3d_shift_s[mask_3d_shift_s == 0] = 1
    mask_3d_shift = torch.FloatTensor(mask_3d_shift.copy()).permute(2, 0, 1)
    mask_3d_shift_s = torch.FloatTensor(mask_3d_shift_s.copy())
    return mask_3d_shift.unsqueeze(0), mask_3d_shift_s.unsqueeze(0)

HR_HSI_test = prepare_data(opt.test_data_path, 5, height=opt.height)
mask_3d_shift_test, mask_3d_shift_s_test = load_mask(opt.test_mask_path)



# saving path
date_time = str(datetime.datetime.now())
date_time = time2file_name(date_time)
opt.outf = os.path.join(opt.outf, date_time)
if not os.path.exists(opt.outf):
    os.makedirs(opt.outf)

logger = gen_log(opt.outf)

# model
model = model_generator(opt, device=device)

if opt.train_phase==2:
    freeze_dict = dict()
    model_dict = model.state_dict()
    for param in model.parameters():
        param.requires_grad = True

    key_list = ['gt_le.']
    print('freeze key list:',end=' ')
    print(key_list)
    for (k, v) in model_dict.items():
            for kl in key_list:
                if kl in k:
                    freeze_dict[k] = v
                    
    if len(freeze_dict)>0:
        model = freeze_model(model=model, to_freeze_dict=freeze_dict)
    
        if opt.continue_2stg==False:    
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.learning_rate, betas=(0.9, 0.999))



# optimizing
optim_params = []
for k, v in model.named_parameters():
    if v.requires_grad:
        optim_params.append(v)
optimizer = torch.optim.Adam(optim_params, lr=opt.learning_rate, betas=(0.9, 0.999))
ema = ExponentialMovingAverage(optim_params, decay=0.999)
scheduler = get_cosine_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=int(np.floor(opt.trainset_num / opt.batch_size)), 
    num_training_steps=int(np.floor(opt.trainset_num / opt.batch_size)) * opt.max_epoch, 
    eta_min=1e-6)


start_epoch = 0

if opt.resume_ckpt_path: # continue training in the same train_phase
    logger.info(f"===> Loading Checkpoint from {opt.resume_ckpt_path}")
    save_state = torch.load(opt.resume_ckpt_path)
    model.load_state_dict(save_state['model'])
    ema.load_state_dict(save_state['ema'])
    optimizer.load_state_dict(save_state['optimizer'])
    scheduler.load_state_dict(save_state['scheduler'])
    start_epoch = save_state['epoch']
if opt.test_mode==0:
    if opt.resume_pre: # pre-trained syntactic data  full model  

        resume_path = opt.pretrained_model_path
        print("===> Loading Checkpoint from {}".format(resume_path))
        
        save_state = torch.load(resume_path)
        model_dict = model.state_dict()
        sd = save_state['model'] 
        print('miss match keys:')
        state_dict = dict()  
        for k,v in list(sd.items()): 
                if ((k in model_dict.keys()) and (model_dict[k].shape==v.shape)):
                    state_dict[k] = v
                elif 'body_GD.0.' in k:
                    k_new = k.replace('body_GD.0.','body_GD.')
                    state_dict[k_new] = sd.pop(k)
                elif 'body_PM.0.' in k:
                    k_new = k.replace('body_PM.0.','body_PM.')
                    state_dict[k_new] = sd.pop(k)   
                else:
                    print(k,end=',')
        

        model_dict.update(state_dict) 
        missing, unexpected = model.load_state_dict(model_dict,strict=False) 
        print(missing)
        print('\n=====')
        
    # load training data
    CAVE = prepare_data_cave(opt.data_path_CAVE, 205, debug=opt.debug)
    KAIST = prepare_data_KAIST(opt.data_path_KAIST, 30, debug=opt.debug)

    
    criterion = losses.CharbonnierLoss().to(device)
    Dataset = dataset(opt, CAVE, KAIST)
    loader_train = tud.DataLoader(Dataset, num_workers=0, batch_size=opt.batch_size, shuffle=True)
def train(epoch):
    model.train()

    epoch_loss = 0

    start_time = time.time()
    model.train()
    for i, (input, label, Phi) in enumerate(loader_train):
        input, label, Phi,  = Variable(input), Variable(label), Variable(Phi)
        input, label, Phi = input.to(device), label.to(device), Phi.to(device)

        out, log_dict = model(input, Phi,label)
        loss = criterion(out, label) #+criterion(log_dict['stage0_x'], label)*0.5
        
          

        if opt.train_phase==2:
            gamma = 1.0
            prior_z = log_dict['prior_z']
            prior = log_dict['prior']
            diff_loss = criterion(prior, prior_z)
            loss = loss + gamma*diff_loss

        epoch_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        ema.update()
        scheduler.step()

        if i % (100) == 0:
            logger.info('%4d %4d / %4d loss = %.10f time = %s' % (
                epoch + 1, i, len(Dataset) // opt.batch_size, epoch_loss / ((i + 1) * opt.batch_size),
                datetime.datetime.now()))
            if i % (500) == 0:
                checkpoint(model, ema, optimizer, scheduler, epoch, opt.outf, logger)
 
    
    elapsed_time = time.time() - start_time
    epoch_loss = epoch_loss / len(Dataset)
    logger.info('epcoh = %4d , loss = %.10f , time = %4.2f s' % (epoch + 1, epoch_loss, elapsed_time))
    if i % (10) == 0:
        checkpoint(model, ema, optimizer, scheduler, epoch, opt.outf, logger)
    return epoch_loss

def test():
    image_log = {}
    model.eval()
    model.training = False
    for j in tqdm(range(5)):
        with torch.no_grad():
            meas = HR_HSI_test[:,:,j]
            meas = meas / (meas.max() + 1e-7) * 0.9
            meas = torch.FloatTensor(meas)
            
            input = meas.unsqueeze(0)
            input = Variable(input)
            input = input.to(device)
            mask_3d_shift_test_ = mask_3d_shift_test.to(device)
            with ema.average_parameters():
                out, log_dict = model(input, mask_3d_shift_test_)
            result = out
            result = result.clamp(min=0., max=1.)
        result = result.squeeze()
    
        result = torch.flip(result, [1, 2])
        

        image_log[f'scene{j}'] = result.permute(1, 2, 0)

    return image_log


if __name__ == "__main__":

    save_file_id = []
    import scipy.io as sio
    import imageio as io
    import matplotlib.pyplot as plt
    
    image_log = test()

   
    show_R = image_log['scene2'][:,:,26].cpu().detach().numpy()
    plt.figure(figsize=(4,4))
    
    
    plt.imshow(show_R) # cmap='viridis',,  origin='lower'
    plt.savefig(os.path.join(opt.outf, f'results_epoch_{0}_s2.png'))
    if opt.test_mode==0:
        for epoch in range(start_epoch+1, opt.max_epoch):
            train_loss = train(epoch)
            if (epoch % 1 == 0) or (epoch==1):
                image_log = test()

                if len(save_file_id)>5:
                    id_old = save_file_id.pop(0)
                    os.remove(os.path.join(opt.outf, f'results_epoch_{id_old}.pth'))
                torch.save(image_log, os.path.join(opt.outf, f'results_epoch_{epoch}.pth'))
                logger.info("results saved to {}".format(os.path.join(opt.outf, f'results_epoch_{epoch}.pth')))
                
                show_R = image_log['scene2'][:,:,26].cpu().detach().numpy()
                plt.figure(figsize=(4,4))
                
                
                plt.imshow(show_R) # cmap='viridis',,  origin='lower'
                plt.savefig(os.path.join(opt.outf, f'results_epoch_{epoch}_s2.png'))
            
            
                save_file_id.append(epoch)
                
            # torch.cuda.empty_cache()