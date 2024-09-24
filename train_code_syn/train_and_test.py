import os
from pprint import pprint
from option import opt
pprint(opt)

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

print(os.environ["CUDA_VISIBLE_DEVICES"])

from utils_mix import *

seed_everything(
    seed = 3407,
    deterministic = True, 
)

import torch
from torch.nn.utils import clip_grad_norm_

from torch_ema import ExponentialMovingAverage


import time

from ptflops import get_model_complexity_info
import numpy as np
from torch.autograd import Variable
import datetime

from tqdm import tqdm

import losses
from schedulers import get_cosine_schedule_with_warmup

device = torch.device("cuda" if torch.cuda.is_available() else "mps")
from architecture import *
# init mask
# noise_act_dict = {'softplus': nn.Softplus(),}
Phi_batch_train = init_mask(
    opt.mask_path, 
    opt.input_mask, 
    opt.batch_size, 
    device=device)
Phi_batch_test = init_mask(
    opt.mask_path, 
    opt.input_mask, 
    1, 
    device=device)

# dataset
if opt.test_mode==0:
    train_set = LoadTraining(opt.data_path, debug=opt.debug)
test_data = LoadTest(opt.test_path)

# saving path
date_time = str(datetime.datetime.now())
date_time = time2file_name(date_time)
result_path = opt.outf + date_time + '/result/'
model_path = opt.outf + date_time + '/model/'
if not os.path.exists(result_path):
    os.makedirs(result_path)
if not os.path.exists(model_path):
    os.makedirs(model_path)
    
if opt.train_phase==1:
    opt.max_epoch = opt.max_epoch*2
else:
    if opt.batch_size>4:
        opt.epoch_sam_num = opt.epoch_sam_num*(opt.batch_size//4) 
        
if opt.debug:
    opt.epoch_sam_num = opt.batch_size*15
elif opt.stage==9 and opt.batch_size<=2:
    opt.epoch_sam_num = opt.epoch_sam_num//2
start_epoch = 0
# model

model = model_generator(method=opt.method,opt=opt)


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
    



# optimizing
optim_params = []
for k, v in model.named_parameters():
    if v.requires_grad:
        optim_params.append(v)
optimizer = torch.optim.Adam(optim_params, lr=opt.learning_rate, betas=(0.9, 0.999))
ema = ExponentialMovingAverage(optim_params, decay=0.999)

para_ema_sh = sum([np.prod(list(p.size())) for p in ema.shadow_params])
if ema.collected_params is not None:
    para_ema_co = sum([np.prod(list(p.size())) for p in ema.collected_params])
para_ema_up = ema.num_updates
scheduler = get_cosine_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=int(np.floor(opt.epoch_sam_num / opt.batch_size)), 
    num_training_steps=int(np.floor(opt.epoch_sam_num / opt.batch_size)) * opt.max_epoch, 
    eta_min=1e-6)
if opt.resume_ckpt_path:
    print("===> Loading Checkpoint from {}".format(opt.resume_ckpt_path))
    save_state = torch.load(opt.resume_ckpt_path)

    model_dict = model.state_dict()
    

    if opt.train_phase==2:
        ckpt_ema_shadow_params = save_state['ema']['shadow_params']
        ckpt_ema_shadow_params_size = sum([np.prod(list(p.size())) for p in ckpt_ema_shadow_params])
        if ckpt_ema_shadow_params_size==para_ema_sh:
            ema.load_state_dict(save_state['ema'])
        else:
            print('ema load failed')
            
            
        sd = save_state['model'] 
        print('miss match keys:')
        state_dict = dict()
        for k,v in sd.items():
                if ((k in model_dict.keys()) and (model_dict[k].shape==v.shape)):
                    state_dict[k] = v
                else:
                    print(k,end=',')
        

        model_dict.update(state_dict) 
        missing, unexpected = model.load_state_dict(model_dict,strict=True) 
        print(unexpected)
        print(missing)
        start_epoch = 0

    else:
        try:
            model.load_state_dict(save_state['model'])
            ema.load_state_dict(save_state['ema'])
            optimizer.load_state_dict(save_state['optimizer'])
            scheduler.load_state_dict(save_state['scheduler'])
            start_epoch = save_state['epoch']
        except:
            model_dict = model.state_dict()

            sd = save_state['model'] 
            print('miss match keys:')
            state_dict = dict()
            for k,v in sd.items():
                    if ((k in model_dict.keys()) and (model_dict[k].shape==v.shape)):
                        state_dict[k] = v
            model_dict.update(state_dict) 
            missing, unexpected = model.load_state_dict(model_dict,strict=False) 
            print(missing)
    print('\n=====')      


criterion = losses.CharbonnierLoss().to(device)

lrs = []
patch_size = opt.patch_size
Phi_batch_train_orig = Phi_batch_train.clone()

print_per_layer_stat = True if opt.debug else False


flops_input_size2 = (28,256,256)
flops_input_size = (256,310)

def train(epoch, logger):
    epoch_loss = 0

    begin = time.time()
    batch_num = int(np.floor(opt.epoch_sam_num / opt.batch_size))
    train_tqdm = tqdm(range(batch_num))
    
    if epoch == 1:
        if opt.train_phase==1:
            flops = get_model_complexity_info(model, [flops_input_size,(28,256,310),flops_input_size2],output_precision=3,
                                          print_per_layer_stat=print_per_layer_stat)
        else:
            flops = get_model_complexity_info(model, [flops_input_size,(28,256,310)],output_precision=3,
                                          print_per_layer_stat=print_per_layer_stat)
        print('flops:')
        print(flops)
    
    model.train()
    for i in train_tqdm:
        gt_batch = shuffle_crop(train_set, opt.batch_size,patch_size)
        gt = Variable(gt_batch).to(device)

            
        input_meas = init_meas(gt, Phi_batch_train_orig, opt.input_setting)
        Phi_batch_train = Phi_batch_train_orig
       
        model_out, log_dict = model(input_meas, Phi_batch_train,gt)
        loss = criterion(model_out, gt) 
        # if opt.stageLoss:    
        # stageLoss = criterion(log_dict['stage0_x'], gt) 
        # loss = loss + stageLoss*0.5
        
       
        if opt.train_phase==2:
            gamma = 1.0
            prior_z = log_dict['prior_z']
            prior = log_dict['prior']
            diff_loss = criterion(prior, prior_z)
            loss = loss + gamma*diff_loss
                
            

        
        
        loss.backward()

        clip_grad_norm_(model.parameters(), max_norm=0.2)
        optimizer.step()
        optimizer.zero_grad()
        ema.update()

        train_tqdm.set_postfix(train_loss="{:.4f}".format(loss.item()))
        epoch_loss += loss.data

        lr = optimizer.state_dict()['param_groups'][0]['lr']
        lrs.append(lr)
        scheduler.step()
        
        
    end = time.time()
    train_loss = epoch_loss / batch_num

        
    logger.info("===> Epoch {} Complete: Avg. Loss: {:.6f} lr: {:.1f}e-4 time: {:.2f} ".
            format(epoch, train_loss, lr*10e3, (end - begin)))
    return train_loss


def test(epoch, logger,mask3d_batch_test=None, diff_test=False):
    psnr_list, ssim_list = [], []
    test_gt = test_data.to(torch.float32).to(device)

    input_meas = init_meas(test_gt, mask3d_batch_test, opt.input_setting)

    model.eval()
    begin = time.time()
    image_log = {}
    
    pred = []
    for k in range(test_gt.shape[0]) if diff_test==False else range(test_gt.shape[0]):
        with torch.no_grad():
            with ema.average_parameters():
                if opt.train_phase==1:
                        model_out, log_dict = model(input_meas[k].unsqueeze(0), mask3d_batch_test,test_gt[k].unsqueeze(0))
                else:
                        model_out, log_dict = model(input_meas[k].unsqueeze(0), mask3d_batch_test)       
            pred.append(model_out)
        psnr_val = torch_psnr(model_out[0, :, :, :], test_gt[k, :, :, :])
        ssim_val = torch_ssim(model_out[0, :, :, :], test_gt[k, :, :, :])
        psnr_list.append(psnr_val.detach().cpu().numpy())
        ssim_list.append(ssim_val.detach().cpu().numpy())
    end = time.time()

    pred = torch.cat(pred, dim = 0)
    pred = np.transpose(pred.detach().cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
    truth = np.transpose(test_gt.cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
    psnr_mean = np.mean(np.asarray(psnr_list))
    ssim_mean = np.mean(np.asarray(ssim_list))
    logger.info('===> Epoch {}: testing psnr = {:.2f}, ssim = {:.3f}, time: {:.2f}'
                .format(epoch, psnr_mean, ssim_mean,(end - begin)))


    return pred, truth, psnr_list, ssim_list, psnr_mean, ssim_mean, image_log


def main():
    logger = gen_log(model_path)
    logger.info(opt)
    logger.info("Learning rate:{}, batch_size:{}.\n".format(opt.learning_rate, opt.batch_size))
    psnr_max = 0


    
    print('init test results:')
    (pred, truth, psnr_all, ssim_all, psnr_mean, ssim_mean, image_log) = test(0, logger,Phi_batch_test)
    if opt.test_mode==0:
        for epoch in range(start_epoch + 1, opt.max_epoch + 1):

            print("==>Epoch{epoch}")
            train_loss = train(epoch, logger)

            (pred, truth, psnr_all, ssim_all, psnr_mean, ssim_mean, image_log) = test(epoch, logger,Phi_batch_test)
            if psnr_mean > psnr_max:
                psnr_max = psnr_mean
                if psnr_mean > 28 or (opt.train_phase==1 and opt.maskLoss and psnr_mean > 8):
                    name = result_path + '/' + 'Test_{}_{:.2f}_{:.3f}'.format(epoch, psnr_max, ssim_mean) + '.mat'
                    # scio.savemat(name, {'truth': truth, 'pred': pred, 'psnr_list': psnr_all, 'ssim_list': ssim_all})
                    checkpoint(model, ema, optimizer, scheduler, epoch, model_path, logger)
            
            # torch.cuda.empty_cache()

if __name__ == '__main__':
    main()


