import argparse
import template

from options import merge_network_opt

parser = argparse.ArgumentParser(description="HyperSpectral Image Reconstruction Toolbox",conflict_handler='resolve')
parser.add_argument('--exp_name', type=str, default="lade_dun", help="name of experiment")
parser.add_argument('--template', default='lade_dun',
                    help='You can set various templates in option.py')

# Hardware specifications
parser.add_argument("--gpu_id", type=str, default='0')

# Data specifications
parser.add_argument('--data_root', type=str, default='../../datasets/', help='dataset directory')
parser.add_argument('--data_path_CAVE', default='/home/wuzongliang/py/dataset/Data_CASSI/cave_1024_28/', type=str,
                        help='path of data')
parser.add_argument('--data_path_KAIST', default='/home/wuzongliang/py/dataset/Data_CASSI/KAIST_CVPR2021_single/', type=str,
                    help='path of data')
parser.add_argument('--mask_path', default='/home/wuzongliang/py/dataset/Data_CASSI/diffMasks/real_Meng/mask_3d_shift.mat', type=str,
                    help='path of mask')
parser.add_argument('--test_data_path', default='/home/wuzongliang/py/dataset/Data_CASSI/Data_real/Testing_real_data/', type=str,
                    help='path of data')
parser.add_argument('--test_mask_path', default='/home/wuzongliang/py/dataset/Data_CASSI/diffMasks/real_Meng/mask_3d_shift.mat', type=str,
                    help='path of mask')



# Saving specifications
parser.add_argument('--outf', type=str, default='./exp/lade_dun/', help='saving_path')

# Model specifications
parser.add_argument('--method', type=str, default='lade_dun', help='method name')
parser.add_argument('--pretrained_model_path', type=str, default=None, help='pretrained model directory')
parser.add_argument('--resume_ckpt_path', type=str, default=None, help='resumed checkpoint directory')
parser.add_argument('--resume_pre', action='store_true')

parser.add_argument("--input_setting", type=str, default='H',
                    help='the input measurement of the network: H, HM or Y')
parser.add_argument("--input_mask", type=str, default='Phi',
                    help='the input mask of the network: Phi, Phi_PhiPhiT, Mask or None')  # Phi: shift_mask   Mask: mask

# Training specifications
parser.add_argument('--batch_size', type=int, default=5, help='the number of HSIs per batch')
parser.add_argument("--height", default=660, type=int, help='cropped patch height')
parser.add_argument("--width", default=660, type=int, help='cropped patch width')
parser.add_argument("--isTrain", default=True, type=bool, help='train or test')
parser.add_argument("--max_epoch", type=int, default=300, help='total epoch')
parser.add_argument("--scheduler", type=str, default='MultiStepLR', help='MultiStepLR or CosineAnnealingLR')
parser.add_argument("--milestones", type=int, default=[50,100,150,200,250], help='milestones for MultiStepLR')
parser.add_argument("--gamma", type=float, default=0.5, help='learning rate decay for MultiStepLR')
parser.add_argument("--epoch_sam_num", type=int, default=5000, help='the number of samples per epoch')
parser.add_argument("--learning_rate", type=float, default=0.0004)

parser.add_argument("--debug", type=int, default=0)


parser.add_argument("--clip_grad", action='store_true', help='whether clip gradients')

parser.add_argument("--tune", action='store_true', help='control the max_epoch and milestones')
parser.add_argument("--stageLoss", type=bool,default=False)

parser.add_argument("--continue_2stg", type=bool,default=False)

parser.add_argument("--test_mode", type=int, default=0)

# opt = parser.parse_args()
opt = parser.parse_known_args()[0]

if opt.template == 'lade_dun':
    parser = merge_network_opt(parser)

opt = parser.parse_known_args()[0]

template.set_template(opt)

opt.trainset_num = 2500

for arg in vars(opt):
    if vars(opt)[arg] == 'True':
        vars(opt)[arg] = True
    elif vars(opt)[arg] == 'False':
        vars(opt)[arg] = False