import argparse
import template

from options import merge_network_opt

parser = argparse.ArgumentParser(description="HyperSpectral Image Reconstruction Toolbox",conflict_handler='resolve')
parser.add_argument('--exp_name', type=str, default="lade_dun", help="name of experiment")
parser.add_argument('--template', default='mst',
                    help='You can set various templates in option.py')

# Hardware specifications
parser.add_argument("--gpu_id", type=str, default='0')

# Data specifications
parser.add_argument('--data_root', type=str, default='/home/wuzongliang/py/dataset/Data_CASSI/', help='dataset directory')

# Saving specifications
parser.add_argument('--outf', type=str, default='./exp/mst_s/', help='saving_path')

# Model specifications
parser.add_argument('--method', type=str, default='mst_s', help='method name')
parser.add_argument('--pretrained_model_path', type=str, default=None, help='pretrained model directory')
parser.add_argument('--resume_ckpt_path', type=str, default=None, help='resumed checkpoint directory')
parser.add_argument("--input_setting", type=str, default='H',
                    help='the input measurement of the network: H, HM or Y')
parser.add_argument("--input_mask", type=str, default='Phi',
                    help='the input mask of the network: Phi, Phi_PhiPhiT, Mask or None')  # Phi: shift_mask   Mask: mask

# Training specifications
parser.add_argument('--batch_size', type=int, default=4, help='the number of HSIs per batch')
parser.add_argument("--patch_size", default=256, type=int, help='cropped patch, ground truth size')
parser.add_argument("--max_epoch", type=int, default=300, help='total epoch')
parser.add_argument("--scheduler", type=str, default='MultiStepLR', help='MultiStepLR or CosineAnnealingLR')
parser.add_argument("--milestones", type=int, default=[50,100,150,200,250], help='milestones for MultiStepLR')
parser.add_argument("--gamma", type=float, default=0.5, help='learning rate decay for MultiStepLR')
parser.add_argument("--epoch_sam_num", type=int, default=5000, help='the number of samples per epoch')
parser.add_argument("--learning_rate", type=float, default=0.0004)
parser.add_argument("--debug", type=int, default=0)
parser.add_argument("--clip_grad", action='store_true', help='whether clip gradients')


parser.add_argument("--test_mode", type=bool,default=False)
parser.add_argument("--ema_decay", type=float,default=0.999)
parser.add_argument("--mask_input", type=bool,  default=False, help='whether use multiply random mask on measurement')

# opt = parser.parse_args()
opt = parser.parse_known_args()[0]

if opt.template == 'lade_dun' :
    parser = merge_network_opt(parser)

opt = parser.parse_known_args()[0]

# opt = parser.parse_args()
template.set_template(opt)

# dataset
opt.data_path = f"{opt.data_root}/cave_1024_28/"
opt.mask_path = f"{opt.data_root}"
opt.test_path = f"{opt.data_root}/Kaist_test/"

for arg in vars(opt):
    if vars(opt)[arg] == 'True':
        vars(opt)[arg] = True
    elif vars(opt)[arg] == 'False':
        vars(opt)[arg] = False