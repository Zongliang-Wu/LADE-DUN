

def model_generator(opt, device="cuda"):
    if opt.method=='lade_dun':
        from .LADE_DUN_arch import LADE_DUN
        model = LADE_DUN(opt).cuda()  
    if opt.method=='lade_dun_ckpt':
        from .LADE_DUN_arch_ckpt import LADE_DUN
        model = LADE_DUN(opt).cuda()  
    else:
        print(f'opt.Method {opt.method} is not defined !!!!')
    
    return model

