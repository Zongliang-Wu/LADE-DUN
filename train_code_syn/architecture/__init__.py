



def model_generator(method, opt=None):
    
    if method=='lade_dun':
        if opt.stage==3:
            from .LADE_DUN_arch import LADE_DUN
        elif opt.stage>3:
            from .LADE_DUN_multi_stage_arch import LADE_DUN
           
        model = LADE_DUN(opt).cuda()  
    else:
        print(f'Method {method} is not defined !!!!')

    return model


