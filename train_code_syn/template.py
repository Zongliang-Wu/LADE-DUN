def set_template(args):
        
    if args.template.find('lade_dun') >= 0: 
        args.input_setting = 'Y'
        args.input_mask = 'Phi'
    
        if args.stage > 5: 
            args.batch_size = 1

        else:
            pass
        