
def get_hyperparams(args, test_code=0):
    if not test_code:
        if args.hyperparams_set_index == 0: 
            args.epochs = 100
            args.start_eval_epoch = 90
            args.batch_size = 128 * 1

            ### optimizer
            args.optimizer = 'SGD'
            args.lr = 0.02 * args.batch_size / 256
            args.momentum = 0.9
            args.weigh_decay_apply_on_all = True 
            args.weight_decay = 5e-5
            args.nesterov = True
            
            ### lr scheduler
            args.scheduler = 'cosine'
            args.warmup_epoch = 0
            args.warmup_lr = args.lr * 0.1

            return args
        
        if args.hyperparams_set_index == 1: 
            args.epochs = 100
            args.start_eval_epoch = 90
            args.batch_size = 256 * 1

            ### optimizer
            args.optimizer = 'SGD'
            args.lr = 0.02 * args.batch_size / 256
            args.momentum = 0.9
            args.weigh_decay_apply_on_all = True 
            args.weight_decay = 5e-5
            args.nesterov = True
            
            ### lr scheduler
            args.scheduler = 'cosine'
            args.warmup_epoch = 0
            args.warmup_lr = args.lr * 0.1
            args.lr_min = 0.
            return args

        elif args.hyperparams_set_index == 2: 
            args.epochs = 100
            args.start_eval_epoch = 90
            args.batch_size = 256 * 2

            ### optimizer
            args.optimizer = 'SGD'
            args.lr = 0.02 * args.batch_size / 256
            args.momentum = 0.9
            args.weigh_decay_apply_on_all = True 
            args.weight_decay = 5e-5
            args.nesterov = True
            
            ### lr scheduler
            args.scheduler = 'cosine'
            args.warmup_epoch = 0
            args.warmup_lr = args.lr * 0.1

            return args
        
        elif args.hyperparams_set_index == 21: 
            args.epochs = 100
            args.start_eval_epoch = 90
            args.batch_size = 256 * 2

            ### optimizer
            args.optimizer = 'SGD'
            args.lr = 0.01 * args.batch_size / 256
            args.momentum = 0.9
            args.weigh_decay_apply_on_all = True 
            args.weight_decay = 5e-5
            args.nesterov = True
            
            ### lr scheduler
            args.scheduler = 'cosine'
            args.warmup_epoch = 0
            args.warmup_lr = args.lr * 0.1

            return args
        
        elif args.hyperparams_set_index == 22: 
            args.epochs = 100
            args.start_eval_epoch = 90
            args.batch_size = 256 * 2

            ### optimizer
            args.optimizer = 'SGD'
            args.lr = 0.04 * args.batch_size / 256
            args.momentum = 0.9
            args.weigh_decay_apply_on_all = True 
            args.weight_decay = 5e-5
            args.nesterov = True
            
            ### lr scheduler
            args.scheduler = 'cosine'
            args.warmup_epoch = 0
            args.warmup_lr = args.lr * 0.1

            return args
        
        elif args.hyperparams_set_index == 23: 
            args.epochs = 100
            args.start_eval_epoch = 90
            args.batch_size = 256 * 2

            ### optimizer
            args.optimizer = 'SGD'
            args.lr = 0.005 * args.batch_size / 256
            args.momentum = 0.9
            args.weigh_decay_apply_on_all = True 
            args.weight_decay = 5e-5
            args.nesterov = True
            
            ### lr scheduler
            args.scheduler = 'cosine'
            args.warmup_epoch = 0
            args.warmup_lr = args.lr * 0.1

            return args

        elif args.hyperparams_set_index == 3: 
            args.epochs = 100
            args.start_eval_epoch = 90
            args.batch_size = 256 * 4

            ### optimizer
            args.optimizer = 'SGD'
            args.lr = 0.02 * args.batch_size / 256
            args.momentum = 0.9
            args.weigh_decay_apply_on_all = True 
            args.weight_decay = 5e-5
            args.nesterov = True

            ### lr scheduler
            args.scheduler = 'cosine'
            args.warmup_epoch = 0
            args.warmup_lr = args.lr * 0.1

            return args
        
        elif args.hyperparams_set_index == 30: 
            args.epochs = 10
            args.start_eval_epoch = 90
            args.batch_size = 256 * 4

            ### optimizer
            args.optimizer = 'SGD'
            args.lr = 0.02 * args.batch_size / 256
            args.momentum = 0.9
            args.weigh_decay_apply_on_all = True 
            args.weight_decay = 5e-5
            args.nesterov = True

            ### lr scheduler
            args.scheduler = 'cosine'
            args.warmup_epoch = 0
            args.warmup_lr = args.lr * 0.1

            return args

        elif args.hyperparams_set_index == 4: 
            args.epochs = 100
            args.start_eval_epoch = 90
            args.batch_size = 256 * 8

            ### optimizer
            args.optimizer = 'SGD'
            args.lr = 0.02 * args.batch_size / 256
            args.momentum = 0.9
            args.weigh_decay_apply_on_all = True 
            args.weight_decay = 5e-5
            args.nesterov = True

            ### lr scheduler
            args.scheduler = 'cosine'
            args.warmup_epoch = 0
            args.warmup_lr = args.lr * 0.1

            return args
        
        elif args.hyperparams_set_index == 5: 
            args.epochs = 300
            args.start_eval_epoch = 0
            args.batch_size = 256 * 4

            ### optimizer
            args.optimizer = 'SGD'
            args.lr = 0.1 * args.batch_size / 256
            args.momentum = 0.9
            args.weigh_decay_apply_on_all = True 
            args.weight_decay = 5e-5
            args.nesterov = True

            ### lr scheduler
            args.scheduler = 'cosine'
            args.warmup_epoch = 10
            args.warmup_lr = args.lr * 0.1

            return args

        elif args.hyperparams_set_index == 6: 
            args.epochs = 200
            args.start_eval_epoch = 0
            args.batch_size = 256 * 4

            ### optimizer
            args.optimizer = 'SGD'
            args.lr = 0.02 * args.batch_size / 256
            args.momentum = 0.9
            args.weigh_decay_apply_on_all = True 
            args.weight_decay = 5e-5
            args.nesterov = True

            ### lr scheduler
            args.scheduler = 'cosine'
            args.warmup_epoch = 0
            args.warmup_lr = args.lr * 0.1

            return args
        
        elif args.hyperparams_set_index == 7: 
            args.epochs = 300
            args.start_eval_epoch = 90
            args.batch_size = 256 * 4

            ### optimizer
            args.optimizer = 'SGD'
            args.lr = 0.02 * args.batch_size / 256
            args.momentum = 0.9
            args.weigh_decay_apply_on_all = True 
            args.weight_decay = 5e-5
            args.nesterov = True

            ### lr scheduler
            args.scheduler = 'cosine'
            args.warmup_epoch = 0
            args.warmup_lr = args.lr * 0.1

            return args
    else:
        args.epochs = 90
        args.start_eval_epoch = 0
        args.batch_size = 128

        ### optimizer
        args.optimizer = 'SGD'
        args.lr = 0.05
        args.momentum = 0.9
        args.weigh_decay_apply_on_all = False 
        args.weight_decay = 1e-4
        args.nesterov = True
        ### lr scheduler
        args.scheduler = 'multistep'
        args.lr_decay_rate = 0.1
        args.lr_decay_step = 30

        return args