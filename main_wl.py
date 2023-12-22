
from genericpath import exists
import os, random, sys
import argparse
import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
import torchvision.transfos as transforms
from collections import namedtuple
import numpy as np
from  creaters.creater import ConvCreater
from creaters.shadow_creater import ShadowCreater
from model_cfg import get_model_fn
from train_weightloss import train_main
from optimizer import get_optimizer
from do_mask import Mask
from train_weightloss import val_pruning, val
from utils import print_log

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='', help='Path to dataset')
parser.add_argument('--weight_path', type=str, default='', help='Path to weight')
parser.add_argument('--dataset', type=str, default='nwpu-45', help='Choose from nwpu-45 and ucml-21')
parser.add_argument('--epoch', type=int, default=300, help='Epoch')

parser.add_argument('--batch_size', type=int, default=64, help='Batchsize')
parser.add_argument('--base_lr', type=float, default=0.001, help='Learning_rate') 
parser.add_argument('--gammas', type=list, default=[0.1, 0.1], help='Learning_rate') 
parser.add_argument('--scheduler', type=list, default=[150, 225], help='Learning_rate') 

parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--weight_decay', type=float, nargs='+', default=1e-4, help='Weight_decay')  


parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--workers', type=int, default='0', help='Number of workers')
parser.add_argument('--manualSeed', type=int, help='manual seed')

parser.add_argument('--arch', type=str, default='sres101', help='Choose from sres101 and sres50')
parser.add_argument('--block_type', type=str, default='shadow', help='Reparameterized or not, shadow is yes, normal is no')
parser.add_argument('--pruning_rate', type=float, default=0.3, help='remianing rate')
parser.add_argument('--layer_begin', type=int, default=0, help='Start layer')
parser.add_argument('--layer_end', type=int, default=0, help='End layer')
parser.add_argument('--layer_inter', type=int, default=3, help='Interval Layer')


args = parser.parse_args()
args.use_cuda = torch.cuda.is_available()

if args.manualSeed is None: 
    args.manualSeed = random.randint(1, 10000) 
random.seed(args.manualSeed) 
torch.manual_seed(args.manualSeed) 
if args.use_cuda: 
    torch.cuda.manual_seed_all(args.manualSeed) 
cudnn.benchmark = True 

if __name__ == '__main__':

    network_type = args.arch 
    block_type = args.block_type
    assert block_type in ['shadow', 'normal']

    ###########################################################################    
    ## prepare dataset
    if args.dataset == 'nwpu-45':  #nwpu-45'
        num_classes = 45
        normalize = transforms.Normalize(mean=[0.36752758, 0.38054053, 0.3431953],
                                std=[0.14521813, 0.13548799, 0.13197055])
    elif args.dataset == 'ucml-21':  #ucml-21'
        num_classes = 21
        normalize = transforms.Normalize(mean=[0.48422759, 0.49005176, 0.45050278],
                                std=[0.17348298, 0.16352356, 0.15547497])

    else:
        assert False, "Unknow dataset : {}".format(args.dataset)

    train_data = dset.ImageFolder(args.data_path+'/train',
    transforms.Compose([
        transforms.RandomResizedCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))
    val_data = dset.ImageFolder(args.data_path+'/val', 
    transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        normalize,
    ]))

    train_loader = torch.utils.data.DataLoader(train_data, batch_size = args.batch_size, shuffle=True,
                                        num_workers=args.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(val_data, batch_size = args.batch_size, shuffle=True,
                                        num_workers=args.workers, pin_memory=True)
    ###############################################################################
    ## define network 
    gamma_init = None

    if args.arch == 'sres50':
        weight_decay = 1e-4
        #   ------------------------------------
        #batch_size = 128
        warmup_epochs = 0
        gamma_init = 0.5
        args.layer_begin = 0
        args.layer_end = 205 # 330-1-3
        args.layer_inter = 3
    if args.arch == 'sres101':
        weight_decay = 1e-4
        #   ------------------------------------
        #batch_size = 128
        warmup_epochs = 0
        gamma_init = 0.5
        args.layer_begin = 0
        args.layer_end = 408 # 330-1-3
        args.layer_inter = 3
    # 构建重参数化结构
    if block_type == 'shadow':
        creater = ShadowCreater(deploy=False, gamma_init=gamma_init)
    else:
        creater = ConvCreater()
    
    net = get_model_fn(args.dataset, args.arch)
    print(net)
    model = net(creater)
    print(model)
    #model = torch.nn.DataParallel(model, device_ids= list(args.gpus))
    model = model.cuda()

    ##Init criterion, optimizer, scheduler
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = get_optimizer(args, model)

    ###############################################################################

    m = Mask(model)
    m.init_length()
    print('-'*10+'pruning begin' + '-'*10)
    print('pruning rate is %f'% args.pruning_rate)

    # top1_val, top5_val, loss_val = val(model = model, criterion = criterion, test_loader = test_loader, epoch = 0, args = args, log = log)
    # print('initial accuracy before pruning is %.3f %%' % top1_val)

    m.model = model
    # m.init_mask(args = args)
    # m.act_mask()

    model = m.model
    if args.use_cuda:
       model = model.cuda()
    # top1_val, top5_val, loss_val = val(model = model, criterion = criterion, test_loader = test_loader, epoch = 0, args = args)
    # print('initial accuracy after pruning is %.3f %%' % top1_val)
    ###############################################################################
    ## begin training
    if not os.path.isdir(args.weight_path): 
        os.makedirs(args.weight_path) 
    log = open(os.path.join(args.weight_path, 'log_seed_{}.txt'.format(args.manualSeed)), 'w') 
    print_log('save path : {}'.format(args.weight_path), log) 
    state = {k: v for k, v in args._get_kwargs()} 
    print_log(state, log) 
    print_log("Random Seed: {}".format(args.manualSeed), log) 
    print_log("python version : {}".format(sys.version.replace('\n', ' ')), log) 
    print_log("torch version : {}".format(torch.__version__), log) 
    print_log("cudnn version : {}".format(torch.backends.cudnn.version()), log) 
    print_log("Pruning Rate: {}".format(args.pruning_rate), log) 
    print_log("Layer Begin: {}".format(args.layer_begin), log) 
    print_log("Layer End: {}".format(args.layer_end), log) 
    print_log("Layer Inter: {}".format(args.layer_inter), log) 
    print_log("Epoch prune: {}".format(args.epoch_pruning), log) 

    train_main(model=model, criterion = criterion, optimizer = optimizer, train_loader = train_loader, 
                         test_loader=test_loader, args = args, log = log, m_mask = m) #


