import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
import os
import numpy as np
import torch
import json
import time
import utils.data_loaders
from utils.data_loaders import PCNDataLoader
from easydict import EasyDict as edict
from importlib import import_module
from pprint import pprint
from manager import Manager

import math
from models.pointmac import CRAPCN, CRAPCN_d
from models.pointmac import ExtendedModel, ExtendedModel2
TRAIN_NAME = os.path.splitext(os.path.basename(__file__))[0]


# Arguments 
parser = argparse.ArgumentParser()
parser.add_argument('--desc', type=str, default='Training/Testing CRA-PCN', help='description')
parser.add_argument('--net_model', type=str, default='model', help='Import module.')
parser.add_argument('--test', dest='test', help='Test neural networks', action='store_true')
parser.add_argument('--inference', dest='inference', help='Inference for benchmark', action='store_true')
parser.add_argument('--output', type=int, default=True, help='Output testing results.')
parser.add_argument('--pretrained', type=str, default='', help='Pretrained path for testing.')
args = parser.parse_args()

# Configuration for PCN
def PCNConfig():
    __C                                              = edict()
    cfg                                              = __C

    #
    # Dataset Config
    #
    __C.DATASETS                                     = edict()
    __C.DATASETS.COMPLETION3D                        = edict()
    __C.DATASETS.COMPLETION3D.CATEGORY_FILE_PATH     = './datasets/Completion3D.json'
    __C.DATASETS.COMPLETION3D.PARTIAL_POINTS_PATH    = '/path/to/datasets/Completion3D/%s/partial/%s/%s.h5'
    __C.DATASETS.COMPLETION3D.COMPLETE_POINTS_PATH   = '/path/to/datasets/Completion3D/%s/gt/%s/%s.h5'
    __C.DATASETS.SHAPENET                            = edict()
    __C.DATASETS.SHAPENET.CATEGORY_FILE_PATH         = './datasets/ShapeNet.json'
    __C.DATASETS.SHAPENET.N_RENDERINGS               = 8
    __C.DATASETS.SHAPENET.N_POINTS                   = 2048
    

    __C.DATASETS.SHAPENET.PARTIAL_POINTS_PATH        =  '../data/PCN/%s/partial/%s/%s/%02d.pcd'
    __C.DATASETS.SHAPENET.COMPLETE_POINTS_PATH       =  '../data/PCN/%s/complete/%s/%s.pcd'

    #
    # Dataset
    #
    __C.DATASET                                      = edict()
    # Dataset Options: Completion3D, ShapeNet, ShapeNetCars, Completion3DPCCT
    # __C.DATASET.TRAIN_DATASET                        = 'ShapeNet'
    # __C.DATASET.TEST_DATASET                         = 'ShapeNet'
    
    ####revise
    __C.DATASET.TRAIN_DATASET                        = 'PCN'
    __C.DATASET.TEST_DATASET                         = 'PCN'

    #
    # Constants
    #
    __C.CONST                                        = edict()

    __C.CONST.NUM_WORKERS                            = 8
    __C.CONST.N_INPUT_POINTS                         = 2048

    #
    # Directories
    #

    __C.DIR                                          = edict()
    __C.DIR.OUT_PATH                                 = 'results/'
    __C.DIR.TEST_PATH                                = 'test/PCN'
    __C.CONST.DEVICE                                 = '0, 1'

    #
    # Network
    #
    __C.NETWORK                                      = edict()
    __C.NETWORK.UPSAMPLE_FACTORS                     = [2, 2, 1, 8] # 16384
    __C.NETWORK.KP_EXTENTS                           = [0.1, 0.1, 0.05, 0.025] # 16384
    #
    # Train
    #
    __C.TRAIN                                        = edict()
    __C.TRAIN.BATCH_SIZE                             = 40
    __C.TRAIN.N_EPOCHS                               = 200
    __C.TRAIN.SAVE_FREQ                              = 25
    __C.TRAIN.LEARNING_RATE                          = 0.001
    __C.TRAIN.LR_MILESTONES                          = [50, 100, 150, 200, 250]
    __C.TRAIN.LR_DECAY_STEP                          = 50
    __C.TRAIN.WARMUP_STEPS                           = 200
    __C.TRAIN.WARMUP_EPOCHS                          = 20
    __C.TRAIN.GAMMA                                  = .5
    __C.TRAIN.BETAS                                  = (.9, .999)
    __C.TRAIN.WEIGHT_DECAY                           = 0
    __C.TRAIN.LR_DECAY                               = 150

    #
    # Test
    #
    __C.TEST                                         = edict()
    __C.TEST.METRIC_NAME                             = 'ChamferDistance'


    return cfg


def train_net(cfg):
    torch.backends.cudnn.benchmark = True


    # train_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TRAIN_DATASET](cfg)
    # val_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
    ###PCN
    train_dataset_loader = ShapeNet('train')
    val_dataset_loader = ShapeNet('test')

    train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset_loader.get_dataset(
        utils.data_loaders.DatasetSubset.TRAIN),
                                                    batch_size=cfg.TRAIN.BATCH_SIZE,
                                                    num_workers=cfg.CONST.NUM_WORKERS,
                                                    collate_fn=utils.data_loaders.collate_fn,
                                                    pin_memory=True,
                                                    shuffle=True,
                                                    drop_last=False)


    val_data_loader = torch.utils.data.DataLoader(dataset=val_dataset_loader.get_dataset(
        utils.data_loaders.DatasetSubset.TEST),
                                                  batch_size=cfg.TRAIN.BATCH_SIZE,
                                                  num_workers=cfg.CONST.NUM_WORKERS//2,
                                                  collate_fn=utils.data_loaders.collate_fn,
                                                  pin_memory=True,
                                                  shuffle=False)


    
    # Set up folders for logs and checkpoints
    timestr = time.strftime('_Log_%Y_%m_%d_%H_%M_%S', time.gmtime())
    cfg.DIR.OUT_PATH = os.path.join(cfg.DIR.OUT_PATH, TRAIN_NAME+timestr)
    cfg.DIR.CHECKPOINTS = os.path.join(cfg.DIR.OUT_PATH, 'checkpoints')
    cfg.DIR.LOGS = cfg.DIR.OUT_PATH
    print('Saving outdir: {}'.format(cfg.DIR.OUT_PATH))
    if not os.path.exists(cfg.DIR.CHECKPOINTS):
        os.makedirs(cfg.DIR.CHECKPOINTS)

    # save config file
    pprint(cfg)
    config_filename = os.path.join(cfg.DIR.LOGS, 'config.json')
    with open(config_filename, 'w') as file:
        json.dump(cfg, file, indent=4, sort_keys=True)

    # Save Arguments
    torch.save(args, os.path.join(cfg.DIR.LOGS, 'args_training.pth'))

    model = CRAPCN() # or 'CRAPCN_d'
    extend1 = ExtendedModel(model)
    extend2 = ExtendedModel2(model, extend1)
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()
        extend1 = torch.nn.DataParallel(extend1).cuda()
        extend2 = torch.nn.DataParallel(extend2).cuda()

    # load existing model
    if 'WEIGHTS' in cfg.CONST:
        print('Recovering from %s ...' % (cfg.CONST.WEIGHTS))
        checkpoint = torch.load(cfg.CONST.WEIGHTS)
        model.load_state_dict(checkpoint['model'])
        print('Recover complete. Current epoch = #%d; best metrics = %s.' % (checkpoint['epoch_index'], checkpoint['best_metrics']))


    ##################
    # Training Manager
    ##################

    manager = Manager(model, extend1, cfg, extend2=extend2)

    # Start training
    manager.train(model,extend1, train_data_loader, val_data_loader, cfg)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    # Check python version
    seed = 1128
    set_seed(seed)
    
    print('cuda available ', torch.cuda.is_available())
    cfg = PCNConfig()
    train_net(cfg)
    

