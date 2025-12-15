import torch
# import torch.nn as nn
# import torch.nn.functional as F
import torch.utils.data as Data
from scipy.io import loadmat
import numpy as np
import time
import os
import argparse
import logging
from models.CMFDNet import CMFDNET
from utils import (trPixel2Patch, tsPixel2Patch, set_seed,LabelSmoothingCrossEntropyLoss,
                   output_metric, train_epoch, valid_epoch, draw_classification_map)

# -------------------------------------------------------------------------------
# Parameter Setting
parser = argparse.ArgumentParser(description="Training for CMFDNet")
parser.add_argument('--gpu_id', default='0',
                    help='gpu id')
parser.add_argument('--seed', type=int, default=1,
                    help='number of seed')
parser.add_argument('--batch_size', type=int, default=64,
                    help='number of batch size')
parser.add_argument('--epochs', type=int, default=500,
                    help='number of epoch')
parser.add_argument('--dataset', choices=['Houston', 'Augsburg', 'Muufl', 'Trento'], default='Muufl',
                    help='dataset to use')
parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='learning rate')
parser.add_argument('--flag', choices=['train', 'test'], default='train',
                    help='testing mark')
parser.add_argument('--patch_size', type=int, default=18,
                    help='cnn input size')
parser.add_argument('--wavename', type=str, default='db2',
                    help='type of wavelet')
parser.add_argument('--attn_kernel_size', type=int, default=9,
                    help='')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

# -------------------------------------------------------------------------------
# create log
logger = logging.getLogger("Trainlog")
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
file_handler = logging.FileHandler("cls_logs/{}/{}_{}.log".format(args.dataset, args.flag, args.dataset))
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)


# -------------------------------------------------------------------------------
def train1time():
    # -------------------------------------------------------------------------------
    if args.dataset == 'Houston':
        num_classes = 15
        DataPath1 = 'data/Houston/HSI.mat'
        DataPath2 = 'data/Houston/LiDAR.mat'
        LabelPath1 = 'data/Houston/TRLabel.mat'
        LabelPath2 = 'data/Houston/TSLabel.mat'
        Data1 = loadmat(DataPath1)['HSI']
        Data2 = loadmat(DataPath2)['LiDAR']
    elif args.dataset == 'Augsburg':
        num_classes = 7
        DataPath1 = 'Data/Augsburg/data_HS_LR.mat'
        DataPath2 = 'Data/Augsburg/data_DSM.mat'
        Data1 = loadmat(DataPath1)['data_HS_LR']
        Data2 = loadmat(DataPath2)['data_DSM']
        LabelPath1 = 'data/Augsburg/train_test_gt.mat'
        LabelPath2 = 'Augsburg/train_test_gt.mat'
        # LabelPath = 'Data/Augsburg/train_test_gt.mat'
    elif args.dataset == 'Muufl':
        num_classes = 11
        DataPath1 = 'data/Muufl/HSI.mat'
        DataPath2 = 'data/Muufl/LiDAR2.mat'
        LabelPath1 = 'Ddata/Muufl/tr_ts_gt_150samples.mat'
        LabelPath2 = 'data/Muufl/tr_ts_gt_150samples.mat'
        Data1 = loadmat(DataPath1)['HSI']
        Data2 = loadmat(DataPath2)['LiDAR']
        # LabelPath = 'D:\PYcharm\Project\FDNet\data\Muufl\All_Label.mat'
    elif args.dataset == 'Trento':
        num_classes = 6
        DataPath1 = 'data/Trento/HSI.mat'
        DataPath2 = 'data/Trento/LiDAR.mat'
        LabelPath1 ='data/Trento/TRLabel.mat'
        LabelPath2 = 'data/Trento/TSLabel.mat'
        Data1 = loadmat(DataPath1)['HSI']
        Data2 = loadmat(DataPath2)['LiDAR']
    else:
        raise "Requires correct dataset name!"

    Data1 = Data1.astype(np.float32)  # hsi
    Data2 = Data2.astype(np.float32)  # lidar
    TrLabel = loadmat(LabelPath1)['train_data'] #TRLabel   TSLabel
    TsLabel = loadmat(LabelPath2)['test_data']  #train_data   test_data

    patchsize = args.patch_size
    pad_width = np.floor(patchsize / 2)
    pad_width = int(pad_width)
    TrainPatch1, TrainPatch2, TrainLabel = trPixel2Patch(
        Data1, Data2, patchsize, pad_width, TrLabel)
    TestPatch1, TestPatch2, TestLabel, _, _ = tsPixel2Patch(
        Data1, Data2, patchsize, pad_width, TsLabel)

    train_dataset = Data.TensorDataset(
        TrainPatch1, TrainPatch2, TrainLabel)
    test_dataset = Data.TensorDataset(
        TestPatch1, TestPatch2, TestLabel)
    train_loader = Data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = Data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=True)

    [H1, W1, _] = np.shape(Data1)
    Data2 = Data2.reshape([H1, W1, -1])
    height1, width1, band1 = Data1.shape
    height2, width2, band2 = Data2.shape
    # data size
    logger.info('\n')
    logger.info("=" * 50)
    logger.info("=" * 50)
    logger.info("hsi_height={0},hsi_width={1},hsi_band={2}".format(height1, width1, band1))
    logger.info("lidar_height={0},lidar_width={1},lidar_band={2}".format(height2, width2, band2))
    # -------------------------------------------------------------------------------
    # create model
    model = CMFDNET(l1=band1, l2=band2, patch_size=args.patch_size, num_classes=num_classes,
                            wavename=args.wavename, attn_kernel_size=args.attn_kernel_size,
                            dim_head=64, heads=8, embed_dim=128, num_heads=8)

    model = model.cuda()
    # criterion = nn.CrossEntropyLoss().cuda()
    criterion = LabelSmoothingCrossEntropyLoss(smoothing=0.1).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    # -------------------------------------------------------------------------------
    # train & test
    if args.flag == 'train':
        get_ts_result = False
        logger.info("start training")
        tic = time.time()
        for epoch in range(args.epochs):
            # train model
            model.train()
            train_acc, train_obj, tar_t, pre_t = train_epoch(
                model, train_loader, criterion, optimizer)
            OA1, AA1, Kappa1, CA1 = output_metric(tar_t, pre_t)
            logger.info("Epoch: {:03d} | train_loss: {:.4f} | train_OA: {:.4f} | train_AA: {:.4f} | train_Kappa: {:.4f}"
                        .format(epoch + 1, train_obj, OA1, AA1, Kappa1))
            scheduler.step()

        torch.save(model.state_dict(), 'cls_param/{}/CMFDNet_{}.pkl'.format(args.dataset, args.dataset))
        toc = time.time()
        model.eval()
        model.load_state_dict(torch.load('cls_param/{}/CMFDNet_{}.pkl'.format(args.dataset, args.dataset)))
        tar_v, pre_v = valid_epoch(model, test_loader, criterion, get_ts_result)
        OA, AA, Kappa, CA = output_metric(tar_v, pre_v)
        logger.info("Final records:")
        logger.info("Maximal Accuracy: %f" % OA)
        logger.info("OA: {:.4f} | AA: {:.4f} | Kappa: {:.4f}".format(OA, AA, Kappa))
        logger.info(CA)
        logger.info("Running Time: {:.2f}".format(toc - tic))

    if args.flag == 'test':
        # test best model
        get_ts_result = False
        tic_ts = time.time()
        model.eval()
        model.load_state_dict(torch.load('cls_param/{}/CMFDNet_{}.pkl'.format(args.dataset, args.dataset)))

        tar_v, pre_v = valid_epoch(model, test_loader, criterion, get_ts_result)
        OA, AA, Kappa, CA = output_metric(tar_v, pre_v)
        toc_ts = time.time()
        logger.info("Test records:")
        logger.info("Maximal Accuracy: %f" % OA)
        logger.info("OA: {:.4f} | AA: {:.4f} | Kappa: {:.4f}".format(OA, AA, Kappa))
        logger.info(CA)
        logger.info("Testing Time: {:.2f}".format(toc_ts - tic_ts))
        logger.info("Parameter:")
        logger.info(vars(args))

        # draw map
        if args.dataset == 'Houston':
            TR_TS_Path = 'data/Houston/tr_ts.mat'
        elif args.dataset == 'Augsburg':
            TR_TS_Path = 'data/Augsburg/train_test_gt.mat'
        elif args.dataset == 'Muufl':
            TR_TS_Path = 'data/Muufl/All_Label.mat'
        elif args.dataset == 'Trento':
            TR_TS_Path = 'Data/Trento/tr_ts.mat'
        else:
            raise "Correct dataset needed!"
#-----------------------------------
        # train_label = loadmat(TR_TS_Path)['train_data']
        # test_label = loadmat(TR_TS_Path)['test_data']
        #
        # # 元素相加
        # TR_TS_Label = train_label + test_label

        TR_TS_Label = loadmat(TR_TS_Path)['All_Label']

        # draw gt map
        draw_classification_map(TR_TS_Label, 'cls_map/{}/{}_groundTruth.png'.format(args.dataset, args.dataset),
                                args.dataset)
        # draw cls map
        TR_TS_Patch1, TR_TS_Patch2, TR_TS_Label, xIndex_list, yIndex_list = tsPixel2Patch(
            Data1, Data2, patchsize, pad_width, TR_TS_Label)
        TR_TS_dataset = Data.TensorDataset(
            TR_TS_Patch1, TR_TS_Patch2, TR_TS_Label)
        best_test_loader = Data.DataLoader(
            TR_TS_dataset, batch_size=args.batch_size, shuffle=False)

        get_ts_result = True  # if True, return cls result
        ts_result = valid_epoch(model, best_test_loader, criterion, get_ts_result)
        ts_result_matrix = np.full((H1, W1), 0)
        for i in range(len(ts_result)):
            ts_result_matrix[xIndex_list[i], yIndex_list[i]] = ts_result[i]
        draw_classification_map(ts_result_matrix, 'cls_map/{}/{}_predLabel.png'.format(args.dataset, args.dataset),
                                args.dataset)


if __name__ == '__main__':
    set_seed(args.seed)
    train1time()
