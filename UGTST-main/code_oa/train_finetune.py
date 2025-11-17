import argparse
import logging
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloaders.dataset import TwoStreamBatchSampler
from dataloaders.ppremo_dataset import create_target_datasets
from networks.unet import UNet
from utils import losses, metrics, ramps
from val_2D import test_single_volume

parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str,
                    default='NPC_WCH/WCH_self_training_UGTST+_5%', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='UNet', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=20000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.001,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=int, nargs=2, default=[128, 128],
                    help='patch size of 2D network input (X, Y)')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--num_classes', type=int,  default=87,
                    help='output channel of network')
# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=12,
                    help='labeled_batch_size per gpu')
# costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')
parser.add_argument('--early_stop_patient', type=float,  default=5000,
                    help='num for early stop patient')
parser.add_argument('--pretrained_path', type=str,
                    default='../model/NPC/source_train/UNet_best_model.pth', help='Path to the pretrained model')
parser.add_argument('--labeled_num', type=int, default=56, help='labeled slices')
parser.add_argument('--active_method', type=str,
                    default='UGTST', help='active learning method')
parser.add_argument('--split_json', type=str, required=True,
                    help='Split JSON describing training/validation data')
parser.add_argument('--roi_x', type=int, default=128, help='ROI size along X axis')
parser.add_argument('--roi_y', type=int, default=128, help='ROI size along Y axis')
parser.add_argument('--roi_z', type=int, default=128, help='ROI size along Z axis (slice depth)')
parser.add_argument('--target_spacing', type=float, nargs=3, default=[0.8, 0.8, 0.8],
                    help='Target voxel spacing for resampling (x y z)')
parser.add_argument('--apply_spacing', dest='apply_spacing', action='store_true', default=True)
parser.add_argument('--no_apply_spacing', dest='apply_spacing', action='store_false')
parser.add_argument('--apply_orientation', dest='apply_orientation', action='store_true', default=True)
parser.add_argument('--no_apply_orientation', dest='apply_orientation', action='store_false')
parser.add_argument('--foreground_only', action='store_true', default=True,
                    help='Enable Draw-EM foreground-only label remapping')
parser.add_argument('--laterality_pairs_json', type=str, default=None,
                    help='JSON file containing LR swapping label pairs')
parser.add_argument('--val_subset', type=str, default='validation',
                    help='Key used for validation subset inside the split JSON')
parser.add_argument('--slices_per_volume', type=int, default=None,
                    help='Override number of axial slices per subject (defaults to roi_z)')
args = parser.parse_args()

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def worker_init_fn(worker_id):
    random.seed(args.seed + worker_id)

def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    early_stop_patient = args.early_stop_patient
    def create_model(ema=False):
        # Network definition
        model = UNet(in_chns=1,
                            class_num=num_classes)
        if args.pretrained_path is not None:
            model.load_state_dict(torch.load(args.pretrained_path))
            logging.info(f"Loaded pretrained model from {args.pretrained_path}")
        model = model.cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model()

    roi_size = (args.roi_x, args.roi_y, args.roi_z)
    spacing = tuple(float(s) for s in args.target_spacing)
    train_dataset, val_dataset = create_target_datasets(
        args.split_json,
        roi_size=roi_size,
        target_spacing=spacing,
        apply_spacing=args.apply_spacing,
        apply_orientation=args.apply_orientation,
        foreground_only=args.foreground_only,
        laterality_pairs_json=args.laterality_pairs_json,
        slices_per_volume=args.slices_per_volume,
        val_subset=args.val_subset,
    )

    total_slices = len(train_dataset)
    if total_slices <= 1:
        raise RuntimeError("Target dataset must contain at least two training slices for semi-supervised sampling")
    labeled_slice = min(args.labeled_num, total_slices - 1)
    labeled_slice = max(labeled_slice, 1)
    print("Total silices is: {}, labeled slices is: {}".format(
        total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size-args.labeled_bs)

    trainloader = DataLoader(train_dataset, batch_sampler=batch_sampler,
                             num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)

    model.train()

    valloader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                           num_workers=0)

    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)

    ce_loss = CrossEntropyLoss(ignore_index=-1)
    dice_loss = losses.DiceLoss(num_classes)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    no_improvement_counter = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch = volume_batch.cuda()
            label_batch = label_batch.cuda()

            output, _ = model(volume_batch)
            outputs_soft = torch.softmax(output, dim=1)

            consistency_weight = get_current_consistency_weight(iter_num // 150)

            labeled_gt = label_batch[:args.labeled_bs]
            labeled_gt_dice = labeled_gt.clone()
            labeled_gt_dice[labeled_gt_dice < 0] = 0
            loss_lab = 0.5 * (ce_loss(output[:args.labeled_bs], labeled_gt.long()) + dice_loss(
                outputs_soft[:args.labeled_bs], labeled_gt_dice.unsqueeze(1)))
            pseudo_outputs = torch.argmax(outputs_soft[args.labeled_bs:].detach(), dim=1, keepdim=False)
            pseudo_supervision = ce_loss(output[args.labeled_bs:], pseudo_outputs)

            # loss = loss_lab
            loss = loss_lab + pseudo_supervision
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            iter_num = iter_num + 1
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar(
                'consistency_weight/consistency_weight', consistency_weight, iter_num)
            writer.add_scalar('loss',
                              loss, iter_num)

            logging.info('iteration %d : lab_loss : %f ulab_loss : %f' % (iter_num, loss_lab.item(), pseudo_supervision.item()))
            if iter_num % 200 == 0:
                image = volume_batch[1, 0:1, :, :]
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(
                    output, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction',
                                 outputs[1, ...] * 50, iter_num)

            if iter_num > 0 and iter_num % 50 == 0:
                model.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume(
                        sampled_batch["image"], sampled_batch["label"], model,
                        classes=num_classes, patch_size=args.patch_size)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(val_dataset)
                for class_i in range(num_classes):
                    writer.add_scalar('info/model1_val_{}_dice'.format(class_i),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/model1_val_{}_hd95'.format(class_i),
                                      metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]

                mean_hd95 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/model1_val_mean_dice', performance, iter_num)
                writer.add_scalar('info/model1_val_mean_hd95', mean_hd95, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path,
                                                  'model1_iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)
                    no_improvement_counter = 0
                else:
                    no_improvement_counter += 50
                logging.info(
                    'iteration %d : model1_mean_dice : %f model1_mean_hd95 : %f' % (iter_num, performance, mean_hd95))
                model.train()


            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                time1 = time.time()
                break
        if no_improvement_counter >= early_stop_patient:
            logging.info('No improvement in Validation mean_dice for {} iterations. Early stopping...'.format(early_stop_patient))
            iterator.close()
            break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    try:
        complete_flag = Path(snapshot_path) / "training_complete.txt"
        complete_flag.write_text("done\n", encoding='utf-8')
    except Exception as exc:
        logging.warning(f"Failed to write completion flag: {exc}")

if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "../model/{}".format(
        args.exp)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
