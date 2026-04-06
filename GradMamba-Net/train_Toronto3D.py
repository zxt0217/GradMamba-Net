import argparse
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

import torch
import datetime
import logging
from pathlib import Path
import sys
import importlib
import shutil
from tqdm import tqdm
import numpy as np
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

from data_utils.Toronto3DBlockDataLoader import Toronto3DDataset
from utils import provider
from utils.error_matrix import ConfusionMatrix

classes = ['Ground', 'Road_markings', 'Natural', 'Building', 'Utility_line', 'Pole', 'Car', 'Fence']
class2label = {cls: i for i, cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {i: cat for i, cat in enumerate(seg_classes.keys())}


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True


def worker_init_fn(worker_id):
    np.random.seed(worker_id + int(time.time()))


def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='diffconv_umamba', help='model name')
    parser.add_argument('--data_root', type=str, default='./data/Toronto3D_blocks',
                        help='path to preprocessed Toronto3D blocks')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch Size [default: 8]')
    parser.add_argument('--epoch', default=150, type=int, help='Epoch to run [default: 150]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD')
    parser.add_argument('--log_dir', type=str, default=None, help='Log path; default uses timestamp to avoid resuming old checkpoints')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--npoint', type=int, default=16384, help='Point Number')
    parser.add_argument('--step_size', type=int, default=5, help='Decay step for lr decay')
    parser.add_argument('--lr_decay', type=float, default=0.8, help='Decay rate for lr decay')
    parser.add_argument('--num_category', type=int, default=8, help='num_category')
    parser.add_argument('--weighted_loss', type=bool, default=True, help='weighted loss')  # 推荐开启以对齐论文
    parser.add_argument('--fence_weight_factor', type=float, default=1.6, help='extra loss weight factor for Fence class')
    parser.add_argument('--fence_sample_boost', type=float, default=1.3,
                        help='sampling weight boost for blocks containing Fence points')
    parser.add_argument('--scan_directions', type=int, default=4, help='number of scan directions used by series indices [1-8]')
    parser.add_argument('--fence_focal_gamma', type=float, default=2.0, help='focal gamma for Fence class term')
    parser.add_argument('--fence_focal_weight', type=float, default=0.15, help='relative weight of Fence focal term')
    parser.add_argument('--warmup_epoch', type=int, default=8, help='linear warmup epochs')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='max grad norm, <=0 disables clipping')
    parser.add_argument('--bn_momentum_init', type=float, default=0.1, help='initial BN momentum')
    parser.add_argument('--bn_decay', type=float, default=0.8, help='BN momentum decay factor')
    parser.add_argument('--bn_min_momentum', type=float, default=0.02, help='minimum BN momentum')
    parser.add_argument('--num_workers', type=int, default=4, help='dataloader workers')
    return parser.parse_args()


def main(args):
    def log_string(s):
        logger.info(s)
        print(s)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
    torch.set_float32_matmul_precision('high')

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('toronto3d_seg')
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)
    log_string(f'Device: {device}')

    root = os.path.abspath(args.data_root)
    log_string(f'Data root: {root}')

    NUM_CLASSES = 8
    NUM_POINT = args.npoint
    BATCH_SIZE = args.batch_size
    fps_n_list = [512, 128, 32]

    log_string("start loading training data ...")
    TRAIN_DATASET = Toronto3DDataset(split='train', data_root=root, fps_n_list=fps_n_list, label_number=NUM_CLASSES,
                                     npoints=NUM_POINT, fence_sample_boost=args.fence_sample_boost,
                                     scan_directions=args.scan_directions)
    log_string("start loading test data ...")
    TEST_DATASET = Toronto3DDataset(split='test', data_root=root, fps_n_list=fps_n_list, label_number=NUM_CLASSES,
                                    npoints=NUM_POINT, fence_sample_boost=1.0,
                                    scan_directions=args.scan_directions)

    train_sampler = None
    train_shuffle = True
    if args.fence_sample_boost > 1.0 and len(TRAIN_DATASET.sample_weights) > 0:
        train_sampler = torch.utils.data.WeightedRandomSampler(
            weights=torch.as_tensor(TRAIN_DATASET.sample_weights, dtype=torch.double),
            num_samples=len(TRAIN_DATASET.sample_weights),
            replacement=True
        )
        train_shuffle = False
        log_string(f"Use fence-aware sampler with boost x{args.fence_sample_boost:.2f}.")
        log_string(f"Fence blocks in train set: {TRAIN_DATASET.fence_block_count}/{len(TRAIN_DATASET)}")

    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=train_shuffle,
                                                  sampler=train_sampler, num_workers=args.num_workers,
                                                  pin_memory=True, drop_last=False,
                                                  worker_init_fn=worker_init_fn)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=BATCH_SIZE, shuffle=False,
                                                 num_workers=args.num_workers,
                                                 pin_memory=True, drop_last=False)

    log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    log_string("The number of test data is: %d" % len(TEST_DATASET))

    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)
    shutil.copy('models/%s.py' % args.model, os.path.join(str(experiment_dir), f"{args.model}.py"))

    classifier = MODEL.get_model(NUM_CLASSES, fps_n_list).to(device)

    criterion = MODEL.get_loss(ignore_index=-1).to(device)
    weight = torch.tensor(TRAIN_DATASET.labelweights, device=device)
    if weight.numel() > 7 and args.fence_weight_factor > 0:
        weight[7] = weight[7] * args.fence_weight_factor
        log_string(f"Apply Fence class loss weight boost x{args.fence_weight_factor:.2f}.")

    if args.weighted_loss:
        log_string("Use weighted loss ...")
        criterion = MODEL.get_loss_weighted(
            ignore_index=-1,
            fence_class_idx=7,
            fence_focal_gamma=args.fence_focal_gamma,
            fence_focal_weight=args.fence_focal_weight
        ).to(device)

    classifier.apply(inplace_relu)

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)

    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except Exception as e:
        log_string(f'Checkpoint load skipped ({type(e).__name__}), starting training from scratch...')
        start_epoch = 0
        classifier = classifier.apply(weights_init)

    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.learning_rate, weight_decay=args.decay_rate)

    def get_lr_scale(cur_epoch):
        warmup = max(0, int(args.warmup_epoch))
        if warmup > 0 and cur_epoch < warmup:
            return float(cur_epoch + 1) / float(warmup)

        if args.epoch <= warmup:
            return 1.0

        progress = (cur_epoch - warmup) / float(max(1, args.epoch - warmup))
        progress = max(0.0, min(1.0, progress))
        min_lr_factor = 0.01
        cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
        return min_lr_factor + (1.0 - min_lr_factor) * cosine

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    best_iou = 0

    print('Start Training...')
    for epoch in range(start_epoch, args.epoch):
        log_string('**** Epoch %d (%d/%s) ****' % (epoch + 1, epoch + 1, args.epoch))

        lr_scale = get_lr_scale(epoch)
        current_lr = args.learning_rate * lr_scale
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        momentum = max(
            args.bn_momentum_init * (args.bn_decay ** (epoch // args.step_size)),
            args.bn_min_momentum
        )
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))
        log_string(f'lr={current_lr:.6e}, bn_momentum={momentum:.4f}')

        num_batches = len(trainDataLoader)
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        classifier = classifier.train()

        for i, (points, target, fps_index_array, series_idx_arrays) in tqdm(enumerate(trainDataLoader),
                                                                            total=len(trainDataLoader), smoothing=0.9):
            if i % 10 == 0:
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

            optimizer.zero_grad()
            points = points.data.numpy()
            points[:, :, :3] = provider.rotate_point_cloud_z(points[:, :, :3])
            points[:, :, :3] = provider.random_scale_point_cloud(points[:, :, :3])
            points = torch.Tensor(points).float().to(device)
            target = target.long().to(device)
            fps_index_array = fps_index_array.long().to(device)
            series_idx_arrays = series_idx_arrays.long().to(device)
            points = points.transpose(2, 1)

            pre = classifier(points, fps_index_array, series_idx_arrays)
            pre = pre.contiguous().view(-1, NUM_CLASSES)
            target = target.view(-1)

            loss = criterion(pre, target, weight)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=args.grad_clip)
            optimizer.step()

            # 修改 3：计算训练准确率时排除 -1 标签的点
            pred_choice = pre.cpu().data.max(1)[1].numpy()
            target_np = target.cpu().data.numpy()
            valid_mask = (target_np != -1)
            total_correct += np.sum(pred_choice[valid_mask] == target_np[valid_mask])
            total_seen += np.sum(valid_mask)
            loss_sum += loss.item()

        log_string('Training mean loss: %f' % (loss_sum / num_batches))
        log_string('Training accuracy: %f' % (total_correct / float(total_seen + 1e-6)))

        '''Evaluation'''
        with torch.no_grad():
            classifier = classifier.eval()
            confusion = ConfusionMatrix(num_classes=NUM_CLASSES, labels=classes)
            total_correct_class = np.zeros(NUM_CLASSES)
            total_iou_deno_class = np.zeros(NUM_CLASSES)

            for i, (points, target, fps_index_array, series_idx_arrays) in tqdm(enumerate(testDataLoader),
                                                                                total=len(testDataLoader)):
                points = torch.Tensor(points).float().to(device).transpose(2, 1)
                target = target.long().to(device)
                fps_index_array = fps_index_array.long().to(device)
                series_idx_arrays = series_idx_arrays.long().to(device)

                pre = classifier(points, fps_index_array, series_idx_arrays)
                pred_val = pre.contiguous().cpu().data.numpy()
                batch_label = target.cpu().data.numpy()

                pred_val = np.argmax(pred_val, 2)

                valid_mask = (batch_label != -1)
                confusion.update(pred_val[valid_mask], batch_label[valid_mask])

                for l in range(NUM_CLASSES):
                    total_correct_class[l] += np.sum((pred_val == l) & (batch_label == l))
                    total_iou_deno_class[l] += np.sum(((pred_val == l) | (batch_label == l)) & valid_mask)

            ave_F1_score, miou, acc = confusion.summary()
            mIoU = np.mean(total_correct_class / (total_iou_deno_class + 1e-6))
            log_string('eval point avg class IoU: %f' % (mIoU))

            if mIoU >= best_iou:
                best_iou = mIoU
                savepath = str(checkpoints_dir) + '/best_model.pth'
                torch.save({'epoch': epoch, 'class_avg_iou': mIoU, 'model_state_dict': classifier.state_dict()},
                           savepath)
            log_string('Best mIoU: %f' % best_iou)
if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    args = parse_args()
    main(args)
