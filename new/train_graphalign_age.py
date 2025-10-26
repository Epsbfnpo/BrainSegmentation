#!/usr/bin/env python3
"""
Main training script for DAUnet - WITH DUAL-BRANCH CROSS-DOMAIN GRAPH ALIGNMENT
Enhanced with dynamic spectral alignment and mismatch-aware penalties
WITH CHECKPOINT RESUME AND SIGNAL HANDLING FOR 2-HOUR CHUNKS
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau
from tensorboardX import SummaryWriter
from monai.networks.nets import SwinUNETR
from monai.utils import set_determinism
import traceback
import gc
import json
from datetime import datetime, timedelta
import numpy as np
import copy
from collections import deque
import glob
import signal
import subprocess
import time
import shutil
import traceback
from datetime import datetime

# Import simplified DAUnet components
from age_aware_modules import SimplifiedDAUnetModule
from trainer_age_aware import train_epoch_age_aware, val_epoch_age_aware, save_checkpoint_simplified
from data_loader_age_aware import get_source_target_dataloaders
from dice_monitor import DiceMonitor
from graph_prior_loss import AgeConditionedGraphPriorLoss


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, os.pardir))
_DEFAULT_SOURCE_PRIOR = os.path.join(_REPO_ROOT, "dHCP_class_prior_foreground.json")
_DEFAULT_TARGET_PRIOR = os.path.join(_REPO_ROOT, "PPREMOPREBO_class_prior_foreground.json")


def is_dist():
    """Check if distributed training is enabled"""
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def get_parser():
    """Get argument parser with production settings"""
    parser = argparse.ArgumentParser(description='DAUnet Training - Dual-Branch Cross-Domain Graph Alignment')

    # Basic training parameters
    parser.add_argument('--epochs', default=400, type=int,
                        help='Number of total epochs to run')
    parser.add_argument('--batch_size', default=4, type=int,
                        help='Mini-batch size per GPU')
    parser.add_argument('--lr', default=5e-5, type=float,
                        help='Initial learning rate')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='Weight decay')

    # Model parameters
    parser.add_argument('--in_channels', default=1, type=int,
                        help='Number of input channels')
    parser.add_argument('--out_channels', default=88, type=int,
                        help='Number of segmentation classes')
    parser.add_argument('--feature_size', default=48, type=int,
                        help='Feature size for transformer')
    parser.add_argument('--roi_x', default=96, type=int,
                        help='ROI size in x direction')
    parser.add_argument('--roi_y', default=96, type=int,
                        help='ROI size in y direction')
    parser.add_argument('--roi_z', default=96, type=int,
                        help='ROI size in z direction')

    # Model regularization
    parser.add_argument('--model_dropout', default=0.0, type=float,
                        help='Dropout rate for the model')

    # Foreground-only training
    parser.add_argument('--foreground_only', action='store_true',
                        help='Focus on foreground classes, ignoring background')

    # Data parameters
    parser.add_argument('--source_split_json', type=str, required=True,
                        help='Path to source domain data split JSON file')
    parser.add_argument('--split_json', type=str, required=True,
                        help='Path to target domain data split JSON file')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of data loading workers per GPU')
    parser.add_argument('--cache_rate', default=0.5, type=float,
                        help='Cache rate for data loading')
    parser.add_argument('--cache_num_workers', default=4, type=int,
                        help='Number of workers for cache dataset')

    # Explicit registration control
    parser.add_argument('--source_is_registered', action='store_true', default=False,
                        help='Source split is rigidly registered to a common space')
    parser.add_argument('--target_is_registered', action='store_true', default=False,
                        help='Target split is rigidly registered to a common space')

    # Spacing and orientation control
    parser.add_argument('--target_spacing', nargs=3, type=float, default=[0.8, 0.8, 0.8],
                        help='Target voxel spacing used in train/val/test')
    parser.add_argument('--apply_spacing', action='store_true', default=True,
                        help='Apply Spacingd to both image/label before crop')
    parser.add_argument('--apply_orientation', action='store_true', default=True,
                        help="Apply Orientationd('RAS') before crop")

    # Laterality pairs for left-right label swapping
    parser.add_argument('--laterality_pairs_json', type=str, default=None,
                        help='JSON path of left-right label pairs (original label IDs, 1..87)')

    # Class prior files
    parser.add_argument('--source_prior_json', type=str, default=None,
                        help='Path to source domain class prior JSON')
    parser.add_argument('--target_prior_json', type=str, default=None,
                        help='Path to target domain class prior JSON')
    parser.add_argument('--enhanced_class_weights', action='store_true',
                        help='Use enhanced class weighting strategy')

    # ========== TARGET DOMAIN GRAPH PRIORS (existing) ==========
    parser.add_argument('--prior_adj_npy', type=str, default=None,
                        help='Path to target domain prior adjacency matrix (.npy file)')
    parser.add_argument('--prior_required_json', type=str, default=None,
                        help='Path to target domain required edges JSON file')
    parser.add_argument('--prior_forbidden_json', type=str, default=None,
                        help='Path to target domain forbidden edges JSON file')
    parser.add_argument('--prior_containment_json', type=str, default=None,
                        help='Path to containment relations JSON file')
    parser.add_argument('--prior_exclusive_json', type=str, default=None,
                        help='Path to exclusive pairs JSON file')

    # ========== AGE-AWARE PRIORS (NEW, minimal-intrusion) ==========
    parser.add_argument('--use_age_conditioning', action='store_true', default=True,
                        help='Enable age conditioning in model and priors')
    parser.add_argument('--age_embed_dim', type=int, default=64,
                        help='Embedding dim for age FiLM-style modulation')
    parser.add_argument('--volume_stats_json', type=str, default=None,
                        help='JSON: age -> {"means":[C], "stds":[C]} (volume fractions)')
    parser.add_argument('--shape_templates_pt', type=str, default=None,
                        help='PT/PTH: dict[class] -> SDT template (optionally age-indexed)')
    parser.add_argument('--weighted_adj_npy', type=str, default=None,
                        help='Weighted adjacency prior (.npy), contact strengths')
    parser.add_argument('--age_weights_json', type=str, default=None,
                        help='JSON: age -> adjacency weight matrix or interpolation spec')
    parser.add_argument('--lambda_volume', type=float, default=0.2,
                        help='Weight for age-aware volume prior loss')
    parser.add_argument('--lambda_shape', type=float, default=0.1,
                        help='Weight for age-aware shape/SDT prior loss')
    parser.add_argument('--lambda_weighted_adj', type=float, default=0.15,
                        help='Weight for age-aware weighted adjacency regression')
    parser.add_argument('--lambda_topo', type=float, default=0.02,
                        help='Weight for weak binary topology prior (required/forbidden)')
    parser.add_argument('--prior_warmup_epochs', type=int, default=10,
                        help='Warmup epochs for age-aware priors')
    parser.add_argument('--prior_temperature', type=float, default=1.0,
                        help='Temperature/scaling for age-aware priors')

    # ========== SOURCE DOMAIN GRAPH PRIORS ==========
    parser.add_argument('--src_prior_adj_npy', type=str, default=None,
                        help='Path to SOURCE domain prior adjacency matrix (.npy file)')
    parser.add_argument('--src_prior_required_json', type=str, default=None,
                        help='Path to SOURCE domain required edges JSON file')
    parser.add_argument('--src_prior_forbidden_json', type=str, default=None,
                        help='Path to SOURCE domain forbidden edges JSON file')

    # Debugging and diagnostics
    parser.add_argument('--debug_mode', action='store_true', default=False,
                        help='Enable verbose debug logging for the first few batches')
    parser.add_argument('--debug_step_limit', type=int, default=2,
                        help='Number of training/validation steps per epoch to print debug info')
    parser.add_argument('--debug_graph_batches', type=int, default=2,
                        help='Number of batches to log inside the age-conditioned graph prior module')
    parser.add_argument('--debug_validate_samples', type=int, default=2,
                        help='Validation batches to dump detailed diagnostics when debug is enabled')

    # Graph loss hyperparameters
    parser.add_argument('--graph_topr', type=int, default=20,
                        help='Number of eigenvalues for spectral alignment')
    parser.add_argument('--lambda_spec', type=float, default=0.1,
                        help='Weight for spectral alignment loss (legacy, split into src/tgt)')
    parser.add_argument('--lambda_edge', type=float, default=0.1,
                        help='Weight for edge consistency loss (legacy, split into src/tgt)')
    parser.add_argument('--lambda_sym', type=float, default=0.05,
                        help='Weight for symmetry consistency loss')
    parser.add_argument('--lambda_struct', type=float, default=0.05,
                        help='Weight for structural constraint loss')

    # ========== CROSS-DOMAIN ALIGNMENT WEIGHTS ==========
    parser.add_argument('--lambda_spec_src', type=float, default=None,
                        help='Weight for SOURCE domain spectral alignment (default: lambda_spec)')
    parser.add_argument('--lambda_edge_src', type=float, default=None,
                        help='Weight for SOURCE domain edge consistency (default: lambda_edge)')
    parser.add_argument('--lambda_spec_tgt', type=float, default=None,
                        help='Weight for TARGET domain spectral alignment (default: 0.3*lambda_spec)')
    parser.add_argument('--lambda_edge_tgt', type=float, default=None,
                        help='Weight for TARGET domain edge consistency (default: 0.3*lambda_edge)')
    parser.add_argument('--graph_align_mode', type=str, default='joint',
                        choices=['src_only', 'tgt_only', 'joint'],
                        help='Graph alignment mode: align to source only, target only, or both')

    # ========== NEW: DYNAMIC SPECTRAL ALIGNMENT PARAMETERS ==========
    parser.add_argument('--lambda_dyn', type=float, default=0.2,
                        help='Weight for dynamic spectral alignment (relative to lambda_spec_src)')
    parser.add_argument('--dyn_top_k', type=int, default=12,
                        help='Number of low-frequency eigenvalues for dynamic alignment')
    parser.add_argument('--dyn_start_epoch', type=int, default=50,
                        help='Epoch to start dynamic spectral alignment')
    parser.add_argument('--dyn_ramp_epochs', type=int, default=50,
                        help='Number of epochs to ramp up dynamic weight')
    parser.add_argument('--align_U_weighted', action='store_true', default=True,
                        help='Use eigenvalue-weighted U subspace alignment')
    parser.add_argument('--qap_mismatch_g', type=float, default=1.5,
                        help='Mismatch penalty factor for QAP-like loss')
    parser.add_argument('--use_restricted_mask', action='store_true', default=True,
                        help='Use restricted alignment mask for forbidden/required/symmetry')
    parser.add_argument('--restricted_mask_path', type=str, default=None,
                        help='Path to precomputed restricted mask R_mask.npy')

    # Graph loss scheduling
    parser.add_argument('--graph_warmup_epochs', type=int, default=10,
                        help='Warmup epochs for graph loss')
    parser.add_argument('--graph_temperature', type=float, default=1.0,
                        help='Temperature for adjacency computation')
    parser.add_argument('--graph_pool_kernel', type=int, default=3,
                        help='Pooling kernel size for adjacency computation')
    parser.add_argument('--graph_pool_stride', type=int, default=2,
                        help='Pooling stride for adjacency computation')
    # =================================================

    # Weighted sampling for small classes
    parser.add_argument('--num_small_classes_boost', default=20, type=int,
                        help='Number of smallest classes to boost in sampling')
    parser.add_argument('--small_class_boost_factor', default=2.0, type=float,
                        help='Boost factor for small classes in sampling')

    # Pretrained model
    parser.add_argument('--pretrained_model', type=str, required=True,
                        help='Path to pretrained model')
    parser.add_argument('--resume_training', action='store_true',
                        help='Resume training from checkpoint')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume from')

    # Training settings
    parser.add_argument('--clip', default=1.0, type=float,
                        help='Gradient clipping')
    parser.add_argument('--val_split', default=0.2, type=float,
                        help='Validation split ratio')
    parser.add_argument('--loss_config', default='dice_focal',
                        choices=['dice_ce', 'dice_focal', 'dice_ce_focal'],
                        help='Loss function configuration')

    # Focal loss parameters
    parser.add_argument('--focal_gamma', default=2.0, type=float,
                        help='Gamma parameter for focal loss')

    # Top-K Dice Loss
    parser.add_argument('--use_topk_dice', action='store_true',
                        help='Use Top-K hardest voxels for Dice loss')
    parser.add_argument('--topk_ratio', default=0.3, type=float,
                        help='Ratio of hardest voxels to use')
    parser.add_argument('--topk_warmup_epochs', default=30, type=int,
                        help='Number of epochs for Top-K warm-up')

    # LR scheduler
    parser.add_argument('--lr_scheduler', default='cosine_warmup_restart',
                        choices=['step', 'cosine_smooth', 'plateau', 'cosine_warmup_restart'],
                        help='Learning rate scheduler type')
    parser.add_argument('--lr_warmup_epochs', default=10, type=int,
                        help='Number of warmup epochs')
    parser.add_argument('--lr_restart_epochs', nargs='+', type=int,
                        default=[100, 200, 300],
                        help='Epochs to restart learning rate')
    parser.add_argument('--lr_min', default=1e-7, type=float,
                        help='Minimum learning rate')

    # Checkpointing and logging
    parser.add_argument('--results_dir', default='./results/', type=str,
                        help='Directory for results')
    parser.add_argument('--save_interval', default=20, type=int,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--eval_num', default=2, type=int,
                        help='Evaluation frequency')
    parser.add_argument('--print_freq', default=10, type=int,
                        help='Print frequency')

    # Validation settings
    parser.add_argument('--sw_batch_size', default=4, type=int,
                        help='Sliding window batch size')
    parser.add_argument('--infer_overlap', default=0.7, type=float,
                        help='Sliding window overlap')
    parser.add_argument('--use_tta', action='store_true', default=True,
                        help='Use Test Time Augmentation')

    # Target label utilization
    parser.add_argument('--use_target_labels', action='store_true', default=True,
                        help='Use target domain labels during training')
    parser.add_argument('--target_label_start_epoch', default=1, type=int,
                        help='Epoch to start using target labels')
    parser.add_argument('--target_label_weight', default=0.5, type=float,
                        help='Weight for target domain segmentation loss')

    # Dice monitoring
    parser.add_argument('--dice_drop_threshold', default=0.3, type=float,
                        help='Threshold for detecting dice score drops')
    parser.add_argument('--dice_window_size', default=5, type=int,
                        help='Window size for moving average')

    # Early stopping
    parser.add_argument('--early_stopping', action='store_true',
                        help='Enable early stopping')
    parser.add_argument('--early_stopping_patience', default=50, type=int,
                        help='Patience for early stopping')

    # H100 optimizations
    parser.add_argument('--use_amp', action='store_true', default=True,
                        help='Use automatic mixed precision')
    parser.add_argument('--amp_dtype', default='bfloat16',
                        choices=['float16', 'bfloat16'],
                        help='AMP dtype (bfloat16 recommended for H100)')

    # Distributed training timeout
    parser.add_argument('--dist_timeout', default=120, type=int,
                        help='Timeout in minutes for distributed operations')
    parser.add_argument('--use_label_crop', action='store_true',
                        help='Use label-based cropping for registered data to ensure all classes are sampled')

    # Time management for preemptible jobs
    parser.add_argument('--job_time_limit', default=115, type=int,
                        help='Job time limit in minutes (default: 115 for 2-hour jobs with buffer)')
    parser.add_argument('--time_buffer_minutes', default=5, type=int,
                        help='Buffer time in minutes before job ends')

    return parser


def load_pretrained_model(model, checkpoint_path, device):
    """Load pretrained model with proper handling"""
    is_main = (not is_dist()) or dist.get_rank() == 0

    if is_main:
        print(f"🔥 Loading pretrained model from: {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Pretrained model not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    model_dict = model.state_dict()
    pretrained_dict = {}

    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]
        if k in model_dict:
            if v.shape == model_dict[k].shape:
                pretrained_dict[k] = v

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    if is_main:
        print(f"✔ Loaded {len(pretrained_dict)}/{len(model_dict)} layers from pretrained model")

    return model


def save_checkpoint_atomic(model, optimizer, epoch, best_acc, args, filepath,
                           dice_history=None, additional_info=None):
    """Save checkpoint atomically to prevent corruption"""
    # Save to temporary file first
    temp_filepath = filepath + '.tmp'

    try:
        save_checkpoint_simplified(
            model, optimizer, epoch, best_acc, args, temp_filepath,
            dice_history=dice_history,
            additional_info=additional_info
        )

        # Verify the temporary file can be loaded
        test_load = torch.load(temp_filepath, map_location='cpu', weights_only=False)
        del test_load

        # Atomic rename (on same filesystem, this is atomic)
        shutil.move(temp_filepath, filepath)
        return True

    except Exception as e:
        print(f"⚠️ Failed to save checkpoint: {e}")
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)
        return False


def load_checkpoint_with_fallback(checkpoint_paths, device):
    """Try to load checkpoint with fallback options"""
    for path in checkpoint_paths:
        if path and os.path.exists(path):
            try:
                checkpoint = torch.load(path, map_location=device, weights_only=False)
                print(f"✅ Successfully loaded checkpoint from: {path}")
                return checkpoint, path
            except Exception as e:
                print(f"⚠️ Failed to load {path}: {e}")
                continue
    return None, None


class TimeManager:
    """Manage training time and predict if operations can complete"""

    def __init__(self, job_time_limit_minutes, buffer_minutes=5):
        self.job_start_time = time.time()
        self.job_time_limit = job_time_limit_minutes * 60  # Convert to seconds
        self.buffer_time = buffer_minutes * 60  # Buffer before job ends
        self.train_times = deque(maxlen=10)  # Keep last 10 train times
        self.val_times = deque(maxlen=5)  # Keep last 5 validation times
        self.epoch_times = deque(maxlen=10)  # Keep last 10 full epoch times

    def record_train_time(self, duration):
        """Record a training iteration time"""
        self.train_times.append(duration)

    def record_val_time(self, duration):
        """Record a validation time"""
        self.val_times.append(duration)

    def record_epoch_time(self, duration):
        """Record a full epoch time"""
        self.epoch_times.append(duration)

    def get_remaining_time(self):
        """Get remaining time in seconds before job ends (with buffer)"""
        elapsed = time.time() - self.job_start_time
        remaining = self.job_time_limit - self.buffer_time - elapsed
        return max(0, remaining)

    def estimate_train_time(self):
        """Estimate time for one training iteration"""
        if not self.train_times:
            return 300  # Default 5 minutes if no data
        return np.mean(self.train_times) * 1.2  # Add 20% safety margin

    def estimate_val_time(self):
        """Estimate time for one validation"""
        if not self.val_times:
            return 600  # Default 10 minutes if no data
        return np.mean(self.val_times) * 1.2  # Add 20% safety margin

    def estimate_epoch_time(self):
        """Estimate time for one full epoch"""
        if self.epoch_times:
            return np.mean(self.epoch_times) * 1.2
        # Estimate from train and val times
        train_est = self.estimate_train_time()
        val_est = self.estimate_val_time() if len(self.val_times) > 0 else 0
        return (train_est + val_est) * 1.2  # Add safety margin

    def can_complete_epoch(self, will_validate=False):
        """Check if there's enough time to complete an epoch"""
        remaining = self.get_remaining_time()

        if will_validate:
            required = self.estimate_train_time() + self.estimate_val_time()
        else:
            required = self.estimate_train_time()

        return remaining > required

    def can_complete_validation(self):
        """Check if there's enough time to complete validation"""
        remaining = self.get_remaining_time()
        required = self.estimate_val_time()
        return remaining > required

    def should_stop_training(self):
        """Determine if training should stop to save checkpoint"""
        remaining = self.get_remaining_time()
        # Need at least 2 minutes to save checkpoint
        return remaining < 120

    def print_status(self, is_main=True):
        """Print time management status"""
        if not is_main:
            return

        remaining = self.get_remaining_time()
        print(f"⏱️ Time Management Status:")
        print(f"  Remaining time: {remaining / 60:.1f} minutes")
        print(f"  Avg train time: {self.estimate_train_time() / 60:.1f} minutes")
        if self.val_times:
            print(f"  Avg val time: {self.estimate_val_time() / 60:.1f} minutes")
        if self.epoch_times:
            print(f"  Avg epoch time: {self.estimate_epoch_time() / 60:.1f} minutes")


def main():
    """Main training function - Dual-Branch Cross-Domain Graph Alignment Version"""
    try:
        # Parse arguments
        parser = get_parser()
        args = parser.parse_args()

        if not args.debug_mode:
            env_debug = os.environ.get("TRAIN_DEBUG", "")
            if env_debug.lower() in {"1", "true", "yes", "y", "on", "debug"}:
                args.debug_mode = True

        args.debug_step_limit = max(1, args.debug_step_limit)
        args.debug_graph_batches = max(1, args.debug_graph_batches)
        args.debug_validate_samples = max(1, args.debug_validate_samples)

        # Initialize distributed training with extended timeout
        if is_dist():
            timeout = timedelta(minutes=args.dist_timeout)
            dist.init_process_group(backend="nccl", timeout=timeout)
            local_rank = int(os.environ["LOCAL_RANK"])
            torch.cuda.set_device(local_rank)
            world_size = dist.get_world_size()
            os.environ['NCCL_TIMEOUT'] = str(args.dist_timeout * 60)
            os.environ['NCCL_BLOCKING_WAIT'] = '1'
        else:
            local_rank = 0
            world_size = 1

        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        is_main = (not is_dist()) or dist.get_rank() == 0

        if args.debug_mode and is_main:
            print("\n🐞 Debug mode enabled")
            print(f"  Train/val step limit: {args.debug_step_limit}")
            print(f"  Graph prior debug batches: {args.debug_graph_batches}")
            print(f"  Validation debug samples: {args.debug_validate_samples}")

        # Initialize time manager
        time_manager = TimeManager(args.job_time_limit, args.time_buffer_minutes)

        # Create directories (only on main process)
        if is_main:
            os.makedirs(args.results_dir, exist_ok=True)
            # Create subdirectories for graph prior outputs
            os.makedirs(os.path.join(args.results_dir, 'graph_analysis'), exist_ok=True)

        # Set random seed
        set_determinism(seed=42 + local_rank)

        # Create tensorboard writer (only on main process)
        writer = SummaryWriter(log_dir=args.results_dir) if is_main else None

        # Create monitors (only on main process)
        dice_monitor = DiceMonitor(
            results_dir=args.results_dir,
            drop_threshold=args.dice_drop_threshold,
            window_size=args.dice_window_size,
            # Pass graph prior info for enhanced monitoring
            graph_prior_enabled=(args.prior_adj_npy is not None or args.weighted_adj_npy is not None),
            cross_domain_enabled=(args.src_prior_adj_npy is not None),
            dual_branch_enabled=True  # track dual-branch metrics
        ) if is_main else None

        # Save configuration (only on main process)
        if is_main:
            config_path = os.path.join(args.results_dir, 'config.json')
            with open(config_path, 'w') as f:
                config_dict = vars(args)
                config_dict['world_size'] = world_size
                config_dict['dist_timeout_minutes'] = args.dist_timeout
                config_dict['version'] = 'DUAL_BRANCH_CROSS_DOMAIN_GRAPH_ALIGNMENT_V2_AGEAWARE'
                config_dict['graph_prior_enabled'] = bool(args.prior_adj_npy or args.weighted_adj_npy)
                config_dict['cross_domain_alignment'] = bool(args.src_prior_adj_npy)
                config_dict['age_aware_priors'] = bool(args.volume_stats_json or args.shape_templates_pt or args.weighted_adj_npy)
                json.dump(config_dict, f, indent=2)

            print("" + "=" * 80)
            print("DAUnet TRAINING - WITH DUAL-BRANCH CROSS-DOMAIN GRAPH ALIGNMENT (Age-aware Priors)")
            print("=" * 80)
            print(f"World Size: {world_size} GPUs")
            print(f"Total Epochs: {args.epochs}")
            print(f"Job Time Limit: {args.job_time_limit} minutes")
            print(f"Time Buffer: {args.time_buffer_minutes} minutes")
            print(f"Source Registered: {args.source_is_registered}")
            print(f"Target Registered: {args.target_is_registered}")
            print(f"Target Spacing: {args.target_spacing}")
            print(f"Apply Spacing: {args.apply_spacing}")
            print(f"Apply Orientation: {args.apply_orientation}")

            # Graph prior info (target)
            if args.prior_adj_npy or args.weighted_adj_npy:
                print("🧠 TARGET DOMAIN PRIORS")
                if args.prior_adj_npy: print(f"  Binary adjacency: {args.prior_adj_npy}")
                if args.weighted_adj_npy: print(f"  Weighted adjacency: {args.weighted_adj_npy}")
                if args.prior_required_json: print(f"  Required edges: {args.prior_required_json}")
                if args.prior_forbidden_json: print(f"  Forbidden edges: {args.prior_forbidden_json}")
                if args.volume_stats_json: print(f"  Volume stats (age-aware): {args.volume_stats_json}")
                if args.shape_templates_pt: print(f"  Shape templates (SDT, age-aware): {args.shape_templates_pt}")
                if args.age_weights_json: print(f"  Age→adjacency weights: {args.age_weights_json}")

            # Graph prior info (source)
            if args.src_prior_adj_npy:
                print("🎯 SOURCE DOMAIN GRAPH PRIORS (CROSS-DOMAIN ALIGNMENT)")
                print(f"  Adjacency matrix: {args.src_prior_adj_npy}")
                if args.src_prior_required_json: print(f"  Required edges: {args.src_prior_required_json}")
                if args.src_prior_forbidden_json: print(f"  Forbidden edges: {args.src_prior_forbidden_json}")
                print(f"  Alignment Mode: {args.graph_align_mode}")
                print(f"  Lambda spectral (src): {args.lambda_spec_src or args.lambda_spec}")
                print(f"  Lambda edge (src): {args.lambda_edge_src or args.lambda_edge}")
                print(f"  Lambda spectral (tgt): {args.lambda_spec_tgt or args.lambda_spec * 0.3}")
                print(f"  Lambda edge (tgt): {args.lambda_edge_tgt or args.lambda_edge * 0.3}")
                print(f"  Lambda symmetry: {args.lambda_sym}")
                print(f"  Top-K eigenvalues: {args.graph_topr}")
                print(f"  Warmup epochs: {args.graph_warmup_epochs}")

                # Dynamic alignment parameters
                print(f"🔄 DYNAMIC SPECTRAL ALIGNMENT")
                print(f"  Lambda dynamic: {args.lambda_dyn}")
                print(f"  Dynamic top-K: {args.dyn_top_k}")
                print(f"  Start epoch: {args.dyn_start_epoch}")
                print(f"  Ramp epochs: {args.dyn_ramp_epochs}")
                print(f"  Weighted U-subspace: {args.align_U_weighted}")
                print(f"  QAP mismatch g: {args.qap_mismatch_g}")
                print(f"  Use restricted mask: {args.use_restricted_mask}")

            if args.laterality_pairs_json:
                print(f"Laterality Pairs JSON: {args.laterality_pairs_json}")
            print("=" * 80)

        # Set default class prior paths if not provided
        if args.target_prior_json is None:
            args.target_prior_json = _DEFAULT_TARGET_PRIOR
        if args.source_prior_json is None:
            args.source_prior_json = _DEFAULT_SOURCE_PRIOR

        if is_main:
            for label, path in (("target", args.target_prior_json), ("source", args.source_prior_json)):
                if not os.path.exists(path):
                    print(f"⚠️  Default {label} class prior not found at {path}")

        # Enable H100 optimizations
        if args.use_amp:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision("high")

        # Get data loaders with DDP support
        if is_main:
            print("📊 Creating data loaders...")

        data_load_start = time.time()
        source_train_loader, source_val_loader, target_train_loader, target_val_loader = get_source_target_dataloaders(
            args, is_distributed=is_dist(), world_size=world_size, rank=local_rank
        )
        data_load_time = time.time() - data_load_start

        if is_main:
            print(f"✔ Data loaders created successfully in {data_load_time / 60:.1f} minutes!")

        # Create base model
        if is_main:
            print("🏗 Creating model...")

        model_create_start = time.time()

        base_model = SwinUNETR(
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            feature_size=args.feature_size,
            drop_rate=args.model_dropout,
            attn_drop_rate=args.model_dropout,
            dropout_path_rate=args.model_dropout,
        )

        # Load pretrained weights
        base_model = load_pretrained_model(base_model, args.pretrained_model, device)
        base_model = base_model.to(device)

        # Create simplified DAUnet module (now with optional age-aware hooks)
        if is_main:
            print("🏗 Creating simplified DAUnet module...")

        model = SimplifiedDAUnetModule(
            base_model=base_model,
            num_classes=args.out_channels,
            roi_size=(args.roi_x, args.roi_y, args.roi_z),
            foreground_only=args.foreground_only,
            class_prior_path=args.target_prior_json,
            enhanced_class_weights=args.enhanced_class_weights,
            # NEW (age-aware hooks; will be ignored if module doesn't use them)
            use_age_conditioning=args.use_age_conditioning,
            age_embed_dim=args.age_embed_dim,
            volume_statistics_path=args.volume_stats_json,
            shape_prior_path=args.shape_templates_pt,
            debug_mode=args.debug_mode,
            debug_step_limit=args.debug_step_limit,
        ).to(device)

        model_create_time = time.time() - model_create_start
        if is_main:
            print(f"✔ Model created in {model_create_time / 60:.1f} minutes")

        # Create optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999),
            fused=torch.cuda.is_available()
        )

        # Wrap model with DDP (after optimizer creation)
        if is_dist():
            model = DDP(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                gradient_as_bucket_view=True,
                static_graph=False
            )
            if is_main:
                print("✔ Model wrapped with DistributedDataParallel")

        age_graph_loss = None

        # Learning rate scheduler with warmup and restarts
        def get_lr_with_warmup_restart(epoch):
            """Custom LR schedule with warmup and periodic restarts"""
            floor = args.lr_min / args.lr
            floor = float(max(0.0, min(floor, 0.999999)))

            if epoch < args.lr_warmup_epochs:
                return (epoch + 1) / args.lr_warmup_epochs

            if epoch in set(args.lr_restart_epochs):
                return 1.0

            current_segment_start = args.lr_warmup_epochs
            current_segment_end = args.epochs
            for r in args.lr_restart_epochs:
                if r <= args.lr_warmup_epochs:
                    continue
                if epoch < r:
                    current_segment_end = r
                    break
                current_segment_start = r

            denom = max(1, current_segment_end - current_segment_start)
            segment_progress = (epoch - current_segment_start) / denom
            segment_progress = max(0.0, min(1.0, segment_progress))

            return floor + (1 - floor) * 0.5 * (1 + np.cos(np.pi * segment_progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr_with_warmup_restart)

        # === CREATE GRAPH PRIOR KWARGS (extended to carry age-aware priors) ===
        actual_model = model.module if hasattr(model, 'module') else model
        class_weights = getattr(actual_model, 'class_weights', None)

        base_prior_kwargs = {
            'volume_stats_path': args.volume_stats_json,
            'shape_templates_path': args.shape_templates_pt,
            'weighted_adj_path': args.weighted_adj_npy,
            'age_weights_path': args.age_weights_json,
            'lambda_volume': args.lambda_volume,
            'lambda_shape': args.lambda_shape,
            'lambda_weighted_adj': args.lambda_weighted_adj,
            'lambda_topo': args.lambda_topo,
            'lambda_sym': args.lambda_sym,
            'prior_warmup_epochs': args.prior_warmup_epochs,
            'prior_temperature': args.prior_temperature,
            'debug_mode': args.debug_mode,
            'debug_max_batches': args.debug_graph_batches,
        }

        graph_specific_kwargs = {}
        has_prior_adj = args.prior_adj_npy and os.path.exists(args.prior_adj_npy)
        has_weighted_adj = args.weighted_adj_npy and os.path.exists(args.weighted_adj_npy)
        if has_prior_adj or has_weighted_adj:
            if is_main:
                print("🧠 Initializing Graph Prior arguments (with age-aware extensions)…")

            lambda_spec_tgt_adjusted = (args.lambda_spec_tgt or args.lambda_spec * 0.3) * 1.25
            lambda_edge_tgt_adjusted = (args.lambda_edge_tgt or args.lambda_edge * 0.3) * 1.2

            graph_specific_kwargs = {
                'prior_adj_path': args.prior_adj_npy,
                'required_json': args.prior_required_json,
                'forbidden_json': args.prior_forbidden_json,
                'lr_pairs_json': args.laterality_pairs_json,
                'src_prior_adj_path': args.src_prior_adj_npy,
                'src_required_json': args.src_prior_required_json,
                'src_forbidden_json': args.src_prior_forbidden_json,
                'lambda_spec': args.lambda_spec,
                'lambda_edge': args.lambda_edge,
                'lambda_spec_src': args.lambda_spec_src,
                'lambda_edge_src': args.lambda_edge_src,
                'lambda_spec_tgt': lambda_spec_tgt_adjusted,
                'lambda_edge_tgt': lambda_edge_tgt_adjusted,
                'top_k': args.graph_topr,
                'temperature': args.graph_temperature if args.graph_temperature is not None else args.prior_temperature,
                'warmup_epochs': args.graph_warmup_epochs if args.graph_warmup_epochs is not None else args.prior_warmup_epochs,
                'pool_kernel': args.graph_pool_kernel,
                'pool_stride': args.graph_pool_stride,
                'graph_align_mode': args.graph_align_mode,
                'class_weights': class_weights,
                'align_U_weighted': args.align_U_weighted,
                'qap_mismatch_g': args.qap_mismatch_g,
                'use_restricted_mask': args.use_restricted_mask,
                'restricted_mask_path': args.restricted_mask_path,
                'lambda_dyn': args.lambda_dyn,
                'dyn_top_k': args.dyn_top_k,
                'dyn_start_epoch': args.dyn_start_epoch,
                'dyn_ramp_epochs': args.dyn_ramp_epochs,
            }

            if is_main:
                print(f"  ✔ Graph prior kwargs prepared (age-aware: {bool(args.volume_stats_json or args.shape_templates_pt or args.weighted_adj_npy)})")

        combined_graph_kwargs = {k: v for k, v in {**base_prior_kwargs, **graph_specific_kwargs}.items() if v is not None}
        args.graph_loss_kwargs = combined_graph_kwargs if combined_graph_kwargs else None

        if combined_graph_kwargs:
            age_graph_loss = AgeConditionedGraphPriorLoss(**combined_graph_kwargs).to(device)
        elif any(base_prior_kwargs.values()):
            age_graph_loss = AgeConditionedGraphPriorLoss(**base_prior_kwargs).to(device)

        # === CHECKPOINT RESUME SUPPORT ===
        start_epoch = 1
        best_val_dice = 0.0
        best_worst10_avg = 0.0
        stagnation_counter = 0
        best_dice_history = deque(maxlen=10)

        # Load checkpoint if resuming
        if args.resume_training:
            # Try multiple checkpoint paths
            checkpoint_paths = []

            if args.checkpoint:
                checkpoint_paths.append(args.checkpoint)

            # Try latest.pth
            checkpoint_paths.append(os.path.join(args.results_dir, "latest.pth"))

            # Try to find the most recent checkpoint_epoch_*.pth
            pattern = os.path.join(args.results_dir, "checkpoint_epoch_*.pth")
            epoch_checkpoints = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
            checkpoint_paths.extend(epoch_checkpoints[:3])  # Try up to 3 most recent

            # Try to load checkpoint with fallback
            checkpoint, loaded_path = load_checkpoint_with_fallback(checkpoint_paths, device)

            if checkpoint:
                if is_main:
                    print(f"🔄 Loading checkpoint from: {loaded_path}")

                # Load model state
                target_model = model.module if hasattr(model, 'module') else model
                missing_keys, unexpected_keys = target_model.load_state_dict(
                    checkpoint['model_state_dict'], strict=False
                )

                if is_main and missing_keys:
                    print(f"⚠️ Missing keys: {missing_keys}")
                if is_main and unexpected_keys:
                    print(f"⚠️ Unexpected keys: {unexpected_keys}")

                # Load optimizer state
                if 'optimizer_state_dict' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

                # Load scheduler state (we need to step it to the right epoch)
                resumed_epoch = checkpoint.get('epoch', 0)
                for _ in range(resumed_epoch):
                    scheduler.step()

                # Load training state
                best_val_dice = checkpoint.get('best_acc', 0.0)
                start_epoch = resumed_epoch + 1

                # Load monitoring history if available
                if dice_monitor and 'dice_history' in checkpoint:
                    dice_monitor.dice_history = checkpoint['dice_history']

                if is_main:
                    print(f"✔ Resumed from epoch {resumed_epoch}")
                    print(f"  Best dice so far: {best_val_dice:.4f}")
                    print(f"  Continuing from epoch {start_epoch}")
            else:
                if is_main:
                    print("⚠️ --resume_training specified but no valid checkpoint found")
                    print("   Starting fresh training...")

        # === SIGNAL HANDLING FOR PREEMPTION ===
        current_epoch_holder = {'value': start_epoch - 1}  # Mutable holder for current epoch

        def save_latest_checkpoint(reason="signal"):
            """Save checkpoint when receiving signal or at specific times"""
            try:
                if is_main:
                    latest_path = os.path.join(args.results_dir, "latest.pth")
                    model_to_save = model.module if hasattr(model, 'module') else model

                    success = save_checkpoint_atomic(
                        model_to_save,
                        optimizer,
                        current_epoch_holder['value'],
                        best_val_dice,
                        args,
                        latest_path,
                        dice_history=dice_monitor.dice_history if dice_monitor else None,
                        additional_info={'saved_by': reason, 'timestamp': datetime.now().isoformat()}
                    )

                    if success:
                        print(f"🛟 Saved latest checkpoint at epoch {current_epoch_holder['value']} due to {reason}")
                    else:
                        print(f"⚠️ Failed to save checkpoint due to {reason}")

            except Exception as e:
                if is_main:
                    print(f"❌ Failed to save checkpoint on {reason}: {e}")

        def sigterm_handler(signum, frame):
            """Handle SIGTERM signal (sent before job time limit)"""
            # Log why we received SIGTERM
            if is_main:
                job_id = os.environ.get("SLURM_JOB_ID")
                if job_id:
                    print(f"🔍 Received SIGTERM for job {job_id}")

                    # Get sacct snapshot to understand job status
                    try:
                        result = subprocess.run(
                            ["sacct", "-j", job_id, "-X",
                             "--format=JobID,JobName,State,Elapsed,ExitCode,Reason,NodeList"],
                            capture_output=True,
                            text=True,
                            timeout=5
                        )
                        if result.returncode == 0:
                            print("🔍 Job status at SIGTERM:")
                            print(result.stdout)
                        else:
                            print(f"⚠️ sacct failed: {result.stderr}")
                    except subprocess.TimeoutExpired:
                        print("⚠️ sacct timed out")
                    except Exception as e:
                        print(f"⚠️ Failed to get sacct info: {e}")
                else:
                    print("🔍 Received SIGTERM (no SLURM_JOB_ID)")

            # Save checkpoint
            save_latest_checkpoint("SIGTERM")

            if is_main:
                print("🔄 Exiting gracefully - bash script will handle resubmission")

            # Clean up and exit
            if is_dist():
                dist.destroy_process_group()
            sys.exit(0)

        # Register signal handler
        signal.signal(signal.SIGTERM, sigterm_handler)
        signal.signal(signal.SIGINT, sigterm_handler)  # Also handle Ctrl+C

        # Clear GPU cache
        torch.cuda.empty_cache()
        gc.collect()

        # Synchronize before training
        if is_dist():
            if is_main:
                print("🔄 Synchronizing all processes before training...")
            dist.barrier()
            if is_main:
                print("✔ All processes synchronized, starting training!")

        # Training loop
        if is_main:
            print(f"🚀 Starting training from epoch {start_epoch} to {args.epochs}...")
            print(f"⏱️ Job time limit: {args.job_time_limit} minutes")
            print(f"⏱️ Time buffer: {args.time_buffer_minutes} minutes")
            print(f"📋 Training Schedule:")
            print(f"  Stage A (Prior Warmup): Epoch 1-{args.dyn_start_epoch}")
            print(
                f"  Stage B (Dynamic Ramp): Epoch {args.dyn_start_epoch}-{args.dyn_start_epoch + args.dyn_ramp_epochs}")
            print(f"  Stage C (Adaptive): Epoch {args.dyn_start_epoch + args.dyn_ramp_epochs}-{args.epochs}")
            time_manager.print_status(is_main)

        # Track why we exited the training loop
        exit_reason = None
        final_epoch = start_epoch - 1

        for epoch in range(start_epoch, args.epochs + 1):
            current_epoch_holder['value'] = epoch  # Update current epoch for signal handler
            final_epoch = epoch
            epoch_start_time = time.time()

            # Check if we have enough time to continue
            if time_manager.should_stop_training():
                if is_main:
                    print(f"⏰ Approaching job time limit, saving checkpoint and exiting gracefully...")
                    time_manager.print_status(is_main)
                save_latest_checkpoint("time_limit")
                exit_reason = "time_limit"
                break

            # Check if we can complete this epoch
            will_validate = (epoch % args.eval_num == 0)
            if not time_manager.can_complete_epoch(will_validate):
                if is_main:
                    print(f"⏰ Not enough time to complete epoch {epoch}, saving checkpoint...")
                    time_manager.print_status(is_main)
                save_latest_checkpoint("time_limit")
                exit_reason = "time_limit"
                break

            if is_main:
                print(f"{'=' * 60}")
                print(f"EPOCH {epoch}/{args.epochs}")
                # Indicate training stage
                if epoch < args.dyn_start_epoch:
                    print(f"Stage A: Prior Warmup")
                elif epoch < args.dyn_start_epoch + args.dyn_ramp_epochs:
                    print(
                        f"Stage B: Dynamic Ramp (progress: {epoch - args.dyn_start_epoch + 1}/{args.dyn_ramp_epochs})")
                else:
                    print(f"Stage C: Adaptive Convergence")
                print(f"{'=' * 60}")
                time_manager.print_status(is_main)

            try:
                # Set epoch for distributed sampler
                if is_dist():
                    if hasattr(source_train_loader.sampler, 'set_epoch'):
                        source_train_loader.sampler.set_epoch(epoch)
                    if hasattr(target_train_loader.sampler, 'set_epoch'):
                        target_train_loader.sampler.set_epoch(epoch)

                # Train one epoch (trainer will consume args.graph_loss_kwargs if present)
                train_start = time.time()
                train_metrics = train_epoch_age_aware(
                    model=model,
                    source_loader=source_train_loader,
                    target_loader=target_train_loader,
                    optimizer=optimizer,
                    epoch=epoch,
                    total_epochs=args.epochs,
                    writer=writer,
                    args=args,
                    device=device,
                    is_distributed=is_dist(),
                    world_size=world_size,
                    rank=local_rank,
                    age_graph_loss=age_graph_loss
                )
                train_time = time.time() - train_start
                time_manager.record_train_time(train_time)

                # Validate
                if epoch % args.eval_num == 0:
                    # Check if we have time for validation
                    if not time_manager.can_complete_validation():
                        if is_main:
                            print(f"⏰ Not enough time for validation, saving checkpoint...")
                            time_manager.print_status(is_main)
                        save_latest_checkpoint("time_limit_before_val")
                        exit_reason = "time_limit"
                        break

                    val_start = time.time()
                    val_metrics = val_epoch_age_aware(
                        model=model,
                        loader=target_val_loader,
                        epoch=epoch,
                        writer=writer,
                        args=args,
                        device=device,
                        is_distributed=is_dist(),
                        world_size=world_size,
                        rank=local_rank
                    )
                    val_time = time.time() - val_start
                    time_manager.record_val_time(val_time)

                    if is_main:
                        # Update monitors
                        dice_monitor.add_dice_score(
                            epoch=epoch,
                            dice_score=val_metrics['dice'],
                            train_metrics=train_metrics,
                            val_metrics=val_metrics
                        )

                        # Calculate worst 10 average
                        worst10_avg = 0.0
                        if val_metrics.get('dice_per_class'):
                            dice_per_class = np.array(val_metrics['dice_per_class'])
                            sorted_dice = np.sort(dice_per_class)
                            worst10_avg = np.mean(sorted_dice[:10])

                        # Track best dice history
                        best_dice_history.append(val_metrics['dice'])

                        # Save best model based on worst-10 average
                        if worst10_avg > best_worst10_avg:
                            improvement = worst10_avg - best_worst10_avg
                            best_worst10_avg = worst10_avg
                            best_val_dice = val_metrics['dice']

                            checkpoint_path = os.path.join(args.results_dir, 'best_model.pth')
                            model_to_save = model.module if is_dist() else model

                            save_checkpoint_atomic(
                                model_to_save, optimizer, epoch, best_val_dice, args, checkpoint_path,
                                dice_history=dice_monitor.dice_history,
                            )

                            print(f"✨ New best Worst-10 Avg: {best_worst10_avg:.4f} (+{improvement:.4f})")
                            print(f"   Overall Dice: {best_val_dice:.4f}")

                        # Check for stagnation
                        if len(best_dice_history) >= 10:
                            recent_improvement = max(best_dice_history) - min(best_dice_history)
                            if recent_improvement < 0.005:
                                stagnation_counter += 1
                            else:
                                stagnation_counter = 0

                # Record epoch time
                epoch_time = time.time() - epoch_start_time
                time_manager.record_epoch_time(epoch_time)

                scheduler.step()

                # Save periodic checkpoint
                if is_main and epoch % args.save_interval == 0:
                    checkpoint_path = os.path.join(args.results_dir, f'checkpoint_epoch_{epoch}.pth')
                    model_to_save = model.module if is_dist() else model

                    save_checkpoint_atomic(
                        model_to_save, optimizer, epoch, best_val_dice, args, checkpoint_path,
                        dice_history=dice_monitor.dice_history if dice_monitor else None,
                    )

                # Clear cache periodically
                if epoch % 20 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

                # Early stopping check
                if args.early_stopping and stagnation_counter >= args.early_stopping_patience:
                    if is_main:
                        print(f"⛔ Early stopping triggered at epoch {epoch}")
                    exit_reason = "early_stopping"
                    break

            except Exception as e:
                if is_main:
                    print(f"❌ Error in epoch {epoch}: {str(e)}")
                    traceback.print_exc()
                    # Save emergency checkpoint
                    save_latest_checkpoint("error")
                exit_reason = "error"
                raise

        # Determine if training actually completed
        training_completed = (final_epoch >= args.epochs) and (exit_reason is None)

        # Only save final model if training actually completed all epochs
        if is_main:
            if training_completed:
                # Training completed successfully - save final model
                final_checkpoint_path = os.path.join(args.results_dir, 'final_model.pth')
                model_to_save = model.module if is_dist() else model

                save_checkpoint_atomic(
                    model_to_save, optimizer, args.epochs, best_val_dice, args, final_checkpoint_path,
                    dice_history=dice_monitor.dice_history if dice_monitor else None,
                    additional_info={'training_complete': True}
                )

                # Generate final reports
                print("📊 Generating final reports...")
                if dice_monitor:
                    dice_monitor.generate_report()
                    dice_monitor.plot_dice_evolution()
                    if hasattr(dice_monitor, 'generate_cross_domain_report'):
                        dice_monitor.generate_cross_domain_report()
                    if hasattr(dice_monitor, 'generate_dual_branch_report'):
                        dice_monitor.generate_dual_branch_report()

                print("" + "=" * 80)
                print("✅ TRAINING COMPLETED SUCCESSFULLY!")
                print("=" * 80)
                print(f"Total epochs trained: {args.epochs}")
                print(f"Best Overall Dice: {best_val_dice:.4f}")
                print(f"Best Worst-10 Avg: {best_worst10_avg:.4f}")
                print(f"Final model: {final_checkpoint_path}")
                print("=" * 80)
            else:
                # Training paused/stopped - do NOT save final model
                print("" + "=" * 80)
                if exit_reason == "time_limit":
                    print("⏸ TRAINING PAUSED DUE TO TIME LIMIT")
                    print(f"Stopped at epoch {final_epoch}/{args.epochs}")
                    print("Training will resume from checkpoint in next job")
                elif exit_reason == "early_stopping":
                    print("ℹ️ TRAINING STOPPED DUE TO EARLY STOPPING")
                    print(f"Stopped at epoch {final_epoch}/{args.epochs}")
                elif exit_reason == "error":
                    print("❌ TRAINING STOPPED DUE TO ERROR")
                    print(f"Stopped at epoch {final_epoch}/{args.epochs}")
                else:
                    print("⏸ TRAINING PAUSED")
                    print(f"Stopped at epoch {final_epoch}/{args.epochs}")

                print("=" * 80)
                print(f"Best Overall Dice so far: {best_val_dice:.4f}")
                print(f"Best Worst-10 Avg so far: {best_worst10_avg:.4f}")
                print(f"Latest checkpoint: {args.results_dir}/latest.pth")

                # Generate intermediate reports if stopped early
                if dice_monitor:
                    dice_monitor.generate_report()
                    dice_monitor.plot_dice_evolution()
                    if hasattr(dice_monitor, 'generate_cross_domain_report'):
                        dice_monitor.generate_cross_domain_report()
                    if hasattr(dice_monitor, 'generate_dual_branch_report'):
                        dice_monitor.generate_dual_branch_report()

                print("=" * 80)

            if writer:
                writer.close()

        # Clean up distributed training
        if is_dist():
            dist.destroy_process_group()

    except Exception as e:
        is_main = (not is_dist()) or dist.get_rank() == 0
        if is_main:
            print(f"❌ Training failed with error: {str(e)}")
            traceback.print_exc()
            # Try to save emergency checkpoint
            try:
                # local helper in scope of main
                def _save_latest(reason="crash"):
                    latest_path = os.path.join(args.results_dir, "latest.pth")
                    model_to_save = model.module if hasattr(model, 'module') else model
                    save_checkpoint_atomic(
                        model_to_save, optimizer, current_epoch_holder['value'] if 'current_epoch_holder' in locals() else -1,
                        0.0, args, latest_path,
                        dice_history=dice_monitor.dice_history if 'dice_monitor' in locals() and dice_monitor else None,
                        additional_info={'saved_by': reason, 'timestamp': datetime.now().isoformat()}
                    )
                _save_latest("crash")
            except Exception:
                pass
        if is_dist():
            dist.destroy_process_group()
        sys.exit(1)


def safe_main():
    """Wrapper to capture per-rank exceptions for better debugging"""
    # Parse args first to get results_dir
    parser = get_parser()
    args = parser.parse_args()

    # Get rank info
    rank = int(os.environ.get("LOCAL_RANK", -1))
    world_rank = int(os.environ.get("RANK", -1))

    # Create error log path
    os.makedirs(args.results_dir, exist_ok=True)
    err_path = os.path.join(args.results_dir, f"crash_rank{rank}_world{world_rank}.log")

    try:
        # Add extra debug info at startup
        if rank >= 0:
            print(f"[Rank {rank}] Starting safe_main wrapper")
            print(f"[Rank {rank}] Error log will be saved to: {err_path}")
            print(f"[Rank {rank}] CUDA device count: {torch.cuda.device_count()}")
            print(
                f"[Rank {rank}] CUDA current device: {torch.cuda.current_device() if torch.cuda.is_available() else 'N/A'}")

        # Call the actual main function
        main()

    except Exception as e:
        # Write detailed error information to rank-specific file
        try:
            with open(err_path, "w") as f:
                f.write(f"{'=' * 80}")
                f.write(f"CRASH REPORT - Rank {rank} (World Rank {world_rank})")
                f.write(f"Time: {datetime.now().isoformat()}")
                f.write(f"{'=' * 80}")

                # Environment info
                f.write("Environment Variables:")
                f.write(f"  LOCAL_RANK: {os.environ.get('LOCAL_RANK', 'N/A')}")
                f.write(f"  RANK: {os.environ.get('RANK', 'N/A')}")
                f.write(f"  WORLD_SIZE: {os.environ.get('WORLD_SIZE', 'N/A')}")
                f.write(f"  MASTER_ADDR: {os.environ.get('MASTER_ADDR', 'N/A')}")
                f.write(f"  MASTER_PORT: {os.environ.get('MASTER_PORT', 'N/A')}")
                f.write(f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'N/A')}")
                f.write(f"")

                # CUDA/GPU info
                if torch.cuda.is_available():
                    f.write("CUDA/GPU Information:")
                    f.write(f"  CUDA available: True")
                    f.write(f"  CUDA device count: {torch.cuda.device_count()}")
                    try:
                        f.write(f"  Current device: {torch.cuda.current_device()}")
                        f.write(f"  Device name: {torch.cuda.get_device_name()}")
                        f.write(f"  Memory allocated: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
                        f.write(f"  Memory reserved: {torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB")
                    except:
                        f.write(f"  Failed to get CUDA device info")
                else:
                    f.write("CUDA/GPU Information: CUDA not available")
                f.write(f"")

                # Exception details
                f.write(f"Exception Type: {type(e).__name__}")
                f.write(f"Exception Message: {str(e)}")

                f.write("Full Traceback:")
                f.write("-" * 40 + "")
                traceback.print_exc(file=f)
                f.write("-" * 40 + "")

                # Try to get more context for common errors
                if "CUDA" in str(e) or "cuda" in str(e).lower():
                    f.write("Additional CUDA Error Context:")
                    try:
                        # Force CUDA synchronization to get any pending errors
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                    except Exception as cuda_e:
                        f.write(f"  CUDA synchronize error: {cuda_e}")

                if "out of memory" in str(e).lower():
                    f.write("Memory Error - Current Memory Status:")
                    if torch.cuda.is_available():
                        try:
                            f.write(f"  Allocated: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
                            f.write(f"  Reserved: {torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB")
                            f.write(f"  Max allocated: {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB")
                        except:
                            f.write("  Failed to get memory stats")

                f.write(f"{'=' * 80}")
                f.write("END OF CRASH REPORT")
                f.write(f"{'=' * 80}")

            # Also print to stdout/stderr for immediate visibility
            print(f"❌ [Rank {rank}] CRASHED with {type(e).__name__}: {str(e)}")
            print(f"   Full traceback saved to: {err_path}")

        except Exception as write_error:
            # If we can't even write the error file, at least print it
            print(f"❌ [Rank {rank}] CRASHED and FAILED TO WRITE ERROR LOG: {write_error}")
            print(f"   Original error was: {type(e).__name__}: {str(e)}")
            traceback.print_exc()

        # Re-raise to let elastic/torchrun handle it properly
        raise


if __name__ == "__main__":
    safe_main()
