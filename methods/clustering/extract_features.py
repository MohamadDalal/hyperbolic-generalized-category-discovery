import torch
from torch.utils.data import DataLoader
import timm
from torchvision import transforms
import torchvision

import argparse
import os
from tqdm import tqdm

from data.stanford_cars import CarsDataset
from data.cifar import CustomCIFAR10, CustomCIFAR100, cifar_10_root, cifar_100_root
from data.herbarium_19 import HerbariumDataset19, herbarium_dataroot
from data.augmentations import get_transform
from data.imagenet import get_imagenet_100_datasets
from data.data_utils import MergedDataset
from data.cub import CustomCub2011, cub_root
from data.fgvc_aircraft import FGVCAircraft, aircraft_root

from project_utils.general_utils import strip_state_dict, str2bool
from copy import deepcopy

from config import feature_extract_dir#, dino_pretrain_path
from models import vision_transformer as vits

def extract_features_dino(model, loader, save_dir):

    model.to(device)
    model.eval()

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader)):

            images, labels, idxs = batch[:3]
            images = images.to(device)

            if args.remove_dyno_head:
                features = model(images)
            else:
                features, _ = model(images)         # CLS_Token for ViT, Average pooled vector for R50

            # Save features
            for f, t, uq in zip(features, labels, idxs):

                t = t.item()
                uq = uq.item()

                save_path = os.path.join(save_dir, f'{t}', f'{uq}.npy')
                torch.save(f.detach().cpu().numpy(), save_path)


def extract_features_timm(model, loader, save_dir):

    model.to(device)
    model.eval()

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader)):

            images, labels, idxs = batch[:3]
            images = images.to(device)

            features, _ = model.forward_features(images)         # CLS_Token for ViT, Average pooled vector for R50

            # Save features
            for f, t, uq in zip(features, labels, idxs):

                t = t.item()
                uq = uq.item()

                save_path = os.path.join(save_dir, f'{t}', f'{uq}.npy')
                torch.save(f.detach().cpu().numpy(), save_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
            description='cluster',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--root_dir', type=str, default=feature_extract_dir)
    parser.add_argument('--warmup_model_dir', type=str,
                        default=None)
    parser.add_argument('--use_best_model', type=str2bool, default=True)
    parser.add_argument('--model_name', type=str, default='vit_dino', help='Format is {model_name}_{pretrain}')
    parser.add_argument('--dataset', type=str, default='aircraft', help='options: cifar10, cifar100, scars')
    parser.add_argument('--exp_id', type=str, default="")
    parser.add_argument('--hyperbolic', type=str2bool, default=False)
    parser.add_argument('--poincare', type=str2bool, default=False)
    parser.add_argument('--euclidean_clipping', type=float, default=None)
    parser.add_argument('--remove_dyno_head', type=str2bool, default=False)
    parser.add_argument('--mlp_out_dim', type=int, default=768)
    parser.add_argument('--use_dinov2', type=str2bool, default=False)

    # ----------------------
    # INIT
    # ----------------------
    args = parser.parse_args()
    device = torch.device('cuda:0')

    args.save_dir = os.path.join(args.root_dir, f'{args.model_name}_{args.dataset}')
    print(args)

    print('Loading model...')
    # ----------------------
    # MODEL
    # ----------------------
    if args.model_name == 'vit_dino':

        extract_features_func = extract_features_dino
        args.interpolation = 3
        args.crop_pct = 0.875
        #pretrain_path = dino_pretrain_path

        model = torch.hub.load('facebookresearch/dinov2:main', 'dinov2_vitb14') if args.use_dinov2 else torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')

        #state_dict = torch.load(pretrain_path, map_location='cpu')
        #model.load_state_dict(state_dict)

        # NOTE: Hardcoded image size as we do not finetune the entire ViT model
        args.image_size = 224
        args.feat_dim = 768
        args.num_mlp_layers = 3
        #args.mlp_out_dim = 65536
        #args.mlp_out_dim = 768

        _, val_transform = get_transform('imagenet', image_size=args.image_size, args=args)
    else:

        raise NotImplementedError

    print("Loading projection head...")

    # ----------------------
    # PROJECTION HEAD
    # ----------------------
    if args.hyperbolic:
        projection_head = vits.__dict__['Hyperbolic_DINOHead'](in_dim=args.feat_dim, out_dim=args.mlp_out_dim,
                                                               nlayers=args.num_mlp_layers, learn_curv=False,
                                                               poincare = args.poincare,
                                                               euclidean_clip_value=args.euclidean_clipping)
    else:
        projection_head = vits.__dict__['DINOHead'](in_dim=args.feat_dim, out_dim=args.mlp_out_dim,
                                                    nlayers=args.num_mlp_layers)
    model = torch.nn.Sequential(model, projection_head).to(device)

    print("Loading model weights...")

    if args.warmup_model_dir is not None:

        #warmup_id = args.warmup_model_dir.split('(')[1].split(')')[0]

        if args.use_best_model:
            args.warmup_model_dir = args.warmup_model_dir[:-3] + '_best.pt'
            if len(args.warmup_model_dir.split('(')) > 1:
                args.save_dir += '_(' + args.warmup_model_dir.split('(')[1].split(')')[0] + ')_best'
            else:
                args.save_dir += args.exp_id + '_best'
            print(f'Using weights from {args.warmup_model_dir} ...')
            state_dict = torch.load(args.warmup_model_dir)
        else:
            if len(args.warmup_model_dir.split('(')) > 1:
                args.save_dir += '_(' + args.warmup_model_dir.split('(')[1].split(')')[0] + ')'
            else:
                args.save_dir += args.exp_id
            print(f'Using weights from {args.warmup_model_dir} ...')
            checkpoint = torch.load(args.warmup_model_dir)#["model_state_dict"]
            if checkpoint.get('model_state_dict', None) is not None:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint

        
        model.load_state_dict(state_dict)
        if args.remove_dyno_head:
            model = model[0]
            args.hyperbolic = False

        print(f'Saving to {args.save_dir}')

    print('Loading data...')
    # ----------------------
    # DATASET
    # ----------------------
    if args.dataset == 'cifar10':

        train_dataset = CustomCIFAR10(root=cifar_10_root, train=True, transform=val_transform)
        test_dataset = CustomCIFAR10(root=cifar_10_root, train=False, transform=val_transform)
        targets = list(set(train_dataset.targets))

    elif args.dataset == 'cifar100':

        train_dataset = CustomCIFAR100(root=cifar_100_root, train=True, transform=val_transform)
        test_dataset = CustomCIFAR100(root=cifar_100_root, train=False, transform=val_transform)
        targets = list(set(train_dataset.targets))

    elif args.dataset == 'scars':

        train_dataset = CarsDataset(train=True, transform=val_transform)
        test_dataset = CarsDataset(train=False, transform=val_transform)
        targets = list(set(train_dataset.target))
        targets = [i - 1 for i in targets]          # SCars are labelled 1 - 197. Change to 0 - 196

    elif args.dataset == 'herbarium_19':

        train_dataset = HerbariumDataset19(root=os.path.join(herbarium_dataroot, 'small-train'),
                                           transform=val_transform)

        test_dataset = HerbariumDataset19(root=os.path.join(herbarium_dataroot, 'small-validation'),
                                           transform=val_transform)

        targets = list(set(train_dataset.targets))

    elif args.dataset == 'imagenet_100':

        datasets = get_imagenet_100_datasets(train_transform=val_transform, test_transform=val_transform,
                                             train_classes=range(50),
                                             prop_train_labels=0.5)

        datasets['train_labelled'].target_transform = None
        datasets['train_unlabelled'].target_transform = None

        train_dataset = MergedDataset(labelled_dataset=deepcopy(datasets['train_labelled']),
                                      unlabelled_dataset=deepcopy(datasets['train_unlabelled']))

        test_dataset = datasets['test']
        targets = list(set(test_dataset.targets))

    elif args.dataset == 'cub':

        train_dataset = CustomCub2011(root=cub_root, transform=val_transform, train=True)
        test_dataset = CustomCub2011(root=cub_root, transform=val_transform, train=False)
        targets = list(set(train_dataset.data.target.values))
        targets = [i - 1 for i in targets]          # SCars are labelled 1 - 200. Change to 0 - 199

    elif args.dataset == 'aircraft':

        train_dataset = FGVCAircraft(root=aircraft_root, transform=val_transform, split='trainval')
        test_dataset = FGVCAircraft(root=aircraft_root, transform=val_transform, split='test')
        targets = list(set([s[1] for s in train_dataset.samples]))

    else:

        raise NotImplementedError

    # ----------------------
    # DATALOADER
    # ----------------------
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print('Creating base directories...')
    # ----------------------
    # INIT SAVE DIRS
    # Create a directory for each class
    # ----------------------
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    for fold in ('train', 'test'):

        fold_dir = os.path.join(args.save_dir, fold)
        if not os.path.exists(fold_dir):
            os.mkdir(fold_dir)

        for t in targets:
            target_dir = os.path.join(fold_dir, f'{t}')
            if not os.path.exists(target_dir):
                os.mkdir(target_dir)

    # ----------------------
    # EXTRACT FEATURES
    # ----------------------
    # Extract train features
    train_save_dir = os.path.join(args.save_dir, 'train')
    print('Extracting features from train split...')
    extract_features_func(model=model, loader=train_loader, save_dir=train_save_dir)

    # Extract test features
    test_save_dir = os.path.join(args.save_dir, 'test')
    print('Extracting features from test split...')
    extract_features_func(model=model, loader=test_loader, save_dir=test_save_dir)

    # Save model parameters
    if args.hyperbolic:
        print('Saving hyperbolic curvature and projection alpha...')
        print(f'Curvature: {model[1].get_curvature().item()}')
        print(f'Projection alpha: {model[1].get_proj_alpha()}')
        torch.save({"curvature": model[1].get_curvature().item(), "proj_alpha": model[1].get_proj_alpha()}, os.path.join(args.save_dir, 'extra_params.pth'))

    print('Done!')