import argparse
import os

from torch.utils.data import DataLoader
import numpy as np
from sklearn.cluster import KMeans
import torch
from project_utils.cluster_utils import str2bool
from project_utils.general_utils import seed_torch
from project_utils.cluster_and_log_utils import log_accs_from_preds

from methods.clustering.feature_vector_dataset import FeatureVectorDataset
from methods.clustering.faster_mix_k_means_pytorch import pairwise_distance
from data.get_datasets import get_datasets, get_class_splits

from tqdm import tqdm
from config import feature_extract_dir
import project_utils.lorentz as L
import project_utils.poincare as P

#import warnings
#warnings.filterwarnings("ignore", category=DeprecationWarning)



def test_cluster(dataloader, centers, args, device, return_ind = False, return_preds=False):
    # Find pairwise distance between test features and cluster centers
    # Use argmin function to assign each feature to the class of closest center
    # Pass predictions with ground truth's and masks to log_accs_from_preds to get accuracies
    preds = []
    targets = []
    mask = []
    uq_idxs = []
    # debug_dist = torch.tensor([[]], device=device)
    # debug_data = None
    centers = centers[:-1]
    # First extract all features
    for batch_idx, (feats, label, uq_idx, mask_lab_) in enumerate(tqdm(dataloader)):

        feats = feats.to(device)
        label = label.to(device)

        if args.hyperbolic:
            if args.poincare:
                dist = P.pairwise_dist(feats, centers, -args.curvature, 1e-6)
            else:
                dist = L.pairwise_dist(feats, centers, args.curvature, 1e-6)
        else:
            feats = torch.nn.functional.normalize(feats, dim=-1)
            dist = pairwise_distance(feats, centers)
        #print(dist.shape)
        # if batch_idx == 0:
        #     debug_dist = dist
        #     debug_data = feats
        # else:
        #     debug_dist = torch.concatenate([debug_dist, dist], dim=0)
        #     debug_data = torch.concatenate([debug_data, feats], dim=0)
        pred = torch.argmin(dist, dim=1)

        preds = np.append(preds, pred.cpu().numpy())
        targets = np.append(targets, label.cpu().numpy())
        mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes))
                                         else False for x in label]))
        uq_idxs = np.append(uq_idxs, uq_idx.cpu().numpy())
        if return_ind:
            torch.save(preds, os.path.join(args.debug_test_dir, "predictions"))
            torch.save(targets, os.path.join(args.debug_test_dir, "targets"))
            torch.save(mask, os.path.join(args.debug_test_dir, "mask"))

    # print(debug_dist.shape)
    # print(debug_dist.mean(dim=0))
    # print(debug_dist.std(dim=0))
    # print(debug_data.shape)
    # print(debug_data.norm(dim=1).mean())
    # print(debug_data.norm(dim=1).max())
    # print(debug_data.norm(dim=1).min())
    # print(debug_data.norm(dim=1).std())
    # print(centers.shape)
    # print(centers.norm(dim=1).mean())
    # print(centers.norm(dim=1).max())
    # print(centers.norm(dim=1).min())
    # print(centers.norm(dim=1).std())
    # #print(debug_data.abs().max(dim=0)[0])
    # #print(debug_data.abs().min(dim=0)[0])
    # print(debug_data.std(dim=0))
    # print(centers.std(dim=0))
    # print(preds.shape)
    # # if args.hyperbolic:
    # #     if args.poincare:
    # #         dist = P.pairwise_dist(debug_data, debug_data, -args.curvature, 1e-6)
    # #     else:
    # #         dist = L.pairwise_dist(debug_data, debug_data, args.curvature, 1e-6)
    # # else:
    # #     debug_data = torch.nn.functional.normalize(debug_data, dim=-1)
    # #     dist = pairwise_distance(debug_data, debug_data)
    # # print(dist.shape)
    # # print(dist.mean(dim=0))
    # # print(dist.std(dim=0))
    print(np.unique(preds, return_counts=True))
    if return_preds:
        return log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask, eval_funcs=args.eval_funcs, save_name="", print_output=True, return_ind=return_ind), targets, preds, uq_idxs
    else:
        return log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask, eval_funcs=args.eval_funcs, save_name="", print_output=True, return_ind=return_ind)

    



if __name__ == "__main__":

    parser = argparse.ArgumentParser(
            description='cluster',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--root_dir', type=str, default=feature_extract_dir)
    parser.add_argument('--warmup_model_exp_id', type=str, default=None)
    parser.add_argument('--model_name', type=str, default='vit_dino', help='Format is {model_name}_{pretrain}')
    parser.add_argument('--dataset_name', type=str, default='aircraft', help='options: cifar10, cifar100, scars')
    parser.add_argument('--prop_train_labels', type=float, default=0.5) # Decides what percentage of the labelled dataset to use?
    parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v2'])
    parser.add_argument('--hyperbolic', type=str2bool, default=False)
    parser.add_argument('--poincare', type=str2bool, default=False)

    # ----------------------
    # INIT
    # ----------------------
    args = parser.parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    cluster_accs = {}

    args.save_dir = os.path.join(args.root_dir, f'{args.model_name}_{args.dataset_name}')

    args = get_class_splits(args)

    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)

    print(args)

    if args.warmup_model_exp_id is not None:
        args.save_dir += '_' + args.warmup_model_exp_id
        print(f'Using features from experiment: {args.warmup_model_exp_id}')
    else:
        print(f'Using pretrained {args.model_name} features...')

    print(args.save_dir)
    args.debug_test_dir = os.path.join(args.save_dir, "debug_test")
    print("Will output debug stuff in:", args.debug_test_dir)
    if not os.path.exists(args.debug_test_dir):
        print(args.debug_test_dir, "does not exist. Creating...")
        os.makedirs(args.debug_test_dir)

    # --------------------
    # LOAD PARAMETERS
    # --------------------

    if args.hyperbolic:
        extra_params = torch.load(os.path.join(args.save_dir, 'extra_params.pth'), map_location=device)
        args.curvature = extra_params['curvature']
        print(f'Loaded params: {extra_params}')

    # --------------------
    # DATASETS
    # --------------------
    print('Building datasets...')
    train_transform, test_transform = None, None
    _, test_dataset, _, datasets = get_datasets(args.dataset_name, train_transform, test_transform, args)

    # Convert to feature vector dataset
    test_dataset = FeatureVectorDataset(base_dataset=test_dataset, feature_root=os.path.join(args.save_dir, 'test'))
    test_loader = DataLoader(test_dataset, num_workers=args.num_workers,
                                      batch_size=args.batch_size, shuffle=False)

    # Load cluster
    print('Loading cluser centers...')
    cluster_load_path = os.path.join(args.save_dir, 'ss_kmeans_cluster_centres.pt')
    cluster_centers = torch.load(cluster_load_path, weights_only=False)

    print('Testing cluster centers on test set...')
    all_acc, old_acc, new_acc, ind, w = test_cluster(test_loader, cluster_centers, args, device, return_ind=True)
    torch.save(ind, os.path.join(args.debug_test_dir, "index_map"))
    torch.save(w, os.path.join(args.debug_test_dir, "scores_matrix"))

    print(f"all: {all_acc}, old: {old_acc}, new: {new_acc}")