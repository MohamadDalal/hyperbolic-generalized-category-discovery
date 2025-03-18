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

#import warnings
#warnings.filterwarnings("ignore", category=DeprecationWarning)



def test_cluster(dataloader, centers, args):
    # Find pairwise distance between test features and cluster centers
    # Use argmin function to assign each feature to the class of closest center
    # Pass predictions with ground truth's and masks to log_accs_from_preds to get accuracies
    preds = []
    targets = []
    mask = []
    # First extract all features
    for batch_idx, (feats, label, _, mask_lab_) in enumerate(tqdm(test_loader)):

        feats = feats.to(device)
        label = label.to(device)

        feats = torch.nn.functional.normalize(feats, dim=-1)

        dist = pairwise_distance(feats, centers)
        pred = torch.argmin(dist, dim=1)

        preds = np.append(preds, pred.cpu().numpy())
        targets = np.append(targets, label.cpu().numpy())
        mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes))
                                         else False for x in label]))

    return log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask, eval_funcs=args.eval_funcs, save_name="")

    



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
    parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v1', 'v2'])

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
    all_acc, old_acc, new_acc= test_cluster(test_loader, cluster_centers, args)
    print(f"all: {all_acc}, old: {old_acc}, new: {new_acc}")