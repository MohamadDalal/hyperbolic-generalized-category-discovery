import argparse
import os

from torch.utils.data import DataLoader
import numpy as np
from sklearn.cluster import KMeans
import torch
from torch.optim import SGD, lr_scheduler
from project_utils.cluster_utils import mixed_eval, AverageMeter
from models import vision_transformer as vits

from project_utils.general_utils import init_experiment, get_mean_lr, str2bool, get_dino_head_weights

from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_class_splits

from tqdm import tqdm

from torch.nn import functional as F

from project_utils.cluster_and_log_utils import log_accs_from_preds
from config import exp_root, dino_pretrain_path

import project_utils.lorentz as L
from methods.clustering.faster_mix_k_means_pytorch import K_Means as SemiSupKMeans
import wandb
from argparse import Namespace

# TODO: Debug
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# TODO: Consider using a learnable temperature like in CLIP and MERU
class SupConLoss(torch.nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR
    From: https://github.com/HobbitLong/SupContrast"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, hyperbolic=False):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.hyperbolic = hyperbolic

    def forward(self, features, labels=None, mask=None, curv=1.0):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
            curv: curvature to use when computing hyperbolic distance
        Returns:
            A loss scalar.
        """

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits. DONE: Make sure that the direction of the values is correct after stabilizing
        if self.hyperbolic:
            # Result of this: Highest distance will be lowest value. Lowest distance will be 0
            minus_distance = - L.pairwise_dist(anchor_feature, contrast_feature, curv=curv, eps=1e-6) / self.temperature
            M = minus_distance

            # for numerical stability, as soft max is translation invariant
            logits_max, _ = torch.max(M[~torch.eye(*M.shape,dtype = torch.bool)].view(M.shape[0], M.shape[1]-1), dim=1, keepdim=True)
            logits = minus_distance - logits_max.detach()
            #logits = minus_distance

        else:
            # Result of this: Lowest similarity will be lowest value. Highest similarity will be 0
            anchor_dot_contrast = torch.div(
                torch.matmul(anchor_feature, contrast_feature.T),
                self.temperature)

            # for numerical stability, as soft max is translation invariant
            logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
            logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases. Setting 0 on the diagonal and 1 for the rest
        # I need to read on what scatter does, but this code gives a matrix of 1 with 0 on the diagonal
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        # Remove diagonal from tiled mask
        mask = mask * logits_mask

        # compute log_prob (softmax)
        exp_logits = torch.exp(logits*logits_mask) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        labels = ("logits", "exp_logits", "log_prob", "mean_log_prob_pos")
        for n, i in enumerate((logits, exp_logits, log_prob, mean_log_prob_pos)):
            if True in i.isnan():
                print(f"{labels[n]} are NaN")
                torch.set_printoptions(profile="full")
                print(i)
                torch.set_printoptions(profile="default")
                print(i.mean())
                print(i.std())
                print(curv)
                for m, j in enumerate((logits, exp_logits, log_prob, mean_log_prob_pos)):
                    torch.save(j, os.path.join(DEBUG_DIR, f"{labels[m]}_debug.pt"))
                    wandb.save(os.path.join(DEBUG_DIR, f"{labels[m]}_debug.pt"))
                torch.save(features, os.path.join(DEBUG_DIR, f"features_debug.pt"))
                wandb.save(os.path.join(DEBUG_DIR, f"features_debug.pt"))
                #wandb.log({labels[n]: i})
                raise ValueError(f'{labels[n]} have NaN')

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss, ((logits.mean(), logits.std(), logits.max(), logits.min()),
                      (torch.exp(logits).mean(), torch.exp(logits).std(), torch.exp(logits).max(), torch.exp(logits).min()),
                      (exp_logits.mean(), exp_logits.std(), exp_logits.max(), exp_logits.min()),
                      (log_prob.mean(), log_prob.std(), log_prob.max(), log_prob.min()),
                      (mean_log_prob_pos.mean(), mean_log_prob_pos.std(), mean_log_prob_pos.max(), mean_log_prob_pos.min()))


class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)]

# TODO: Check why this does not use temperature, while supervised loss uses temperature
def info_nce_logits(features, args, curv=1.0):

    b_ = 0.5 * int(features.size(0))

    labels = torch.cat([torch.arange(b_) for i in range(args.n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)

    if args.hyperbolic:
        similarity_matrix = - L.pairwise_dist(features, features, curv=curv, eps=1e-6)
        if True in similarity_matrix.isnan():
            #print(similarity_matrix)
            print("Hyperbolic distance is NaN")
            torch.set_printoptions(profile="full")
            print(similarity_matrix)
            torch.set_printoptions(profile="default")
            print(similarity_matrix.mean())
            print(similarity_matrix.std())
            print(curv)
            #wandb.log({"logits": similarity_matrix})
            raise ValueError('Hyperbolic distance has NaN')
        #print(similarity_matrix.mean())
        #print(similarity_matrix.std())
        #print(curv)
    else:
        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    logits = logits / args.temperature
    return logits, labels


def train(model, train_loader, test_loader, unlabelled_train_loader, args, optimizer, scheduler,
          best_test_acc = 0, start_epoch = 0):

    sup_con_crit = SupConLoss(hyperbolic=args.hyperbolic)
    best_test_acc_lab = best_test_acc
    freeze_curv_for_warmup = args.freeze_curvature.lower() == "warmup" and args.hyperbolic
    if freeze_curv_for_warmup:
        model[1].train_curvature(False)

    for epoch in range(start_epoch, args.epochs):

        loss_record = AverageMeter()
        con_loss_record = AverageMeter()
        sup_con_loss_record = AverageMeter()
        train_acc_record = AverageMeter()

        if epoch >= args.epochs_warmup and freeze_curv_for_warmup:
            model[1].train_curvature(True)
            freeze_curv_for_warmup = False
            print("Unfreezing curvature at epoch {}".format(epoch))

        model.train()

        for batch_idx, batch in enumerate(tqdm(train_loader)):
            #with torch.autograd.detect_anomaly(check_nan=True):
            if True:
                step_log_dict = {}

                images, class_labels, uq_idxs, mask_lab = batch
                mask_lab = mask_lab[:, 0]

                class_labels, mask_lab = class_labels.to(device), mask_lab.to(device).bool()
                images = torch.cat(images, dim=0).to(device)

                # Extract features with model
                features, output_log_stats = model(images)

                # L2-normalize features if not using hyperbolic space
                features = features if args.hyperbolic else torch.nn.functional.normalize(features, dim=-1)

                # Choose which instances to run the contrastive loss on
                if args.contrast_unlabel_only:
                    # Contrastive loss only on unlabelled instances
                    f1, f2 = [f[~mask_lab] for f in features.chunk(2)]
                    con_feats = torch.cat([f1, f2], dim=0)
                else:
                    # Contrastive loss for all examples
                    con_feats = features

                # In normal contrastive learning we use similarity measures
                # So we doing cross entropy we assign the target to the positive pair
                # Such that loss is decreased when positive pairs have higher similarity and negative pairs have lower similarities
                # In cosine similarity this would correspond to highest being 1 and lowest being -1
                # In hyperbolic space we use the distance measure, which is the opposite. That is why we take the negative of the distance
                # We still assign the target to the positive pair
                # Now positive pairs need lower distance and negative pairs need higher distance (before taking minus of distance)
                # Since the range has changed from (-1,1) to (-infty to 0) the loss can behave differently. I have not investigated how different that is 
                if args.hyperbolic:
                    contrastive_logits, contrastive_labels = info_nce_logits(features=con_feats, args=args, curv=model[1].get_curvature())
                else:
                    contrastive_logits, contrastive_labels = info_nce_logits(features=con_feats, args=args)
                # TODO: Do we need to use a hyperbolic cross entropy loss? I forgot to consider that.
                contrastive_loss = torch.nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels)

                # Supervised contrastive loss
                f1, f2 = [f[mask_lab] for f in features.chunk(2)]
                sup_con_feats = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                sup_con_labels = class_labels[mask_lab]

                if args.hyperbolic:
                    sup_con_loss, SCL_log_stats = sup_con_crit(sup_con_feats, labels=sup_con_labels, curv=model[1].get_curvature())
                else:
                    sup_con_loss, SCL_log_stats = sup_con_crit(sup_con_feats, labels=sup_con_labels)

                # Total loss
                loss = (1 - args.sup_con_weight) * contrastive_loss + args.sup_con_weight * sup_con_loss
                #loss = contrastive_loss
                if loss.isnan():
                    print(f"Loss is NaN. con_loss is: {contrastive_loss}, sup_con_loss is: {sup_con_loss}")
                    #exit()
                #if contrastive_loss.isnan():
                #    print(contrastive_logits)

                # Train acc
                _, pred = contrastive_logits.max(1)
                acc = (pred == contrastive_labels).float().mean().item()
                train_acc_record.update(acc, pred.size(0))

                loss_record.update(loss.item(), class_labels.size(0))
                con_loss_record.update(contrastive_loss.item(), class_labels.size(0))
                sup_con_loss_record.update(sup_con_loss.item(), class_labels.size(0))
                step_log_dict["step/train/contrastive_loss"] = contrastive_loss.item()
                step_log_dict["step/train/sup_con_loss"] = sup_con_loss.item()
                step_log_dict["step/train/full_loss"] = loss.item()
                step_log_dict["step/train/acc"] = acc
                step_log_dict["step/train/embed_mean"] = output_log_stats[0][0]
                step_log_dict["step/train/embed_stddiv"] = output_log_stats[0][1]
                step_log_dict["step/train/embed_max"] = output_log_stats[0][2]
                step_log_dict["step/train/embed_min"] = output_log_stats[0][3]
                step_log_dict["debug/step/train/SCL_logits_mean"] = SCL_log_stats[0][0]
                step_log_dict["debug/step/train/SCL_logits_stddiv"] = SCL_log_stats[0][1]
                step_log_dict["debug/step/train/SCL_logits_max"] = SCL_log_stats[0][2]
                step_log_dict["debug/step/train/SCL_logits_min"] = SCL_log_stats[0][3]
                step_log_dict["debug/step/train/SCL_exp_logits_mean"] = SCL_log_stats[1][0]
                step_log_dict["debug/step/train/SCL_exp_logits_stddiv"] = SCL_log_stats[1][1]
                step_log_dict["debug/step/train/SCL_exp_logits_max"] = SCL_log_stats[1][2]
                step_log_dict["debug/step/train/SCL_exp_logits_min"] = SCL_log_stats[1][3]
                step_log_dict["debug/step/train/SCL_exp_logits_masked_mean"] = SCL_log_stats[2][0]
                step_log_dict["debug/step/train/SCL_exp_logits_masked_stddiv"] = SCL_log_stats[2][1]
                step_log_dict["debug/step/train/SCL_exp_logits_masked_max"] = SCL_log_stats[2][2]
                step_log_dict["debug/step/train/SCL_exp_logits_masked_min"] = SCL_log_stats[2][3]
                step_log_dict["debug/step/train/SCL_log_prob_mean"] = SCL_log_stats[3][0]
                step_log_dict["debug/step/train/SCL_log_prob_stddiv"] = SCL_log_stats[3][1]
                step_log_dict["debug/step/train/SCL_log_prob_max"] = SCL_log_stats[3][2]
                step_log_dict["debug/step/train/SCL_log_prob_min"] = SCL_log_stats[3][3]
                step_log_dict["debug/step/train/SCL_log_prob_masked_mean"] = SCL_log_stats[4][0]
                step_log_dict["debug/step/train/SCL_log_prob_masked_stddiv"] = SCL_log_stats[4][1]
                step_log_dict["debug/step/train/SCL_log_prob_masked_max"] = SCL_log_stats[4][2]
                step_log_dict["debug/step/train/SCL_log_prob_masked_min"] = SCL_log_stats[4][3]
                if args.hyperbolic:
                    if args.euclidean_clipping is not None:
                        step_log_dict["step/train/cliped_embed_mean"] = output_log_stats[2][0]
                        step_log_dict["step/train/cliped_embed_stddiv"] = output_log_stats[2][1]
                        step_log_dict["step/train/cliped_embed_max"] = output_log_stats[2][2]
                        step_log_dict["step/train/cliped_embed_min"] = output_log_stats[2][3]
                        step_log_dict["step/train/cliped_embed2_mean"] = output_log_stats[3][0]
                        step_log_dict["step/train/cliped_embed2_stddiv"] = output_log_stats[3][1]
                        step_log_dict["step/train/cliped_embed2_max"] = output_log_stats[3][2]
                        step_log_dict["step/train/cliped_embed2_min"] = output_log_stats[3][3] 
                    step_log_dict["step/train/curvature"] = model[1].get_curvature()
                    step_log_dict["step/train/proj_alpha"] = model[1].get_proj_alpha()
                    step_log_dict["step/train/hyp_embed_mean"] = output_log_stats[1][0]
                    step_log_dict["step/train/hyp_embed_stddiv"] = output_log_stats[1][1]
                    step_log_dict["step/train/hyp_embed_max"] = output_log_stats[1][2]
                    step_log_dict["step/train/hyp_embed_min"] = output_log_stats[1][3]
                optimizer.zero_grad()
                loss.backward()
                # Add gradient clipping or normalization here
                optimizer.step()
                wandb.log(step_log_dict)


        print('Train Epoch: {} Avg Loss: {:.4f} | Seen Class Acc: {:.4f} '.format(epoch+1, loss_record.avg,
                                                                                  train_acc_record.avg))
        epoch_log_dict = {"epoch": epoch+1, "epoch/train/loss": loss_record.avg, "epoch/train/acc": train_acc_record.avg,
                          "epoch/train/contrstive_loss": con_loss_record.avg, "epoch/train/sup_con_loss": sup_con_loss_record.avg,
                          "epoch/train/mean_learning_rate": get_mean_lr(optimizer), "epoch/train/learning_rate": scheduler.get_last_lr()[0]}
        if args.hyperbolic:
            print(f"Current curvature: {model[1].get_curvature()}")
            print(f"Current projection weight: {model[1].get_proj_alpha()}")
            epoch_log_dict["epoch/train/curvature"] = model[1].get_curvature()
            epoch_log_dict["epoch/train/proj_alpha"] = model[1].get_proj_alpha()

        if loss.isnan():
            break

        if args.kmeans and ((epoch+1) % args.kmeans_frequency == 0):
            with torch.no_grad():

                print('Testing on unlabelled examples in the training data...')
                all_acc, old_acc, new_acc = test_kmeans(model, unlabelled_train_loader,
                                                        epoch=epoch, save_name='Train ACC Unlabelled',
                                                        args=args)

                #print('Testing on disjoint test set...')
                #all_acc_test, old_acc_test, new_acc_test = test_kmeans(model, test_loader,
                #                                                    epoch=epoch, save_name='Test ACC',
                #                                                    args=args)
        #exit()
        # ----------------
        # LOG
        # ----------------
        args.writer.add_scalar('Loss', loss_record.avg, epoch)
        args.writer.add_scalar('Train Acc Labelled Data', train_acc_record.avg, epoch)
        args.writer.add_scalar('LR', get_mean_lr(optimizer), epoch)

        if args.kmeans and ((epoch+1) % args.kmeans_frequency == 0):
            print('Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc,
                                                                                new_acc))
            epoch_log_dict["epoch/kmeans/all_acc"] = all_acc
            epoch_log_dict["epoch/kmeans/old_acc"] = old_acc
            epoch_log_dict["epoch/kmeans/new_acc"] = new_acc
            #print('Test Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc_test, old_acc_test,
            #                                                                        new_acc_test))

        # Step schedule
        scheduler.step()

        if args.kmeans and ((epoch+1) % args.kmeans_frequency == 0):
            if old_acc > best_test_acc_lab:

                print('Best Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc,
                                                                                    new_acc))

                torch.save(model.state_dict(), args.model_path[:-3] + f'_best.pt')
                print("model saved to {}.".format(args.model_path[:-3] + f'_best.pt'))
                wandb.save(args.model_path[:-3] + f'_best.pt')

                best_test_acc_lab = old_acc
        #torch.save(model.state_dict(), args.model_path)
        args_copy = Namespace(**vars(args))
        args_copy.writer = None
        torch.save({
            "epoch": epoch+1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "arguments": args_copy,
            "wandb_run_id": wandb.run.id,
            "best_test_acc": best_test_acc_lab,
        }, args.model_path)
        print("model saved to {}.".format(args.model_path))

        wandb.log(epoch_log_dict)
        wandb.save(args.model_path)


def test_kmeans(model, test_loader,
                epoch, save_name,
                args):

    model.eval()

    all_feats = []
    targets = np.array([])
    mask = np.array([])

    print('Collating features...')
    # First extract all features
    for batch_idx, (images, label, _) in enumerate(tqdm(test_loader)):

        images = images.cuda()

        # Pass features through base model and then additional learnable transform (linear layer)
        feats, _ = model(images)

        feats = torch.nn.functional.normalize(feats, dim=-1)

        all_feats.append(feats.cpu().detach().numpy())
        targets = np.append(targets, label.cpu().numpy())
        mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes))
                                         else False for x in label]))

    # -----------------------
    # K-MEANS
    # -----------------------
    print('Fitting K-Means...')
    all_feats = np.concatenate(all_feats)
    if args.hyperbolic:
        #TODO: Investigate why K++ is failing to assign more than one center (Points too close to one another maybe?)
        kmeans = SemiSupKMeans(k=args.num_labeled_classes + args.num_unlabeled_classes, random_state=0, hyperbolic=True, curv=model[1].get_curvature(), init="random")
        kmeans.fit(all_feats)
        preds = kmeans.labels_.numpy()
    else:
        kmeans = KMeans(n_clusters=args.num_labeled_classes + args.num_unlabeled_classes, random_state=0).fit(all_feats)
        preds = kmeans.labels_
    print('Done!')

    # -----------------------
    # EVALUATE
    # -----------------------
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                    writer=args.writer)

    return all_acc, old_acc, new_acc


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
            description='cluster',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v1', 'v2'])

    parser.add_argument('--warmup_model_dir', type=str, default=None)
    parser.add_argument('--model_name', type=str, default='vit_dino', help='Format is {model_name}_{pretrain}')
    parser.add_argument('--dataset_name', type=str, default='scars', help='options: cifar10, cifar100, scars')
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--use_ssb_splits', type=str2bool, default=False)

    parser.add_argument('--grad_from_block', type=int, default=11)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--save_best_thresh', type=float, default=None)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--exp_root', type=str, default=exp_root)
    parser.add_argument('--exp_id', type=str, default=None)
    parser.add_argument('--transform', type=str, default='imagenet')
    parser.add_argument('--seed', default=1, type=int)

    parser.add_argument('--base_model', type=str, default='vit_dino')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--sup_con_weight', type=float, default=0.5)
    parser.add_argument('--n_views', default=2, type=int)
    parser.add_argument('--contrast_unlabel_only', type=str2bool, default=False)

    parser.add_argument('--wandb_mode', type=str, default="online")
    parser.add_argument('--epochs_warmup', default=2, type=int)
    parser.add_argument('--hyperbolic', type=str2bool, default=False)
    parser.add_argument('--kmeans', type=str2bool, default=False)
    parser.add_argument('--kmeans_frequency', type=int, default=20)
    parser.add_argument('--curvature', type=float, default=1.0)
    parser.add_argument('--euclidean_clipping', type=float, default=None)
    parser.add_argument('--freeze_curvature', type=str, default="false")
    parser.add_argument('--proj_alpha', type=float, default=1.7035**-1)
    parser.add_argument('--freeze_proj_alpha', type=str, default="false")
    parser.add_argument('--checkpoint_path', type=str)

    # ----------------------
    # INIT
    # ----------------------
    args = parser.parse_args()
    device = torch.device('cuda:0')
    args = get_class_splits(args)

    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)

    init_experiment(args, runner_name=['metric_learn_gcd'], exp_id=args.exp_id)
    print(f'Using evaluation function {args.eval_funcs[0]} to print results')

    #global DEBUG_DIR
    DEBUG_DIR = args.debug_dir

    # ----------------------
    # BASE MODEL
    # ----------------------
    if args.base_model == 'vit_dino':

        args.interpolation = 3
        args.crop_pct = 0.875
        #pretrain_path = dino_pretrain_path

        #model = vits.__dict__['vit_base']()
        model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')

        #state_dict = torch.load(pretrain_path, map_location='cpu')
        #model.load_state_dict(state_dict)

        if args.warmup_model_dir is not None:
            print(f'Loading weights from {args.warmup_model_dir}')
            model.load_state_dict(torch.load(args.warmup_model_dir, map_location='cpu'))

        # NOTE: Hardcoded image size as we do not finetune the entire ViT model
        args.image_size = 224
        args.feat_dim = 768
        args.num_mlp_layers = 3
        #args.mlp_out_dim = 65536
        args.mlp_out_dim = 768

        # ----------------------
        # HOW MUCH OF BASE MODEL TO FINETUNE
        # ----------------------
        for m in model.parameters():
            m.requires_grad = False

        # Only finetune layers from block 'args.grad_from_block' onwards
        for name, m in model.named_parameters():
            if 'block' in name:
                block_num = int(name.split('.')[1])
                if block_num >= args.grad_from_block:
                    m.requires_grad = True

    else:

        raise NotImplementedError

    # --------------------
    # CONTRASTIVE TRANSFORM
    # --------------------
    train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
    train_transform = ContrastiveLearningViewGenerator(base_transform=train_transform, n_views=args.n_views)

    # --------------------
    # DATASETS
    # --------------------
    train_dataset, test_dataset, unlabelled_train_examples_test, datasets = get_datasets(args.dataset_name,
                                                                                         train_transform,
                                                                                         test_transform,
                                                                                         args)


    # --------------------
    # SAMPLER
    # Sampler which balances labelled and unlabelled examples in each batch
    # --------------------
    label_len = len(train_dataset.labelled_dataset)
    unlabelled_len = len(train_dataset.unlabelled_dataset)
    sample_weights = [1 if i < label_len else label_len / unlabelled_len for i in range(len(train_dataset))]
    sample_weights = torch.DoubleTensor(sample_weights)
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(train_dataset))

    # --------------------
    # DATALOADERS
    # --------------------
    train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False,
                              sampler=sampler, drop_last=True)
    test_loader_unlabelled = DataLoader(unlabelled_train_examples_test, num_workers=args.num_workers,
                                        batch_size=args.batch_size, shuffle=False)
    test_loader_labelled = DataLoader(test_dataset, num_workers=args.num_workers,
                                      batch_size=args.batch_size, shuffle=False)

    # ----------------------
    # PROJECTION HEAD
    # ----------------------
    if args.hyperbolic:
        projection_head = vits.__dict__['Hyperbolic_DINOHead'](in_dim=args.feat_dim, out_dim=args.mlp_out_dim,
                                                               nlayers=args.num_mlp_layers, curv_init=args.curvature,
                                                               learn_curv=not args.freeze_curvature.lower() == "full",
                                                               alpha_init=args.proj_alpha,
                                                               learn_alpha=not args.freeze_proj_alpha.lower() == "full",
                                                               euclidean_clip_value=args.euclidean_clipping)
    else:
        projection_head = vits.__dict__['DINOHead'](in_dim=args.feat_dim, out_dim=args.mlp_out_dim,
                                                    nlayers=args.num_mlp_layers)
    model = torch.nn.Sequential(model, projection_head).to(device)
    #print(model[1].parameters)
    #exit()

    # ----------------------
    # OPTIMIZER AND SCHEDULER
    # ----------------------
    optimizer = SGD(vits.get_params_groups(model), lr=args.lr, momentum=args.momentum,
                    weight_decay=args.weight_decay)

    scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.lr * 1e-3,
        )

    # ----------------------
    # LOAD CHECKPOINT
    # ----------------------
    checkpoint = {}
    start_epoch = 0
    best_test_acc = 0
    if not args.checkpoint_path is None:
            checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
            if not "model_state_dict" in checkpoint.keys():
                model.load_state_dict(checkpoint)
            else:
                #checkpoint = torch.load(args.checkpoint_path)
                model.load_state_dict(checkpoint["model_state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                start_epoch = checkpoint["epoch"]
                best_test_acc = checkpoint["best_test_acc"]

    # ----------------------
    # INITIALIZE WANDB
    # ----------------------
    wandb.login()
    if checkpoint.get("wandb_run_id", None) is None or args.wandb_mode != "online":# or args.wandb_new_id:
        wandb.init(config = args,
                #dir = args.save_dir + '/wandb_logs',
                dir = "wandb_logs/",
                project = 'Hyperbolic_GCD',
                name = args.exp_id + '-' + str(args.seed),
                mode = args.wandb_mode)
    else:
        wandb.init(config = args,
                #dir = args.save_dir + '/wandb_logs',
                dir = "wandb_logs/",
                project = 'Hyperbolic_GCD',
                name = args.exp_id + '-' + str(args.seed),
                id = checkpoint["wandb_run_id"],
                resume = 'must',
                mode = args.wandb_mode)
    wandb.watch(model, log="all", log_graph=True, log_freq=100)

    # ----------------------
    # TRAIN
    # ----------------------
    if start_epoch < args.epochs:
        train(model, train_loader, test_loader_labelled, test_loader_unlabelled, args, optimizer, scheduler,
              best_test_acc=best_test_acc, start_epoch=start_epoch)
    #TODO: Make it load and test on best model if inner Kmeans is being used
    print('Testing on disjoint test set...')
    all_acc_test, old_acc_test, new_acc_test = test_kmeans(model, test_loader_labelled,
                                                        epoch=args.epochs, save_name='Test ACC',
                                                        args=args)
    print('Test Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc_test, old_acc_test,
                                                                                    new_acc_test))
    print("Finished training!")
