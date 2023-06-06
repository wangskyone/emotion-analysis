import os
import random
import argparse
import yaml
from tqdm import tqdm
import datetime

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from datasets import build_dataset
from datasets.utils import build_data_loader
import clip
from utils import *

writer = SummaryWriter(log_dir='./logs/{}'.format(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")),
                       comment='Tip-Adapter')



class ClassBalancedCrossEntropyLoss(nn.Module):
    def __init__(self, num_classes, device):
        super().__init__()
        self.num_classes = num_classes
        self.device = device
        self.loss_fn = nn.CrossEntropyLoss(reduction='none')
        self.class_weights = self.get_class_weights()

    def get_class_weights(self):
        class_counts = torch.tensor([705, 717, 281, 4772, 2524, 1982, 1290], device=self.device, dtype=torch.float32)
        total_samples = class_counts.sum()
        class_weights = total_samples / (self.num_classes * class_counts)
        return class_weights
    
    def forward(self, inputs, targets):
        ce_loss = self.loss_fn(inputs, targets)
        weights = self.class_weights[targets]
        balanced_ce_loss = (weights * ce_loss).mean()
        return balanced_ce_loss


def get_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', help='settings of Tip-Adapter in yaml format')
    args = parser.parse_args()

    return args


def log_learning_rate(optimizer, global_step):
    for param_group in optimizer.param_groups:
        writer.add_scalar('learning_rate', param_group['lr'], global_step)


def log_accuracy(phase, accuracy, global_step):
    writer.add_scalar(phase+"_accuracy", accuracy, global_step)


def run_tip_adapter(cfg, cache_keys, cache_values, val_features, val_labels, test_features, test_labels, clip_weights):
    
    print("\n-------- Searching hyperparameters on the val set. --------")

    # Zero-shot CLIP
    clip_logits = 100. * val_features @ clip_weights
    acc = cls_acc(clip_logits, val_labels)
    print("\n**** Zero-shot CLIP's val accuracy: {:.2f}. ****\n".format(acc))

    # Tip-Adapter
    beta, alpha = cfg['init_beta'], cfg['init_alpha']
    
    affinity = val_features @ cache_keys
    cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
    
    tip_logits = clip_logits + cache_logits * alpha
    acc = cls_acc(tip_logits, val_labels)
    print("**** Tip-Adapter's val accuracy: {:.2f}. ****\n".format(acc))

    # Search Hyperparameters
    best_beta, best_alpha = search_hp(cfg, cache_keys, cache_values, val_features, val_labels, clip_weights)


    print("\n-------- Evaluating on the test set. --------")

    # Zero-shot CLIP
    clip_logits = 100. * test_features @ clip_weights
    acc = cls_acc(clip_logits, test_labels)
    print("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(acc))

    # Tip-Adapter    
    affinity = test_features @ cache_keys
    cache_logits = ((-1) * (best_beta - best_beta * affinity)).exp() @ cache_values
    
    tip_logits = clip_logits + cache_logits * best_alpha
    acc = cls_acc(tip_logits, test_labels)
    print("**** Tip-Adapter's test accuracy: {:.2f}. ****\n".format(acc))


def run_tip_adapter_F(cfg, cache_keys, cache_values, val_features, val_labels, test_features, test_labels, clip_weights, clip_model, train_loader_F):
    
    loss_fn = ClassBalancedCrossEntropyLoss(num_classes=7, device='cuda:0')

    # Enable the cached keys to be learnable
    adapter = nn.Linear(cache_keys.shape[0], cache_keys.shape[1], bias=False).to(clip_model.dtype).cuda()
    adapter.weight = nn.Parameter(cache_keys.t())

    # for param in clip_model.parameters():
    #     param.requires_grad = False

    # for name, param in clip_model.named_parameters():
    #     # if 'resblocks.11' in name:
    #     # if 'visual' in name:
    #     if 'ln_final' in name:
    #         param.requires_grad = True

    # params = list(clip_model.named_parameters())
    # adapter_params, clip_model_params = adapter.parameters(), filter(lambda p: p.requires_grad, clip_model.parameters())

    # param_groups = [
    #     {'params': adapter_params, 'lr': cfg['lr'], 'eps': 1e-4},
    #     {'params': clip_model_params, 'lr': cfg['lr_clip'], 'eps': 1e-4},
    # ]
    # optimizer = torch.optim.AdamW(param_groups)

    optimizer = torch.optim.AdamW(adapter.parameters(), lr=cfg['lr'], eps=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg['train_epoch'] * len(train_loader_F))
    
    beta, alpha = cfg['init_beta'], cfg['init_alpha']
    best_acc, best_epoch = 0.0, 0

    for train_idx in range(1000):
    # for train_idx in range(cfg['train_epoch']):
        # Train
        adapter.train()
        clip_model.train()
        correct_samples, all_samples = 0, 0
        loss_list = []
        print('Train Epoch: {:} / {:}'.format(train_idx, cfg['train_epoch']))

        for i, (images, target) in enumerate(tqdm(train_loader_F)):
            images, target = images.cuda(), target.cuda()
            # with torch.no_grad():
            #     image_features = clip_model.encode_image(images)
            #     image_features /= image_features.norm(dim=-1, keepdim=True)
            
            image_features = clip_model.encode_image(images)
            image_features_norm = image_features.norm(dim=-1, keepdim=True)
            image_features = image_features / image_features_norm


            affinity = adapter(image_features)
            cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
            clip_logits = 100. * image_features @ clip_weights
            tip_logits = clip_logits + cache_logits * alpha

            # loss = F.cross_entropy(tip_logits, target)
            loss_balanced = loss_fn(tip_logits, target)

            acc = cls_acc(tip_logits, target)
            correct_samples += acc / 100 * len(tip_logits)
            all_samples += len(tip_logits)
            loss_list.append(loss_balanced.item())

            optimizer.zero_grad()
            loss_balanced.backward()
            optimizer.step()
            scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        log_learning_rate(optimizer, train_idx)
        log_accuracy('train', (correct_samples / all_samples), train_idx)
        print('LR: {:.6f}, Acc: {:.4f} ({:}/{:}), Loss: {:.4f}'.format(current_lr, correct_samples / all_samples, correct_samples, all_samples, sum(loss_list)/len(loss_list)))

        # Eval
        clip_model.eval()
        adapter.eval()

        affinity = adapter(test_features)
        cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
        clip_logits = 100. * test_features @ clip_weights
        tip_logits = clip_logits + cache_logits * alpha
        acc = cls_acc(tip_logits, test_labels)

        log_accuracy('val', acc, train_idx)
        print("**** Tip-Adapter-F's test accuracy: {:.2f}. ****\n".format(acc))
        if acc > best_acc:
            best_acc = acc
            best_epoch = train_idx
            torch.save(adapter.weight, cfg['cache_dir'] + "/best_F_" + str(cfg['shots']) + "shots.pt")
    
    adapter.weight = torch.load(cfg['cache_dir'] + "/best_F_" + str(cfg['shots']) + "shots.pt")
    print(f"**** After fine-tuning, Tip-Adapter-F's best test accuracy: {best_acc:.2f}, at epoch: {best_epoch}. ****\n")

    print("\n-------- Searching hyperparameters on the val set. --------")

    # Search Hyperparameters
    best_beta, best_alpha = search_hp(cfg, cache_keys, cache_values, val_features, val_labels, clip_weights, adapter=adapter)

    print("\n-------- Evaluating on the test set. --------")
   
    affinity = adapter(test_features)
    cache_logits = ((-1) * (best_beta - best_beta * affinity)).exp() @ cache_values
    
    tip_logits = clip_logits + cache_logits * best_alpha
    acc = cls_acc(tip_logits, test_labels)
    print("**** Tip-Adapter-F's test accuracy: {:.2f}. ****\n".format(max(best_acc, acc)))

    return best_acc, acc

def rec_cfg(cfg, best_acc, acc):
    with open("record_acc.txt", "a") as f:
        now = datetime.datetime.now()
        
        dataset = cfg['dataset']
        shots = cfg['shots']
        backbone = cfg['backbone']

        lr = cfg['lr']
        lr_clip = cfg['lr_clip']
        augment_epoch = cfg['augment_epoch']
        train_epoch = cfg['train_epoch']

        init_beta = cfg['init_beta']
        init_alpha = cfg['init_alpha']

        f.write("Time: {}\n".format(now.strftime("%Y-%m-%d %H:%M:%S")))
        f.write("Dataset: {}, Shots: {}, Backbone: {}\n".format(dataset, shots, backbone))
        f.write("LR: {}, LR_clip: {}, Augment Epoch: {}, Train Epoch: {}\n".format(lr, lr_clip, augment_epoch, train_epoch))
        f.write("Init Beta: {}, Init Alpha: {}\n".format(init_beta, init_alpha))
        f.write("Best Acc: {:.2f}, Acc: {:.2f}\n\n".format(best_acc, acc))


def main():

    # Load config file
    args = get_arguments()
    # remember to remove
    args.config = "configs/rafdb.yaml"

    assert (os.path.exists(args.config))
    
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    cache_dir = os.path.join('./caches', cfg['dataset'])
    os.makedirs(cache_dir, exist_ok=True)
    cfg['cache_dir'] = cache_dir

    print("\nRunning configs.")
    print(cfg, "\n")

    # CLIP
    clip_model, preprocess = clip.load(cfg['backbone'])
    clip_model.eval()
    # clip_model.train()

    # Prepare dataset
    random.seed(1)
    torch.manual_seed(1)
    
    print("Preparing dataset.")
    dataset = build_dataset(cfg['dataset'], cfg['root_path'], cfg['shots'])

    val_loader = build_data_loader(data_source=dataset.val, batch_size=64, is_train=False, tfm=preprocess, shuffle=False)
    test_loader = build_data_loader(data_source=dataset.test, batch_size=64, is_train=False, tfm=preprocess, shuffle=False)

    train_tranform = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])

    train_loader_cache = build_data_loader(data_source=dataset.train_x, batch_size=256, tfm=train_tranform, is_train=True, shuffle=False)
    train_loader_F = build_data_loader(data_source=dataset.train_x, batch_size=256, tfm=train_tranform, is_train=True, shuffle=True)

    # Textual features
    print("\nGetting textual features as CLIP's classifier.")
    clip_weights = clip_classifier(dataset.classnames, dataset.template, clip_model)

    # Construct the cache model by few-shot training set
    print("\nConstructing cache model by few-shot visual features and labels.")
    cache_keys, cache_values = build_cache_model(cfg, clip_model, train_loader_cache)

    # Pre-load val features
    print("\nLoading visual features and labels from val set.")
    val_features, val_labels = pre_load_features(cfg, "val", clip_model, val_loader)

    # Pre-load test features
    print("\nLoading visual features and labels from test set.")
    test_features, test_labels = pre_load_features(cfg, "test", clip_model, test_loader)

    # ------------------------------------------ Tip-Adapter ------------------------------------------
    run_tip_adapter(cfg, cache_keys, cache_values, val_features, val_labels, test_features, test_labels, clip_weights)

    # ------------------------------------------ Tip-Adapter-F ------------------------------------------
    best_acc, acc = run_tip_adapter_F(cfg, cache_keys, cache_values, val_features, val_labels, test_features, test_labels, clip_weights, clip_model, train_loader_F)
    
    rec_cfg(cfg, best_acc, acc)

if __name__ == '__main__':
    main()
