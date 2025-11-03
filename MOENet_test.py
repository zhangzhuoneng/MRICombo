import argparse
import os, sys
sys.path.append("..")
sys.path.append("../..")
import torch
import numpy as np
import torch.backends.cudnn as cudnn
import os.path as osp
from torch.utils.tensorboard import SummaryWriter
import logging
import pandas as pd
from models.OmniNet import omni_seg_cls
from MRI_dataset_seg import UnisegDataset,tr_seg_collate
from MRI_dataset_cls import UniclsDataset, tr_cls_collate,val_cls_collate
import random
import timeit
from loss_functions import omni_loss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader, WeightedRandomSampler
from utils import ChinaTimeFormatter, adjust_learning_rate, weight_base_init, WeightedRandomSamplerDDP
from utils import ChinaTimeFormatter, adjust_learning_all_rate, WeightedRandomSamplerDDP,new_dice,Hd_95
from utils import TEMPLATE,ORGAN_NAME,get_key_task
from monai.inferers import sliding_window_inference
from sklearn.metrics import accuracy_score
from torch.nn.parallel import DistributedDataParallel 
from sklearn.metrics import accuracy_score
from PCGrad import PCGrad
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, average_precision_score
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, average_precision_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="UniMRINet")
    parser.add_argument("--data_dir", type=str, default="/data/zzn/UniMRINet/dataset/")
    parser.add_argument("--excel_dir", type=str, default='csv/MRICombo_output_external_liver_tumor_merged_seg')
    # parser.add_argument("--val_seg_list", type=str, default="/data/zzn/UniMRINet/dataset/classification/cls_val_demo.txt")
    # parser.add_argument("--val_seg_list", type=str, default= "/data/zzn/UniMRINet/dataset/segmentation/seg_test_1.txt" )
    parser.add_argument("--val_seg_list", type=str, default= "/data/zzn/UniMRINet/dataset/segmentation/external/liver_seg_test_1.txt" )
    # parser.add_argument("--val_cls_list", type=str, default='/data/zzn/UniMRINet/dataset/dataset_orginal/txt/12NPC/test_new.txt')
    parser.add_argument("--val_cls_list", type=str, default='/data/zzn/UniMRINet/dataset/classification/cls_test_new.txt')
    parser.add_argument('--backbone_name', default='MRICombo', help='backbone unet,swinunetr,DeepFusionUniMRINet')
    # parser.add_argument("--reload_path", type=str, default="./snapshots/omni_seg_cls_MRICombo_lb_4experts_0.05_400_0.5cls_weight/omni_cls_e395.pth")
    parser.add_argument("--reload_path", type=str, default='../snapshots/omni_seg_cls_2_encoder_5_26_CROSS/omni_cls_e395.pth')
    parser.add_argument('--trans_encoding', default='word_embedding', 
                        help='the type of encoding: rand_embedding or word_embedding')
    parser.add_argument('--word_embedding', default='../pretrained_weights/sequence_cancer_encoding.pth', 
                        help='The path of word embedding')
    parser.add_argument("--reload_from_checkpoint", type=str2bool, default=True)
    parser.add_argument("--save_path", type=str, default='outputs/MRICombo_output') 
    parser.add_argument("--input_size", type=str, default='96,96,96')
    parser.add_argument("--in_channels", type=int, default=1)
    
    parser.add_argument('--roi_x', default=96, type=int, help='roi size in x direction')
    parser.add_argument('--roi_y', default=96, type=int, help='roi size in y direction')
    parser.add_argument('--roi_z', default=96, type=int, help='roi size in z direction')
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seg_classes", type=int, default=27)
    parser.add_argument("--cls_classes", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--random_scale", default=True)
    parser.add_argument("--random_mirror",  default=True)
    return parser
    
def plot_roc_pr_curves(all_labels, all_probs, dataset_names, save_dir):
    """
    绘制AUROC和AUPRC曲线图
    """
    # 创建图片保存目录
    plot_dir = os.path.join(save_dir, 'plots')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir, exist_ok=True)
    
    # 设置图形参数
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['savefig.facecolor'] = 'white'
    
    # 青色系颜色列表，与您提供的图片风格一致
    colors = ['#00CED1', '#20B2AA', '#48D1CC', '#40E0D0', '#00FFFF']  # 青色系
    
    # 为每个数据集创建单独的图
    for idx, dataset_name in enumerate(dataset_names):
        labels = np.array(all_labels[dataset_name])
        probs = np.array(all_probs[dataset_name])
        
        if len(labels) == 0:
            continue
            
        # 创建子图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.patch.set_facecolor('white')
        fig.suptitle(f'{dataset_name} Dataset - ROC and PR Curves', fontsize=14, fontweight='bold', color='black')
        
        if dataset_name == "NPC":  # 四分类数据集
            labels_bin = label_binarize(labels, classes=[0, 1, 2, 3])
            
            # 绘制ROC曲线
            for i in range(4):
                fpr, tpr, _ = roc_curve(labels_bin[:, i], probs[:, i])
                roc_auc = roc_auc_score(labels_bin[:, i], probs[:, i])
                ax1.plot(fpr, tpr, label=f'Class {i} (AUROC = {roc_auc:.3f})', 
                        color=colors[i % len(colors)], linewidth=2.5)
            
            ax1.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)
            ax1.set_xlim([0.0, 1.0])
            ax1.set_ylim([0.0, 1.05])
            ax1.set_xlabel('False Positive Rate', fontsize=11, color='black')
            ax1.set_ylabel('True Positive Rate', fontsize=11, color='black')
            ax1.set_title('ROC Curves', fontsize=12, color='black')
            ax1.legend(loc="lower right", fontsize=10, framealpha=0.9)
            ax1.grid(True, alpha=0.2, color='gray')
            ax1.set_facecolor('white')
            ax1.tick_params(colors='black')
            
            # 绘制PR曲线
            for i in range(4):
                precision, recall, _ = precision_recall_curve(labels_bin[:, i], probs[:, i])
                pr_auc = average_precision_score(labels_bin[:, i], probs[:, i])
                ax2.plot(recall, precision, label=f'Class {i} (AUPRC = {pr_auc:.3f})', 
                        color=colors[i % len(colors)], linewidth=2.5)
            
            ax2.set_xlim([0.0, 1.0])
            ax2.set_ylim([0.0, 1.05])
            ax2.set_xlabel('Recall', fontsize=11, color='black')
            ax2.set_ylabel('Precision', fontsize=11, color='black')
            ax2.set_title('PR Curves', fontsize=12, color='black')
            ax2.legend(loc="lower left", fontsize=10, framealpha=0.9)
            ax2.grid(True, alpha=0.2, color='gray')
            ax2.set_facecolor('white')
            ax2.tick_params(colors='black')
            
        else:  # 二分类数据集
            # 绘制ROC曲线
            fpr, tpr, _ = roc_curve(labels, probs[:, 1])
            roc_auc = roc_auc_score(labels, probs[:, 1])
            ax1.plot(fpr, tpr, color=colors[0], 
                    linewidth=3, label=f'Our Model (AUROC = {roc_auc:.3f})')
            ax1.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)
            ax1.set_xlim([0.0, 1.0])
            ax1.set_ylim([0.0, 1.05])
            ax1.set_xlabel('False Positive Rate', fontsize=11, color='black')
            ax1.set_ylabel('True Positive Rate', fontsize=11, color='black')
            ax1.set_title(f'{dataset_name}', fontsize=12, color='black')
            ax1.legend(loc="lower right", fontsize=11, framealpha=0.9)
            ax1.grid(True, alpha=0.2, color='gray')
            ax1.set_facecolor('white')
            ax1.tick_params(colors='black')
            
            # 绘制PR曲线  
            precision, recall, _ = precision_recall_curve(labels, probs[:, 1])
            pr_auc = average_precision_score(labels, probs[:, 1])
            ax2.plot(recall, precision, color=colors[0], 
                    linewidth=3, label=f'Our Model (AUPRC = {pr_auc:.3f})')
            ax2.set_xlim([0.0, 1.0])
            ax2.set_ylim([0.0, 1.05])
            ax2.set_xlabel('Recall', fontsize=11, color='black')
            ax2.set_ylabel('Precision', fontsize=11, color='black')
            ax2.set_title(f'{dataset_name}', fontsize=12, color='black')
            ax2.legend(loc="lower left", fontsize=11, framealpha=0.9)
            ax2.grid(True, alpha=0.2, color='gray')
            ax2.set_facecolor('white')
            ax2.tick_params(colors='black')
        
        plt.tight_layout()
        
        # 保存每个数据集的图片
        dataset_path = os.path.join(plot_dir, f'{dataset_name}_roc_pr_curves.png')
        plt.savefig(dataset_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
    
    # 创建汇总图
    plot_summary_curves(all_labels, all_probs, dataset_names, plot_dir)
    
    print(f'AUROC and AUPRC curves saved to: {plot_dir}')

def plot_summary_curves(all_labels, all_probs, dataset_names, plot_dir):
    """
    绘制所有数据集的汇总ROC和PR曲线
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.patch.set_facecolor('white')
    
    # 青色系颜色列表
    colors = ['#00CED1', '#20B2AA', '#48D1CC', '#40E0D0', '#00FFFF']
    
    # 绘制所有数据集的ROC曲线在一个图上
    for idx, dataset_name in enumerate(dataset_names):
        labels = np.array(all_labels[dataset_name])
        probs = np.array(all_probs[dataset_name])
        
        if len(labels) == 0:
            continue
            
        if dataset_name == "NPC":  # 四分类数据集
            labels_bin = label_binarize(labels, classes=[0, 1, 2, 3])
            auc_scores = []
            for i in range(4):
                auc_scores.append(roc_auc_score(labels_bin[:, i], probs[:, i]))
            auc = np.mean(auc_scores)
            
            # 计算宏平均ROC
            all_fpr = np.unique(np.concatenate([roc_curve(labels_bin[:, i], probs[:, i])[0] 
                                              for i in range(4)]))
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(4):
                fpr, tpr, _ = roc_curve(labels_bin[:, i], probs[:, i])
                mean_tpr += np.interp(all_fpr, fpr, tpr)
            mean_tpr /= 4
            
            ax1.plot(all_fpr, mean_tpr, color=colors[idx % len(colors)], 
                    linewidth=3, label=f'{dataset_name} (AUROC = {auc:.3f})')
                    
            auprc_scores = []
            for i in range(4):
                auprc_scores.append(average_precision_score(labels_bin[:, i], probs[:, i]))
            auprc = np.mean(auprc_scores)
            
            # 计算宏平均PR
            all_recall = np.linspace(0, 1, 100)
            mean_precision = np.zeros_like(all_recall)
            for i in range(4):
                precision, recall, _ = precision_recall_curve(labels_bin[:, i], probs[:, i])
                mean_precision += np.interp(all_recall, recall[::-1], precision[::-1])
            mean_precision /= 4
            
            ax2.plot(all_recall, mean_precision, color=colors[idx % len(colors)], 
                    linewidth=3, label=f'{dataset_name} (AUPRC = {auprc:.3f})')
            
        else:  # 二分类数据集
            fpr, tpr, _ = roc_curve(labels, probs[:, 1])
            roc_auc = roc_auc_score(labels, probs[:, 1])
            ax1.plot(fpr, tpr, color=colors[idx % len(colors)], 
                    linewidth=3, label=f'{dataset_name} (AUROC = {roc_auc:.3f})')
            
            precision, recall, _ = precision_recall_curve(labels, probs[:, 1])
            pr_auc = average_precision_score(labels, probs[:, 1])
            ax2.plot(recall, precision, color=colors[idx % len(colors)], 
                    linewidth=3, label=f'{dataset_name} (AUPRC = {pr_auc:.3f})')
    
    # 设置ROC图
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate', fontsize=12, color='black')
    ax1.set_ylabel('True Positive Rate', fontsize=12, color='black')
    ax1.set_title('ROC Curves - All Datasets', fontsize=14, fontweight='bold', color='black')
    ax1.legend(loc="lower right", fontsize=11, framealpha=0.9)
    ax1.grid(True, alpha=0.2, color='gray')
    ax1.set_facecolor('white')
    ax1.tick_params(colors='black')
    
    # 设置PR图
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall', fontsize=12, color='black')
    ax2.set_ylabel('Precision', fontsize=12, color='black')
    ax2.set_title('Precision-Recall Curves - All Datasets', fontsize=14, fontweight='bold', color='black')
    ax2.legend(loc="lower left", fontsize=11, framealpha=0.9)
    ax2.grid(True, alpha=0.2, color='gray')
    ax2.set_facecolor('white')
    ax2.tick_params(colors='black')
    
    plt.tight_layout()
    
    # 保存汇总图
    summary_path = os.path.join(plot_dir, 'summary_roc_pr_curves.png')
    plt.savefig(summary_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

def validate(args, input_size, model, Val_cls_Loader, Val_seg_Loader, device, num_classes):
    val_cls_loss = []
    all_labels = {}
    all_preds = {}
    all_probs = {}
    cls_accuracy = []
    cls_auroc = []  # 改为AUROC
    cls_auprc = []  # 添加AUPRC列表
    dataset_names = ["cen", "NPC", "LLD", "Bra","Bre"]
    # dataset_names = ["LLD"]
    # 初始化字典，存储每个数据集的标签和预测
    for dataset_name in dataset_names:
        all_labels[dataset_name] = []
        all_preds[dataset_name] = []
        all_probs[dataset_name] = []
    model.eval()
    with torch.no_grad():
        for index, batch in enumerate(Val_cls_Loader):
            # 加载批次数据
            cls_x1 = torch.from_numpy(batch['x1']).cuda()
            cls_x2 = torch.from_numpy(batch['x2']).cuda()
            cls_x3 = torch.from_numpy(batch['x3']).cuda()
            cls_x4 = torch.from_numpy(batch['x4']).cuda()
            cls_x5 = torch.from_numpy(batch['x5']).cuda()
            cls_x6 = torch.from_numpy(batch['x6']).cuda()
            cls_x7 = torch.from_numpy(batch['x7']).cuda()
            cls_x8 = torch.from_numpy(batch['x8']).cuda()
            # cls_x7 = torch.zeros_like(cls_x1).cuda()  # 假设形状与 cls_x1 相同
            # cls_x8 = torch.zeros_like(cls_x1).cuda()  # 假设形状与 cls_x1 相同
            # x7 = torch.from_numpy(batch['x7']).cuda()
            # x8 = torch.from_numpy(batch['x8']).cuda()
            sequence_cls_code = torch.from_numpy(batch['sequence_code']).cuda()
            cls_region_ids = torch.from_numpy(batch['region_ids']).cuda()
            labels = torch.from_numpy(batch['label']).cuda()
            labels = labels.long()
            volumeName = batch['name']
            
            # cls_preds = model(cls_x1,cls_x2,cls_x3,cls_x4,cls_x5,cls_x6, cls_x7,cls_x8,sequence_cls_code, 'cls',volumeName)
            cls_inputs = (cls_x1, cls_x2, cls_x3, cls_x4, cls_x5, cls_x6, cls_x7, cls_x8)
            seg_preds, cls_preds = model(
                seg_inputs=None,
                cls_inputs=cls_inputs,
                seg_sequence_code= None,
                cls_sequence_code=sequence_cls_code,
                names=volumeName,
                seg_region_ids=None,
                cls_region_ids=cls_region_ids
            )
            
            ""
            N = labels.shape[0]
            total_loss = 0
            # # print( volumeName,N)
            # for i in range(N):
            #     # print
            #     dataset_name = volumeName[i][:3]  # 获取当前样本的数据集名称 
            #     sample_pred = cls_preds[i]  # 添加批次维度
            #     sample_label = labels[i].unsqueeze(0)   # 添加批次维度
            #     ce_loss = loss_cls(sample_pred, sample_label)
            #     # print(i,dataset_name,sample_label)
            #     total_loss = total_loss + ce_loss
                
            # avg_loss = total_loss / labels.shape[0]
            # val_cls_loss.append(float(avg_loss))
            preds = torch.argmax(cls_preds[0], dim=1)
            probs = torch.softmax(cls_preds[0], dim=1)
            # print(volumeName, labels,preds)
            ""
        
            # ce_loss= loss_cls(cls_preds,labels)
            # val_cls_loss.append(float(ce_loss))
            # 获取预测的类别标签和概率
            # preds = torch.argmax(cls_preds, dim=1)
            # probs = torch.softmax(cls_preds, dim=1)
            
            # 从卷名中提取数据集前缀
            dataset_prefix = volumeName[0][:3]
            
            # 处理numpy数组的情况
            if isinstance(dataset_prefix, np.ndarray):
                dataset_prefix = dataset_prefix.item()
            
            # 为每个数据集存储标签和预测
            if dataset_prefix in all_labels:
                all_labels[dataset_prefix].extend(labels.cpu().numpy().tolist())  # 转换为列表
                all_preds[dataset_prefix].extend(preds.cpu().numpy().tolist())    # 转换为列表
                all_probs[dataset_prefix].extend(probs.cpu().numpy().tolist())    # 转换为列表
            else:
                print(f"警告：数据集前缀 '{dataset_prefix}' 无法识别")

    # 计算每个数据集的指标
    cls_loss = np.mean(val_cls_loss)
    dataset_aucs = {}  # 用于存储每个数据集的AUC值


    # if (args.local_rank == 0):
    for dataset_name in dataset_names:
        accuracy = accuracy_score(all_labels[dataset_name], all_preds[dataset_name])
        cls_accuracy.append(accuracy)
        print(f'{dataset_name} ACC: {accuracy:.4f}')
    # 计算每个数据集的AUC
    for dataset_name in dataset_names:
        labels = np.array(all_labels[dataset_name])
        probs = np.array(all_probs[dataset_name])
        
        if dataset_name == "NPC":  # 四分类数据集
            labels_bin = label_binarize(labels, classes=[0, 1, 2, 3])
            auc_scores = []
            for i in range(4):
                # print(i,roc_auc_score(labels_bin[:, i], probs[:, i]))
                auc_scores.append(roc_auc_score(labels_bin[:, i], probs[:, i]))
            auc = np.mean(auc_scores)
        else:  # 二分类数据集
            auc = roc_auc_score(labels, probs[:, 1])  # 假设第二列是正类概率

        print(f'{dataset_name} AUROC: {auc:.4f}')
        
        cls_auroc.append(auc)
    
    # 计算每个数据集的AUPRC
    for dataset_name in dataset_names:
        labels = np.array(all_labels[dataset_name])
        probs = np.array(all_probs[dataset_name])
        
        if dataset_name == "NPC":  # 四分类数据集
            labels_bin = label_binarize(labels, classes=[0, 1, 2, 3])
            auprc_scores = []
            for i in range(4):
                auprc_scores.append(average_precision_score(labels_bin[:, i], probs[:, i]))
            auprc = np.mean(auprc_scores)
        else:  # 二分类数据集
            auprc = average_precision_score(labels, probs[:, 1])  # 假设第二列是正类概率

        print(f'{dataset_name} AUPRC: {auprc:.4f}')
        
        cls_auprc.append(auprc)
    
    # 绘制AUROC和AUPRC图片
    plot_roc_pr_curves(all_labels, all_probs, dataset_names, args.excel_dir)
    
    # 计算并打印平均分类指标
    avg_auc = np.mean(cls_auroc)
    avg_auprc = np.mean(cls_auprc)
    avg_accuracy = np.mean(cls_accuracy)
    print(f'\nAverage classification metrics across all datasets:')
    print(f'Average ACC: {avg_accuracy:.4f}')
    print(f'Average AUROC: {avg_auc:.4f}')
    print(f'Average AUPRC: {avg_auprc:.4f}')
    print('=' * 50)
    
    def predictor_wrapper(inputs,sequence_seg_code,seg_region_ids):
                # 将 inputs 拆分为多个输入
                x1 = inputs[:, 0:1, ...]  # 输入 1
                x2 = inputs[:, 1:2, ...] # 输入 2
                x3 = inputs[:, 2:3, ...]  # 输入 3
                x4 = inputs[:, 3:4, ...]  # 输入 4
                x5 = inputs[:, 4:5, ...]  # 输入 2
                x6 = inputs[:, 5:6, ...]  # 输入 3
                x7 = inputs[:, 6:7, ...]  # 输入 4
                x8 = inputs[:, 7:8, ...]  # 输入 4
                seg_inputs = (x1, x2, x3, x4, x5, x6, x7, x8)
                seg_preds, cls_preds = model(
                        seg_inputs=seg_inputs,
                        cls_inputs=None,
                        seg_sequence_code=sequence_seg_code,
                        cls_sequence_code=None,
                        names=None,
                        seg_region_ids=seg_region_ids,
                        cls_region_ids=None
                    )
                # 调用模型
                # 
                return seg_preds
        

    # model.eval()
    dice_list = {}
    results = []
    # 初始化总 Dice 和计数
    total_dice = 0.0
    total_count = 0
    for key in TEMPLATE.keys():
        dice_list[key] = np.zeros((2,args.seg_classes)) # 1st row for dice, 2nd row for count
    with torch.no_grad():
        for index, batch in enumerate(Val_seg_Loader):
            sta = timeit.default_timer()
            # x1,x2,x3,x4,x5,x6,x7,x8, name, label, mask_code, affine= batch
            
            # # print(x1.shape)
            # mask_code = mask_code.to(device)

            # x1 = x1.to(device)
            # x2 = x2.to(device)
            # x3 = x3.to(device)
            # x4 = x4.to(device)
            # x5 = x5.to(device)
            # x6 = x6.to(device)
            # x7 = x7.to(device)
            # x8 = x8.to(device)
            x1,x2,x3,x4,x5,x6,x7,x8, name, label, mask_code, affine,seg_region_ids, task_ids = batch

            # 设备
            seg_region_ids = seg_region_ids.to(device)
            mask_code = mask_code.to(device)
            x1=x1.to(device); x2=x2.to(device); x3=x3.to(device); x4=x4.to(device)
            x5=x5.to(device); x6=x6.to(device); x7=x7.to(device); x8=x8.to(device)
            # volumeName = batch['name']
            # #  # 将多个输入拼接在通道维度上
            # inputs = torch.cat([x1, x2, x3, x4,x5,x6,x7, x8], dim=1) 
            # # print(name)
            # pred_sigmoid = sliding_window_inference(
            #     inputs = inputs,
            #     roi_size=(args.roi_x,args.roi_y,args.roi_z),
            #     sw_batch_size=1,
            #     predictor=lambda inputs: predictor_wrapper(inputs,mask_code,name),
            #     overlap=0.5,
            #     mode="constant",
            #     # progress=True,
            # )
            inputs = torch.cat([x1, x2, x3, x4,x5,x6,x7, x8], dim=1) 
                    # print(name)
            pred_sigmoid = sliding_window_inference(
                inputs = inputs,
                roi_size=(args.roi_x,args.roi_y,args.roi_z),
                sw_batch_size=1,
                predictor=lambda inputs: predictor_wrapper(inputs,mask_code,seg_region_ids),
                overlap=0.5,
                mode="constant",
                # progress=True,
            )

            cur_output = torch.sigmoid(pred_sigmoid)
            pred_binary  = np.asarray(np.around(cur_output.cpu()), dtype=np.uint8)
            label_binary = label.numpy().astype(np.uint8)
            template_key = get_key_task(name[0]) 
            # print(name[0])
            # print( template_key)
            organ_list = TEMPLATE[template_key]
            # print( organ_list)
            end = timeit.default_timer()
            for organ in organ_list:
            
                # 检查Brain标签通道是否为空
                if np.sum(label_binary[:, organ - 1, :, :, :]) == 0:
                    continue
                val_dice = new_dice(pred_binary[:,organ-1,:,:,:], label_binary[:,organ-1,:,:,:])
                hd95_distance = Hd_95(pred_binary[:,organ-1,:,:,:], label_binary[:,organ-1,:,:,:])
                print('%s: %s dice = %.3f hd_95 = %.2f seconds=%.2f '%(name[0],ORGAN_NAME[organ - 1],val_dice, hd95_distance,end - sta))
                dice_list[template_key][0][organ-1] += val_dice.item()
                dice_list[template_key][1][organ-1] += 1    
                # 累加总 Dice 和计数
                
                results.append([name[0], ORGAN_NAME[organ - 1], val_dice, hd95_distance])
                df = pd.DataFrame(results, columns=['Name', 'Organ', 'Dice', 'HD_95'])
                df.to_csv(os.path.join(args.excel_dir, f'omni_seg_cls_our_96.csv'), index=False)

    # if (args.local_rank == 0):
    for key in TEMPLATE.keys():
        organ_list = TEMPLATE[key]
        content = 'Task%s|'%(key)
        
        for organ in organ_list:
            dice = dice_list[key][0][organ-1] / dice_list[key][1][organ-1]
            total_dice += dice.item()
            total_count += 1
            content += '%s: %.3f, '%(ORGAN_NAME[organ-1], dice)
        # print('val_dice in 27 posistion: {}'.format(content))
        print(content)
    # 输出 27 个器官的平均 Dice
    if total_count > 0:
        avg_dice = total_dice / total_count
    
        print("all average Dice: %.3f" % avg_dice)
            
    return all_labels, all_preds, all_probs, cls_auprc
def main():
    """Create the model and start the evaluate."""
    parser = get_arguments()
    # print(parser)
    args = parser.parse_args()
    d, h, w = map(int, args.input_size.split(','))
    input_size = (d, h, w)

    # model = omni_seg_cls(
    # img_size=(args.roi_x,args.roi_y,args.roi_z),
    # in_channels=1,
    # out_channels=args.seg_classes,
    # backbone = args.backbone_name,
    # cls_classes=args.cls_classes)
    model = omni_seg_cls(
    img_size=(args.roi_x,args.roi_y,args.roi_z),
    seg_in_channels=args.in_channels,
    cls_in_channels=args.in_channels,
    out_channels=args.seg_classes,
    backbone = args.backbone_name,
    cls_classes=args.cls_classes)
    
    
    device = torch.device('cuda:0')
    model.to(device)
    
    if not os.path.exists(args.excel_dir):
        os.makedirs(args.excel_dir, exist_ok=True)  # 创建目录，如果目录不存在的话
    
    # load checkpoint...
    # if args.reload_from_checkpoint:
    #     if os.path.exists(args.reload_path):
    #         checkpoint = torch.load(args.reload_path)
    #         model.load_state_dict(checkpoint['model'])
    #         # model.module.load_state_dict(checkpoint['model'])
    #         optimizer.load_state_dict(checkpoint['optimizer'])
    #         start_epoch = checkpoint['epoch']
    #         print('loading from checkpoint: {}'.format(args.reload_path))
            
    #     else:
    #         print('File not exists in the reload path: {}'.format(args.reload_path))

    # if args.reload_from_checkpoint:
    #     if os.path.exists(args.reload_path):
    #         print('loading from checkpoint: {}'.format(args.reload_path))
    #         model.load_state_dict(torch.load(args.reload_path, map_location=device))
    #     else:
    #         print('File not exists in the reload path: {}'.format(args.reload_path))

    if args.reload_from_checkpoint:
        print('loading from checkpoint: {}'.format(args.reload_path))
        state_dict=torch.load(args.reload_path, map_location=device,weights_only=True)
        # create new OrderedDict that does not contain `module.`
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
          
            # name = 'module.' + k   # add `module.`
            name = k[7:]  # - `module.`
            new_state_dict[name] = v
        # load params
        model.load_state_dict(new_state_dict)
    # if args.reload_from_checkpoint:
    #     print('loading from checkpoint: {}'.format(args.reload_path))
    #     state_dict = torch.load(args.reload_path, map_location=device,weights_only=True)

    #     # 如果权重名称以 "module." 开头，移除前缀
    #     from collections import OrderedDict
    #     new_state_dict = OrderedDict()
    #     for k, v in state_dict['model'].items():
    #     # for k, v in state_dict.items():
    #         if k.startswith("module."):
    #             new_state_dict[k[7:]] = v  # 移除 "module." 前缀
    #         else:
    #             new_state_dict[k] = v
    #     model.load_state_dict(new_state_dict)
    # if args.reload_from_checkpoint:
    #     print('loading from checkpoint: {}'.format(args.reload_path))
    #     state_dict = torch.load(args.reload_path, map_location=device)

    #     # 如果权重名称以 "module." 开头，移除前缀
    #     from collections import OrderedDict
    #     new_state_dict = OrderedDict()
    #     for k, v in state_dict.items():
    #     # for k, v in state_dict.items():
    #         if k.startswith("module."):
    #             new_state_dict[k[7:]] = v  # 移除 "module." 前缀
    #         else:
    #             new_state_dict[k] = v

    # # 加载权重
    # model.load_state_dict(new_state_dict)
        
    # mask_code = [1,1,0,0,0,0]
    mask_code = [1,1,1,1,0,0,0,0]
    val_cls_dataset = UniclsDataset(args.data_dir, args.val_cls_list, split="val",code=mask_code,
                                crop_size=(args.roi_x,args.roi_y,args.roi_z))

    val_cls_loader = DataLoader(val_cls_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=False,
                                shuffle=False,
                                collate_fn=tr_cls_collate
                                )
    val_seg_dataset = UnisegDataset(args.data_dir, args.val_seg_list, split="val",
                                crop_size=(args.roi_x,args.roi_y,args.roi_z), scale=args.random_scale, mirror=args.random_mirror)
    val_seg_loader = DataLoader(val_seg_dataset,
                                batch_size=1,
                                num_workers=args.num_workers,
                                drop_last=False,
                                shuffle=False,
                                pin_memory=True,
                                # sampler = weighted_seg_sampler,
                                # collate_fn=tr_seg_collate
                                )


    print('validate ...')
    validate(args, input_size, model, val_cls_loader,val_seg_loader,device,args.seg_classes)

    end = timeit.default_timer()
    # print(end - start, 'seconds')


if __name__ == '__main__':
    main()
# CUDA_VISIBLE_DEVICES=1 python test.py

