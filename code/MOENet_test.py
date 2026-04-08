import argparse
import os, sys
sys.path.append("..")
sys.path.append("../..")
import torch
import numpy as np
import os.path as osp
import pandas as pd
from network.OmniNet import omni_seg_cls
from MOE_dataset_seg import UnisegDataset
from MOE_dataset_cls import UniclsDataset, tr_cls_collate
import timeit
from torch.utils.data import DataLoader
from utils import new_dice, Hd_95
from utils import TEMPLATE, ORGAN_NAME, get_key_task, assd_score
from monai.inferers import sliding_window_inference
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import nibabel as nib
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
    parser = argparse.ArgumentParser(description="MRICombo")
    parser.add_argument("--data_dir", type=str, default="/data/zzn/UniMRINet/code/MRICombo/dataset/", help="Path to dataset root directory")
    parser.add_argument("--excel_dir", type=str, default='../csv/MRICombo_mae')
    parser.add_argument("--val_seg_list", type=str, default="../dataset/segmentation/seg_test.txt", help="Path to test segmentation list")
    parser.add_argument("--val_cls_list", type=str, default='../dataset/classification/cls_test.txt', help="Path to test classification list")
    parser.add_argument('--backbone_name', default='MRICombo', help='backbone unet,swinunetr,DeepFusionUniMRINet')
    parser.add_argument("--reload_path", type=str, default='../snapshots/Best_MRICombo.pth', help="Path to trained model checkpoint")
    parser.add_argument("--reload_from_checkpoint", type=str2bool, default=True)
    parser.add_argument("--save_path", type=str, default='../outputs/MRICombo_mae')
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


def _modal_inputs_from_np_batch(batch, device):
    return tuple(torch.from_numpy(batch[f"x{i}"]).to(device) for i in range(1, 9))


def _compute_auc_auprc(dataset_name, labels, probs):
    if dataset_name == "NPC":
        labels_bin = label_binarize(labels, classes=[0, 1, 2, 3])
        auc = np.mean([roc_auc_score(labels_bin[:, i], probs[:, i]) for i in range(4)])
        auprc = np.mean([average_precision_score(labels_bin[:, i], probs[:, i]) for i in range(4)])
    else:
        auc = roc_auc_score(labels, probs[:, 1])
        auprc = average_precision_score(labels, probs[:, 1])
    return float(auc), float(auprc)


def plot_roc_pr_curves(all_labels, all_probs, dataset_names, save_dir):
    """
    AUROC and AUPRC
    """

    plot_dir = os.path.join(save_dir, 'plots')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir, exist_ok=True)


    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['savefig.facecolor'] = 'white'


    colors = ['#00CED1', '#20B2AA', '#48D1CC', '#40E0D0', '#00FFFF']


    for idx, dataset_name in enumerate(dataset_names):
        labels = np.array(all_labels[dataset_name])
        probs = np.array(all_probs[dataset_name])

        if len(labels) == 0:
            continue

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.patch.set_facecolor('white')
        fig.suptitle(f'{dataset_name} Dataset - ROC and PR Curves', fontsize=14, fontweight='bold', color='black')

        if dataset_name == "NPC":
            labels_bin = label_binarize(labels, classes=[0, 1, 2, 3])


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

        else:

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


        dataset_path = os.path.join(plot_dir, f'{dataset_name}_roc_pr_curves.png')
        plt.savefig(dataset_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()


    plot_summary_curves(all_labels, all_probs, dataset_names, plot_dir)

    print(f'AUROC and AUPRC curves saved to: {plot_dir}')

def plot_summary_curves(all_labels, all_probs, dataset_names, plot_dir):
    """

    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.patch.set_facecolor('white')


    colors = ['#00CED1', '#20B2AA', '#48D1CC', '#40E0D0', '#00FFFF']


    for idx, dataset_name in enumerate(dataset_names):
        labels = np.array(all_labels[dataset_name])
        probs = np.array(all_probs[dataset_name])

        if len(labels) == 0:
            continue

        if dataset_name == "NPC":
            labels_bin = label_binarize(labels, classes=[0, 1, 2, 3])
            auc_scores = []
            for i in range(4):
                auc_scores.append(roc_auc_score(labels_bin[:, i], probs[:, i]))
            auc = np.mean(auc_scores)


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


            all_recall = np.linspace(0, 1, 100)
            mean_precision = np.zeros_like(all_recall)
            for i in range(4):
                precision, recall, _ = precision_recall_curve(labels_bin[:, i], probs[:, i])
                mean_precision += np.interp(all_recall, recall[::-1], precision[::-1])
            mean_precision /= 4

            ax2.plot(all_recall, mean_precision, color=colors[idx % len(colors)],
                    linewidth=3, label=f'{dataset_name} (AUPRC = {auprc:.3f})')

        else:
            fpr, tpr, _ = roc_curve(labels, probs[:, 1])
            roc_auc = roc_auc_score(labels, probs[:, 1])
            ax1.plot(fpr, tpr, color=colors[idx % len(colors)],
                    linewidth=3, label=f'{dataset_name} (AUROC = {roc_auc:.3f})')

            precision, recall, _ = precision_recall_curve(labels, probs[:, 1])
            pr_auc = average_precision_score(labels, probs[:, 1])
            ax2.plot(recall, precision, color=colors[idx % len(colors)],
                    linewidth=3, label=f'{dataset_name} (AUPRC = {pr_auc:.3f})')


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

    summary_path = os.path.join(plot_dir, 'summary_roc_pr_curves.png')
    plt.savefig(summary_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()


def save_nii(args, seg_pred, seg_label, name, affine):
    """Save prediction and label as NIfTI files."""
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)
    case_name = name[0] if isinstance(name[0], str) else str(name[0])
    seg_label_save_p = osp.join(args.save_path, f'{case_name}_label.nii.gz')
    seg_pred_save_p = osp.join(args.save_path, f'{case_name}_pred.nii.gz')
    nib.save(seg_label, seg_label_save_p)
    nib.save(seg_pred, seg_pred_save_p)


def save_detailed_results_to_excel(all_sample_names, all_labels, all_preds, all_probs,
                                   dataset_names, save_dir):
    """
    Save per-sample classification results to Excel.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    for dataset_name in dataset_names:
        if len(all_sample_names[dataset_name]) == 0:
            print(f"Skip {dataset_name}: no sample data")
            continue
        sample_names = all_sample_names[dataset_name]
        true_labels = all_labels[dataset_name]
        pred_labels = all_preds[dataset_name]
        probs = np.array(all_probs[dataset_name])
        data = {
            'Sample_Name': sample_names,
            'True_Label': true_labels,
            'Predicted_Label': pred_labels,
        }
        num_classes = probs.shape[1]
        for i in range(num_classes):
            data[f'Prob_Class_{i}'] = probs[:, i]
        data['Correct'] = [int(t == p) for t, p in zip(true_labels, pred_labels)]
        df = pd.DataFrame(data)
        excel_path = os.path.join(save_dir, f'{dataset_name}_detailed_results.xlsx')
        df.to_excel(excel_path, index=False, engine='openpyxl')
        print(f"{dataset_name} detailed results saved to: {excel_path}")
        print(f"   total samples: {len(df)}")

    all_data = []
    for dataset_name in dataset_names:
        if len(all_sample_names[dataset_name]) == 0:
            continue
        sample_names = all_sample_names[dataset_name]
        true_labels = all_labels[dataset_name]
        pred_labels = all_preds[dataset_name]
        probs = np.array(all_probs[dataset_name])
        for i, sample_name in enumerate(sample_names):
            row = {
                'Dataset': dataset_name,
                'Sample_Name': sample_name,
                'True_Label': true_labels[i],
                'Predicted_Label': pred_labels[i],
                'Correct': int(true_labels[i] == pred_labels[i])
            }
            for j in range(probs.shape[1]):
                row[f'Prob_Class_{j}'] = probs[i, j]
            all_data.append(row)
    if all_data:
        summary_df = pd.DataFrame(all_data)
        summary_path = os.path.join(save_dir, 'all_datasets_summary.xlsx')
        summary_df.to_excel(summary_path, index=False, engine='openpyxl')
        print(f"\nAll-dataset summary saved to: {summary_path}")
        print(f"   total samples: {len(summary_df)}")


def validate(args, input_size, model, Val_cls_Loader, Val_seg_Loader, device, num_classes):
    dataset_names = ["cen", "NPC", "LLD", "Bra", "Bre"]
    all_labels = {k: [] for k in dataset_names}
    all_preds = {k: [] for k in dataset_names}
    all_probs = {k: [] for k in dataset_names}
    all_sample_names = {k: [] for k in dataset_names}
    cls_accuracy = []
    cls_auroc = []
    cls_auprc = []
    model.eval()
    with torch.no_grad():
        for index, batch in enumerate(Val_cls_Loader):

            cls_inputs = _modal_inputs_from_np_batch(batch, device)
            sequence_cls_code = torch.from_numpy(batch['sequence_code']).to(device)
            cls_region_ids = torch.from_numpy(batch['region_ids']).to(device)
            labels = torch.from_numpy(batch['label']).to(device)
            labels = labels.long()
            volumeName = batch['name']

            seg_preds, cls_preds = model(
                seg_inputs=None,
                cls_inputs=cls_inputs,
                seg_sequence_code= None,
                cls_sequence_code=sequence_cls_code,
                names=volumeName,
                seg_region_ids=None,
                cls_region_ids=cls_region_ids
            )

            preds = torch.argmax(cls_preds[0], dim=1)
            probs = torch.softmax(cls_preds[0], dim=1)

            dataset_prefix = volumeName[0][:3]

            if isinstance(dataset_prefix, np.ndarray):
                dataset_prefix = dataset_prefix.item()


            if dataset_prefix in all_labels:
                all_labels[dataset_prefix].extend(labels.cpu().numpy().tolist())
                all_preds[dataset_prefix].extend(preds.cpu().numpy().tolist())
                all_probs[dataset_prefix].extend(probs.cpu().numpy().tolist())
                for i in range(labels.shape[0]):
                    sn = volumeName[i] if isinstance(volumeName[i], str) else str(volumeName[i])
                    all_sample_names[dataset_prefix].append(sn)
            else:
                print(f"error：dataset prefix '{dataset_prefix}' not regonization")


    for dataset_name in dataset_names:
        accuracy = accuracy_score(all_labels[dataset_name], all_preds[dataset_name])
        cls_accuracy.append(accuracy)
        print(f'{dataset_name} ACC: {accuracy:.4f}')

    for dataset_name in dataset_names:
        labels = np.array(all_labels[dataset_name])
        probs = np.array(all_probs[dataset_name])
        auc, auprc = _compute_auc_auprc(dataset_name, labels, probs)
        print(f'{dataset_name} AUROC: {auc:.4f}')
        cls_auroc.append(auc)
        print(f'{dataset_name} AUPRC: {auprc:.4f}')
        cls_auprc.append(auprc)


    plot_roc_pr_curves(all_labels, all_probs, dataset_names, args.excel_dir)


    avg_auc = np.mean(cls_auroc)
    avg_auprc = np.mean(cls_auprc)
    avg_accuracy = np.mean(cls_accuracy)
    print(f'\nAverage classification metrics across all datasets:')
    print(f'Average ACC: {avg_accuracy:.4f}')
    print(f'Average AUROC: {avg_auc:.4f}')
    print(f'Average AUPRC: {avg_auprc:.4f}')
    print('=' * 50)


    print('\nSaving detailed classification results to Excel...')
    save_detailed_results_to_excel(all_sample_names, all_labels, all_preds, all_probs,
                                   dataset_names, args.excel_dir)

    def predictor_wrapper(inputs,sequence_seg_code,seg_region_ids):

                x1 = inputs[:, 0:1, ...]
                x2 = inputs[:, 1:2, ...]
                x3 = inputs[:, 2:3, ...]
                x4 = inputs[:, 3:4, ...]
                x5 = inputs[:, 4:5, ...]
                x6 = inputs[:, 5:6, ...]
                x7 = inputs[:, 6:7, ...]
                x8 = inputs[:, 7:8, ...]
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

                return seg_preds



    dice_list = {}
    results = []

    total_dice = 0.0
    total_count = 0
    for key in TEMPLATE.keys():
        dice_list[key] = np.zeros((2,args.seg_classes))
    with torch.no_grad():
        for index, batch in enumerate(Val_seg_Loader):
            sta = timeit.default_timer()
            x1,x2,x3,x4,x5,x6,x7,x8, name, label, mask_code, affine,seg_region_ids, task_ids = batch

            seg_region_ids = seg_region_ids.to(device)
            mask_code = mask_code.to(device)
            x1=x1.to(device); x2=x2.to(device); x3=x3.to(device); x4=x4.to(device)
            x5=x5.to(device); x6=x6.to(device); x7=x7.to(device); x8=x8.to(device)

            inputs = torch.cat([x1, x2, x3, x4,x5,x6,x7, x8], dim=1)

            pred_sigmoid = sliding_window_inference(
                inputs = inputs,
                roi_size=(args.roi_x,args.roi_y,args.roi_z),
                sw_batch_size=1,
                predictor=lambda inputs: predictor_wrapper(inputs,mask_code,seg_region_ids),
                overlap=0.5,
                mode="constant",

            )

            cur_output = torch.sigmoid(pred_sigmoid)
            pred_binary  = np.asarray(np.around(cur_output.cpu()), dtype=np.uint8)
            label_binary = label.numpy().astype(np.uint8)
            template_key = get_key_task(name[0])
            organ_list = TEMPLATE[template_key]


            if getattr(args, 'save_predictions', True):
                try:
                    idx = slice(organ_list[0] - 1, organ_list[-1])
                    last_labels = label_binary[:, idx, :, :, :]
                    last_preds = pred_binary[:, idx, :, :, :]
                    combined_labels = np.zeros_like(last_labels[0, 0, :, :, :])
                    combined_preds = np.zeros_like(last_preds[0, 0, :, :, :])
                    for i in range(last_labels.shape[1]):
                        combined_labels = np.where(last_labels[:, i] == 1, i + 1, combined_labels)
                        combined_preds = np.where(last_preds[:, i] == 1, i + 1, combined_preds)
                    label_transposed = np.squeeze(combined_labels, axis=0).transpose(1, 2, 0)
                    pred_transposed = np.squeeze(combined_preds, axis=0).transpose(1, 2, 0)
                    aff = np.asarray(affine, dtype=np.float32)
                    if aff.ndim == 3:
                        aff = aff[0]
                    seg_pred = nib.Nifti1Image(pred_transposed.astype(np.uint8), aff)
                    seg_label = nib.Nifti1Image(label_transposed.astype(np.uint8), aff)
                    save_nii(args, seg_pred, seg_label, name, affine)
                except Exception as e:
                    print(f"Warning: failed to save prediction for {name[0]}: {e}")

            end = timeit.default_timer()
            for organ in organ_list:


                if np.sum(label_binary[:, organ - 1, :, :, :]) == 0:
                    continue
                val_dice = new_dice(pred_binary[:,organ-1,:,:,:], label_binary[:,organ-1,:,:,:])
                hd95_distance = Hd_95(pred_binary[:,organ-1,:,:,:], label_binary[:,organ-1,:,:,:])
                asd_distance = assd_score(pred_binary[:,organ-1,:,:,:], label_binary[:,organ-1,:,:,:])
                print('%s: %s dice = %.3f hd_95 = %.2f ASD = %.2f seconds=%.2f '%(name[0],ORGAN_NAME[organ - 1],val_dice, hd95_distance, asd_distance, end - sta))
                dice_list[template_key][0][organ-1] += val_dice.item()
                dice_list[template_key][1][organ-1] += 1


                results.append([name[0], ORGAN_NAME[organ - 1], val_dice, hd95_distance, asd_distance])


    for key in TEMPLATE.keys():
        organ_list = TEMPLATE[key]
        content = 'Task%s|'%(key)

        for organ in organ_list:
            dice = dice_list[key][0][organ-1] / dice_list[key][1][organ-1]
            total_dice += dice.item()
            total_count += 1
            content += '%s: %.3f, '%(ORGAN_NAME[organ-1], dice)

        print(content)

    if total_count > 0:
        avg_dice = total_dice / total_count

        print("all average Dice: %.3f" % avg_dice)
    if results:
        pd.DataFrame(results, columns=['Name', 'Organ', 'Dice', 'HD_95', 'ASD']).to_csv(
            os.path.join(args.excel_dir, 'omni_seg_cls_our_96.csv'),
            index=False,
        )

    return all_labels, all_preds, all_probs, cls_auprc
def main():
    """Create the model and start the evaluate."""
    parser = get_arguments()

    args = parser.parse_args()
    d, h, w = map(int, args.input_size.split(','))
    input_size = (d, h, w)


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
        os.makedirs(args.excel_dir, exist_ok=True)
        
    if args.reload_from_checkpoint:
        print('loading from checkpoint: {}'.format(args.reload_path))
        state_dict=torch.load(args.reload_path, map_location=device,weights_only=True)

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():


            name = k[7:]
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)

    mask_code = [1,1,1,1,1,1,1,1]
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
                                )
    print('validate ...')

    validate(args, input_size, model, val_cls_loader,val_seg_loader,device,args.seg_classes)


if __name__ == '__main__':
    main()

