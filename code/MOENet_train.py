






































































































































































































































































































































































































































































































































































































































































































































import argparse
import os, sys
sys.path.append("..")
sys.path.append("../../")
sys.path.append(".")
import torch
import torch.nn.functional as F
import numpy as np
import torch.backends.cudnn as cudnn
import os.path as osp
from torch.utils.tensorboard import SummaryWriter
import logging
from network.OmniNet import omni_seg_cls
from MOE_dataset_seg import UnisegDataset,tr_seg_collate
from MOE_dataset_cls import UniclsDataset, tr_cls_collate
import random
import timeit
from loss_functions import omni_loss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader, WeightedRandomSampler
from utils import ChinaTimeFormatter, weight_base_init_new, WeightedRandomSamplerDDP
from utils import adjust_learning_all_rate, new_dice, Hd_95
from utils import TEMPLATE,ORGAN_NAME,get_key_task
from monai.inferers import sliding_window_inference
from torch.nn.parallel import DistributedDataParallel 
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, accuracy_score



start = timeit.default_timer()

def get_arguments():
    parser = argparse.ArgumentParser(description="UniMRINet")
    parser.add_argument('--dist', dest='dist', type=bool, default=False,
                        help='distributed training or not')
    parser.add_argument("--data_dir", type=str, default="/data/zzn/UniMRINet/code/MRICombo/dataset/", help="Path to dataset root directory")
    parser.add_argument("--log_dir", type=str, default='../log/log_omni_MRICombo_0.25')
    parser.add_argument("--tensorboard_log_name", type=str, default='/omni_seg_cls_MRICombo')
    parser.add_argument("--snapshot_dir", type=str, default='../snapshots/omni_seg_cls_MRICombo_0.25')
    parser.add_argument("--cls_weight", type=float, default=0.5)
    parser.add_argument('--backbone_name', default='MRICombo', help='backbone unet,swinunetr') 
    parser.add_argument("--train_seg_list", type=str, default="../dataset/segmentation/seg_val.txt", help="Path to training segmentation list")
    parser.add_argument("--val_seg_list", type=str, default="../dataset/segmentation/seg_val.txt", help="Path to validation segmentation list")
    parser.add_argument("--train_cls_list", type=str, default="../dataset/classification/cls_train.txt", help="Path to training classification list")
    parser.add_argument("--val_cls_list", type=str, default="../dataset/classification/cls_val.txt", help="Path to validation classification list")
    parser.add_argument("--reload_path", type=str, default='../snapshots/', help="Path to pretrained model checkpoint")
    parser.add_argument("--reload_from_checkpoint", default=False)
    parser.add_argument('--log_name', default='unet', help='The path resume from checkpoint')
    parser.add_argument('--roi_x', default=96, type=int, help='roi size in x direction')
    parser.add_argument('--roi_y', default=96, type=int, help='roi size in y direction')
    parser.add_argument('--roi_z', default=96, type=int, help='roi size in z direction')
    parser.add_argument("--batch_size", type=int, default = 4)
    parser.add_argument("--num_gpus", type=int, default = 2)
    parser.add_argument('--local-rank', type=int, default = 0)
    parser.add_argument("--num_epochs", type=int, default=400)
    parser.add_argument("--itrs_each_epoch", type=int, default=1)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--start_epoch", type=int, default=-1)
    parser.add_argument("--val_pred_every", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=0.00003)
    parser.add_argument("--lb_coeff", type=float, default=0.05)
    parser.add_argument("--in_channels", type=int, default=1)
    parser.add_argument("--seg_classes", type=int, default=27)
    parser.add_argument("--cls_classes", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--power", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.00005)
    parser.add_argument("--ignore_label", type=int, default=255)
    parser.add_argument("--is_training", action="store_true")
    parser.add_argument("--random_mirror",  default=True)
    parser.add_argument("--random_scale", default=True)
    parser.add_argument("--random_seed", type=int, default=1234)
    parser.add_argument("--gpu", type=str, default='0')

    parser.add_argument('--use_mae', type=bool, default=True,
                        help='enable MAE self-supervised learning')
    parser.add_argument('--mae_initial_weight', type=float, default=1.0,
                        help='MAE initial weight (epoch 0)')
    parser.add_argument('--mae_warmup_epochs', type=int, default=100,
                        help='MAE warmup epochs (weight decays from initial to 0)')
    parser.add_argument('--mae_mask_ratio', type=float, default=0.25,
                        help='MAE masking ratio (0.5 = mask 50% voxels)')
    
    parser.add_argument('--sup_initial_weight', type=float, default=0.01,
                        help='supervised loss weight at epoch 0 (ramps to 1.0)')
    parser.add_argument('--sup_warmup_epochs', type=int, default=100,
                        help='epochs to ramp supervised loss weight to 1.0')
    return parser


def init_randon(seed):
    torch.manual_seed(seed)        
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed) 
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    cudnn.benchmark = True        
    cudnn.deterministic = True


def _modal_inputs_from_np_batch(batch, device):
    """Convert x1..x8 numpy arrays to cuda tensors and return tuple."""
    return tuple(torch.from_numpy(batch[f'x{i}']).to(device) for i in range(1, 9))

def main():
    """Create the model and start the training."""
    
    parser = get_arguments()
    args=parser.parse_args()

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)
    
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

  
    formatter = ChinaTimeFormatter('[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%Y-%m-%d %H:%M')

  
    logging.basicConfig(
        filename=args.log_dir + "/log.txt",
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%Y-%m-%d %H:%M'  
    )

   
    logger = logging.getLogger()
    logger.handlers[0].setFormatter(formatter)  
    logger.addHandler(logging.StreamHandler(sys.stdout))  
    logger.handlers[1].setFormatter(formatter)  

 
    logging.info(str(args))
    init_randon(args.random_seed)


    if args.dist:


        args.local_rank = int(os.environ['LOCAL_RANK'])  
        torch.cuda.set_device(args.local_rank)
        world_size = int(os.environ['WORLD_SIZE'])
        torch.distributed.init_process_group(backend='nccl')
        torch.distributed.barrier()
    else:

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.cuda.set_device(0)
    model = omni_seg_cls(
    img_size=(args.roi_x,args.roi_y,args.roi_z),
    seg_in_channels=args.in_channels,
    cls_in_channels=args.in_channels,
    out_channels=args.seg_classes,
    backbone = args.backbone_name,
    cls_classes=args.cls_classes)

   
    model.train()
  
    device = torch.device('cuda:{}'.format(args.local_rank))
    model.to(device)
 
    if args.dist:

        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DistributedDataParallel(model, device_ids=[args.local_rank],find_unused_parameters=True)
      
    else:
        model = torch.nn.DataParallel(model).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    if args.reload_from_checkpoint:
        if os.path.exists(args.reload_path):
            checkpoint = torch.load(args.reload_path, map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint['model'])

            args.start_epoch = checkpoint['epoch']
            print('loading from checkpoint: {}'.format(args.reload_path))
            
        else:
            print('File not exists in the reload path: {}'.format(args.reload_path))


    loss_seg_DICE = omni_loss.DiceLoss(num_classes=args.seg_classes).to(device)
    loss_seg_CE = omni_loss.CELoss(num_classes=args.seg_classes).to(device)


    loss_cls = CrossEntropyLoss(ignore_index=-1).to(device)

    if args.use_mae and args.mae_initial_weight > 0:
        logging.info("=" * 60)
        logging.info("🚀 Self-Supervised Learning: Input-Layer MAE")
        logging.info(f"   MAE initial weight: {args.mae_initial_weight}")
        logging.info(f"   MAE warmup epochs: {args.mae_warmup_epochs} (linear decay to 0)")
        logging.info(f"   Mask ratio: {args.mae_mask_ratio}")
        logging.info(f"   Mask strategy: Random voxel-level (per sequence)")
        logging.info(f"   Architecture: Mask inputs → Extract features → Reconstruct")
        logging.info("   Input-level MAE mask is applied per sequence")
        logging.info("   Reconstruct full fused features from masked inputs")
        logging.info("=" * 60)



    train_seg_dataset = UnisegDataset(args.data_dir, args.train_seg_list, split="train",
                                crop_size=(args.roi_x,args.roi_y,args.roi_z), scale=args.random_scale, mirror=args.random_mirror)
    seg_sample_weight = weight_base_init_new(train_seg_dataset,'seg')

    if args.dist:

        weighted_seg_sampler = WeightedRandomSamplerDDP(
                                        data_set = train_seg_dataset,
                                        weights = seg_sample_weight,
                                        num_replicas = world_size,
                                        rank = args.local_rank,
                                        num_samples=len(seg_sample_weight),
                                        replacement=True)
    else:

        weighted_seg_sampler = WeightedRandomSampler(
            seg_sample_weight, num_samples=len(seg_sample_weight), replacement=True) 

    train_seg_loader = DataLoader(train_seg_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=False,
                                shuffle=False,
                                pin_memory=True,
                                sampler = weighted_seg_sampler,
                                collate_fn=tr_seg_collate)
    val_seg_dataset = UnisegDataset(args.data_dir, args.val_seg_list, split="val",
                                crop_size=(args.roi_x,args.roi_y,args.roi_z), scale=args.random_scale, mirror=args.random_mirror)
    val_seg_loader = DataLoader(val_seg_dataset,
                                batch_size=1,
                                num_workers=args.num_workers,
                                drop_last=False,
                                shuffle=False,
                                pin_memory=True,


                                )
    

    train_cls_dataset = UniclsDataset(args.data_dir, args.train_cls_list, split="train",
                                crop_size=(args.roi_x,args.roi_y,args.roi_z), scale=args.random_scale, mirror=args.random_mirror)
    cls_sample_weight = weight_base_init_new(train_cls_dataset,'cls')
    if args.dist:
         weighted_cls_sampler = WeightedRandomSamplerDDP(
                                        data_set = train_cls_dataset,
                                        weights = cls_sample_weight,
                                        num_replicas = world_size,
                                        rank = args.local_rank,
                                        num_samples=len(cls_sample_weight),
                                        replacement=True)
    else:
        weighted_cls_sampler = WeightedRandomSampler(cls_sample_weight, num_samples=len(cls_sample_weight), replacement=True)

    train_cls_loader = DataLoader(train_cls_dataset,
                                       batch_size=args.batch_size,
                                       num_workers=args.num_workers,
                                       drop_last=False,
                                       shuffle=False,
                                       pin_memory=True,
                                       sampler = weighted_cls_sampler,
                                       collate_fn=tr_cls_collate)

    val_cls_dataset = UniclsDataset(args.data_dir, args.val_cls_list, split="val",
                                crop_size=(args.roi_x,args.roi_y,args.roi_z), scale=args.random_scale, mirror=args.random_mirror)

    val_cls_loader = DataLoader(val_cls_dataset,
                                batch_size=1,
                                num_workers=args.num_workers,
                                drop_last=False,
                                shuffle=False, 
                                collate_fn=tr_cls_collate
                                )

    writer = SummaryWriter('../tensorboard'+args.tensorboard_log_name)
    

    all_tr_seg_loss = []
    all_va_seg_loss = []
    all_tr_cls_loss = []
    all_va_cls_loss = []
    best_loss = np.inf
    best_dice = 0.7
    for epoch in range(args.start_epoch+1,args.num_epochs):


        if args.dist:
            weighted_seg_sampler.set_epoch(epoch)
            weighted_cls_sampler.set_epoch(epoch)
        epoch_seg_loss = []
        epoch_cls_loss = []
        epoch_ssl_seg_loss = []
        epoch_ssl_cls_loss = []
        start_epoch_time = timeit.default_timer()
        lr=adjust_learning_all_rate(optimizer, epoch, args.learning_rate, args.num_epochs, args.power)
        for (seg_batch, cls_batch) in zip(train_seg_loader, train_cls_loader):

            seg_inputs = _modal_inputs_from_np_batch(seg_batch, device)
            sequence_seg_code = torch.from_numpy(seg_batch['sequence_code']).to(device)
            seg_labels = torch.from_numpy(seg_batch['label']).to(device)
            seg_volumeName = seg_batch['name']
            

            seg_region_ids = torch.from_numpy(seg_batch['region_ids']).to(device)
            sequence_seg_code_run = sequence_seg_code
            seg_region_ids_run = seg_region_ids
            seg_volumeName_run = seg_volumeName


            if epoch >= 250:
                cls_inputs = _modal_inputs_from_np_batch(cls_batch, device)
                sequence_cls_code = torch.from_numpy(cls_batch['sequence_code']).to(device)
                cls_labels = torch.from_numpy(cls_batch['label']).to(device)
                cls_volumeName = cls_batch['name']
                

                cls_region_ids = torch.from_numpy(cls_batch['region_ids']).to(device)
                sequence_cls_code_run = sequence_cls_code
                cls_region_ids_run = cls_region_ids
                cls_volumeName_run = cls_volumeName
                



                if epoch < args.mae_warmup_epochs:

                    current_mae_weight = args.mae_initial_weight * (1.0 - epoch / args.mae_warmup_epochs)
                    use_masked_encoder = True
                else:

                    current_mae_weight = 0.0
                    use_masked_encoder = False
                
                mae_enabled = args.use_mae and current_mae_weight > 0
                

                if epoch < args.sup_warmup_epochs:
                    current_sup_weight = args.sup_initial_weight + (1.0 - args.sup_initial_weight) * (epoch / args.sup_warmup_epochs)
                else:
                    current_sup_weight = 1.0
                try:
                    if mae_enabled:
                        seg_preds, cls_preds, mae_recon, mae_target, input_masks = model(
                            seg_inputs=seg_inputs,
                            cls_inputs=cls_inputs,
                            seg_sequence_code=sequence_seg_code_run,
                            cls_sequence_code=sequence_cls_code_run,
                            names=cls_volumeName_run,
                            seg_region_ids=seg_region_ids_run,
                            cls_region_ids=cls_region_ids_run,
                            return_mae_recon=True,
                            mae_mask_ratio=args.mae_mask_ratio,
                            use_masked_encoder=use_masked_encoder
                        )
                    else:
                        seg_preds, cls_preds = model(
                            seg_inputs=seg_inputs,
                            cls_inputs=cls_inputs,
                            seg_sequence_code=sequence_seg_code_run,
                            cls_sequence_code=sequence_cls_code_run,
                            names=cls_volumeName_run,
                            seg_region_ids=seg_region_ids_run,
                            cls_region_ids=cls_region_ids_run
                        )
                finally:
                    pass


                seg_preds_sup = seg_preds
                term_seg_Dice = loss_seg_DICE.forward(seg_preds_sup, seg_labels, seg_volumeName, TEMPLATE)
                term_seg_BCE = loss_seg_CE.forward(seg_preds_sup, seg_labels, seg_volumeName, TEMPLATE)
                term_all_seg = term_seg_Dice + term_seg_BCE
                epoch_seg_loss.append(float(term_all_seg))

              
                N = cls_labels.shape[0]
                total_loss = 0
                for i in range(N):
                    dataset_name = cls_volumeName[i][:3]
                    sample_pred = cls_preds[i]
                    sample_label = cls_labels[i].unsqueeze(0)
                    ce_loss = loss_cls(sample_pred, sample_label)
                    total_loss = total_loss + ce_loss

                avg_ce_loss = total_loss / cls_labels.shape[0]
                epoch_cls_loss.append(float(avg_ce_loss))
                
               

                supervised_loss = term_all_seg + args.cls_weight * avg_ce_loss
                all_loss = current_sup_weight * supervised_loss

                lb_loss = (model.module.last_lb_loss if hasattr(model, 'module') else model.last_lb_loss)
                all_loss = all_loss + current_sup_weight * (args.lb_coeff * lb_loss)
                

                if mae_enabled and mae_recon is not None:


                    all_masks = torch.stack(input_masks, dim=0)
                    avg_mask = all_masks.mean(dim=0)


                    mae_mask_inv = 1.0 - avg_mask
                    loss_mae = F.l1_loss(mae_recon * mae_mask_inv, mae_target * mae_mask_inv, reduction='sum')
                    num_masked_voxels = mae_mask_inv.sum() + 1e-6
                    num_channels = mae_recon.shape[1]
                    loss_mae = loss_mae / (num_masked_voxels * num_channels)
                    
                    all_loss = all_loss + current_mae_weight * loss_mae
                    epoch_ssl_seg_loss.append(float(loss_mae.detach().cpu()))
                    


                else:
                    epoch_ssl_seg_loss.append(0.0)
                
                epoch_ssl_cls_loss.append(0.0)
            else:
                


                if epoch < args.mae_warmup_epochs:
                    current_mae_weight = args.mae_initial_weight * (1.0 - epoch / args.mae_warmup_epochs)
                    use_masked_encoder = True
                else:
                    current_mae_weight = 0.0
                    use_masked_encoder = False
                
                mae_enabled = args.use_mae and current_mae_weight > 0


                if epoch < args.sup_warmup_epochs:
                    current_sup_weight = args.sup_initial_weight + (1.0 - args.sup_initial_weight) * (epoch / args.sup_warmup_epochs)
                else:
                    current_sup_weight = 1.0
                






                
                if mae_enabled:
                    seg_preds, _, mae_recon, mae_target, input_masks = model(
                        seg_inputs=seg_inputs,
                        cls_inputs=None,
                        seg_sequence_code=sequence_seg_code_run,
                        cls_sequence_code=None,
                        names=seg_volumeName_run,
                        seg_region_ids=seg_region_ids_run,
                        cls_region_ids=None,
                        return_mae_recon=True,
                        mae_mask_ratio=args.mae_mask_ratio,
                        use_masked_encoder=use_masked_encoder
                    )
                else:
                    seg_preds, _ = model(
                        seg_inputs=seg_inputs,
                        cls_inputs=None,
                        seg_sequence_code=sequence_seg_code_run,
                        cls_sequence_code=None,
                        names=seg_volumeName_run,
                        seg_region_ids=seg_region_ids_run,
                        cls_region_ids=None
                    )

                seg_preds_sup = seg_preds
                term_seg_Dice = loss_seg_DICE.forward(seg_preds_sup, seg_labels, seg_volumeName, TEMPLATE)
                term_seg_BCE = loss_seg_CE.forward(seg_preds_sup, seg_labels, seg_volumeName, TEMPLATE)
                term_all_seg = term_seg_Dice + term_seg_BCE
                epoch_seg_loss.append(float(term_all_seg))
                lb_loss = (model.module.last_lb_loss if hasattr(model, 'module') else model.last_lb_loss)
                

                supervised_loss = term_all_seg
                all_loss = current_sup_weight * supervised_loss
                all_loss = all_loss + current_sup_weight * (args.lb_coeff * lb_loss)
                epoch_cls_loss.append(0.0)  


                if mae_enabled and mae_recon is not None:


                    all_masks = torch.stack(input_masks, dim=0)
                    avg_mask = all_masks.mean(dim=0)


                    mae_mask_inv = 1.0 - avg_mask
                    loss_mae = F.l1_loss(mae_recon * mae_mask_inv, mae_target * mae_mask_inv, reduction='sum')
                    num_masked_voxels = mae_mask_inv.sum() + 1e-6
                    num_channels = mae_recon.shape[1]
                    loss_mae = loss_mae / (num_masked_voxels * num_channels)
                    
                    all_loss = all_loss + current_mae_weight * loss_mae
                    epoch_ssl_seg_loss.append(float(loss_mae.detach().cpu()))
                    


                else:
                    epoch_ssl_seg_loss.append(0.0)
                
                epoch_ssl_cls_loss.append(0.0)

          
            optimizer.zero_grad()
            all_loss.backward()
            optimizer.step()
            
        epoch_seg_loss = np.mean(epoch_seg_loss)
        all_tr_seg_loss.append(epoch_seg_loss)
        end_epoch_time = timeit.default_timer()
       
        writer.add_scalar('train_seg_loss', epoch_seg_loss, epoch)

        cur_lb = (model.module.last_lb_loss if hasattr(model, 'module') else model.last_lb_loss)
        writer.add_scalar('train_lb_loss', float(cur_lb.detach().cpu()), epoch)


        if len(epoch_ssl_seg_loss) > 0:
            writer.add_scalar('train_mae_loss', float(np.mean(epoch_ssl_seg_loss)), epoch)
        

        if epoch < args.mae_warmup_epochs:
            mae_weight_current = args.mae_initial_weight * (1.0 - epoch / args.mae_warmup_epochs)
        else:
            mae_weight_current = 0.0
        writer.add_scalar('train_mae_weight', mae_weight_current, epoch)
        

        if epoch < args.sup_warmup_epochs:
            sup_weight_current = args.sup_initial_weight + (1.0 - args.sup_initial_weight) * (epoch / args.sup_warmup_epochs)
        else:
            sup_weight_current = 1.0
        writer.add_scalar('train_sup_weight', sup_weight_current, epoch)

   
        logging.info('epoch_{} lr = {:.7f}'.format(epoch, lr))
     
       
        epoch_cls_loss = np.mean(epoch_cls_loss)
        all_tr_cls_loss.append(epoch_cls_loss) 
        end_epoch_time = timeit.default_timer()   
        epoch_time = end_epoch_time-start_epoch_time


        writer.add_scalar('train_cls_loss', epoch_cls_loss, epoch) 


        ssl_seg_mean = float(np.mean(epoch_ssl_seg_loss)) if len(epoch_ssl_seg_loss) > 0 else 0.0
        ssl_cls_mean = float(np.mean(epoch_ssl_cls_loss)) if len(epoch_ssl_cls_loss) > 0 else 0.0

        logging.info(
            'train--seg loss = {:.3f} (ssl: {:.3f}), train--cls loss = {:.3f} (ssl: {:.3f}), time = {:.3f} seconds'
            .format(epoch_seg_loss.item(), ssl_seg_mean, epoch_cls_loss.item(), ssl_cls_mean, epoch_time)
        )
        
        if epoch % 5 == 0:
            if epoch >= 250:
                print('cls  evaling..........')
                    
                val_cls_loss = []
                all_labels = {}
                all_preds = {}
                all_probs = {}
                cls_accuracy = []
                cls_auc = []
                dataset_names = ["cen", "NPC", "LLD", "Bra","Bre"]

              
                for dataset_name in dataset_names:
                    all_labels[dataset_name] = []
                    all_preds[dataset_name] = []
                    all_probs[dataset_name] = []
                
                model.eval()
                with torch.no_grad():
                    for index, batch in enumerate(val_cls_loader):
                       
                        cls_inputs = _modal_inputs_from_np_batch(batch, device)
                        sequence_cls_code = torch.from_numpy(batch['sequence_code']).to(device)
                        labels = torch.from_numpy(batch['label']).to(device)
                        labels = labels.long()
                        volumeName = batch['name']
                        cls_region_ids = torch.from_numpy(batch['region_ids']).to(device)
                    
                        seg_preds, cls_preds = model(
                            seg_inputs=None,
                            cls_inputs=cls_inputs,
                            seg_sequence_code= None,
                            cls_sequence_code=sequence_cls_code,
                            names=volumeName,
                            seg_region_ids=None,
                            cls_region_ids=cls_region_ids
                        )
                        
                        
                       
      
                        N = labels.shape[0]
                        total_loss = 0

                        for i in range(N):

                            dataset_name = volumeName[i][:3]  
                            sample_pred = cls_preds[i]  
                            sample_label = labels[i].unsqueeze(0)   

                            ce_loss = loss_cls(sample_pred, sample_label)
                            
                            total_loss = total_loss + ce_loss
                            
                        avg_loss = total_loss / labels.shape[0]
                        val_cls_loss.append(float(avg_loss))
                        preds = torch.argmax(cls_preds[0], dim=1)
                        probs = torch.softmax(cls_preds[0], dim=1)
                        
                        
                       
                        dataset_prefix = volumeName[0][:3]
                        
                     
                        if isinstance(dataset_prefix, np.ndarray):
                            dataset_prefix = dataset_prefix.item()
                        
                     
                        if dataset_prefix in all_labels:
                            all_labels[dataset_prefix].extend(labels.cpu().numpy().tolist())  
                            all_preds[dataset_prefix].extend(preds.cpu().numpy().tolist())    
                            all_probs[dataset_prefix].extend(probs.cpu().numpy().tolist())    
                        else:
                            print(f"error:'{dataset_prefix}'")

              
                cls_loss = np.mean(val_cls_loss)
                dataset_aucs = {}  

                
                if (args.local_rank == 0):
                    for dataset_name in dataset_names:
                        accuracy = accuracy_score(all_labels[dataset_name], all_preds[dataset_name])
                        cls_accuracy.append(accuracy)
                        logging.info(f'{dataset_name} ACC: {accuracy:.4f}')
                  
                    for dataset_name in dataset_names:
                        labels = np.array(all_labels[dataset_name])
                        probs = np.array(all_probs[dataset_name])
                        
                        if dataset_name == "NPC":  
                            labels_bin = label_binarize(labels, classes=[0, 1, 2, 3])
                            auc_scores = []
                            for i in range(4):

                                auc_scores.append(roc_auc_score(labels_bin[:, i], probs[:, i]))
                            auc = np.mean(auc_scores)
                        else:
                            auc = roc_auc_score(labels, probs[:, 1])  

                        logging.info(f'{dataset_name} AUC: {auc:.4f}')
                        
                        cls_auc.append(auc)
                   
                if (args.local_rank == 0):
                 
                    cls_acc = np.mean(cls_accuracy)
                    cls_auc = np.mean(cls_auc)

                  
                    writer.add_scalar('val_cls_loss', cls_loss, epoch)
                    writer.add_scalar('val_cls_acc', cls_acc, epoch)
                    writer.add_scalar('val_cls_auc', cls_auc, epoch)

                   
                    end_epoch_time = timeit.default_timer()
                    epoch_time = end_epoch_time - start_epoch_time

                  

                    logging.info('val-- cls loss = {:.4f} cls acc = {:.4f}, cls auc = {:.4f}, time = {:.3f} seconds'.format(
                        cls_loss, cls_acc, cls_auc, epoch_time))

                   
                    if not hasattr(args, 'best_acc'):
                        args.best_acc = 0.0
                        args.best_epoch = 0

                  
                    is_best = cls_acc > args.best_acc

                   
                    if is_best:
                        args.best_acc = cls_acc
                        args.best_epoch = epoch
                        logging.info(f'New best acc: {args.best_acc:.4f} at epoch {args.best_epoch}')
                        
                       
                        if args.local_rank == 0: 
                            torch.save(model.state_dict(),os.path.join(args.snapshot_dir, 'best_model.pth'))  
                    if epoch % 5 == 0:      
                        torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'omni_cls'+'_e'+ str(epoch)+'.pth'))
            
                
        if epoch % 10 == 0:
            print('seg  evaling..........')
        

            def predictor_wrapper(inputs, sequence_seg_code, seg_region_ids):

                current_batch_size = inputs.shape[0]
                

                if sequence_seg_code.shape[0] == 1 and current_batch_size > 1:
                    sequence_seg_code = sequence_seg_code.repeat(current_batch_size, 1)
                if seg_region_ids.shape[0] == 1 and current_batch_size > 1:
                    seg_region_ids = seg_region_ids.repeat(current_batch_size, 1)
                
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

            model.eval()
            dice_list = {}
          
            total_dice = 0.0
            total_count = 0
            for key in TEMPLATE.keys():
                dice_list[key] = np.zeros((2,args.seg_classes))
            with torch.no_grad():
                for index, batch in enumerate(val_seg_loader):
        
                    x1,x2,x3,x4,x5,x6,x7,x8, name, label, mask_code, affine,seg_region_ids,task_ids= batch
                    

                    mask_code = mask_code.to(device)
                    x1, x2, x3, x4, x5, x6, x7, x8 = [x.to(device) for x in (x1, x2, x3, x4, x5, x6, x7, x8)]
                   
                 
                    inputs = torch.cat([x1, x2, x3, x4,x5,x6,x7, x8], dim=1) 

                    pred_sigmoid = sliding_window_inference(
                        inputs = inputs,
                        roi_size=(args.roi_x,args.roi_y,args.roi_z),
                        sw_batch_size=4,
                        predictor=lambda inputs: predictor_wrapper(inputs,mask_code,seg_region_ids),
                        overlap=0.5,
                        mode="constant",

                    )

                    cur_output = torch.sigmoid(pred_sigmoid)
                    pred_binary  = np.asarray(np.around(cur_output.cpu()), dtype=np.uint8)
                    label_binary = label.numpy().astype(np.uint8)
                    template_key = get_key_task(name[0]) 

                    organ_list = TEMPLATE[template_key]

                    end = timeit.default_timer()
                    for organ in organ_list:
                    
                        
                        if np.sum(label_binary[:, organ - 1, :, :, :]) == 0:
                            continue
                        val_dice = new_dice(pred_binary[:,organ-1,:,:,:], label_binary[:,organ-1,:,:,:])
                        hd95_distance = Hd_95(pred_binary[:,organ-1,:,:,:], label_binary[:,organ-1,:,:,:])

                        dice_list[template_key][0][organ-1] += val_dice.item()
                        dice_list[template_key][1][organ-1] += 1    
                        
                        total_dice += val_dice.item()
                        total_count += 1

            if (args.local_rank == 0):
                for key in TEMPLATE.keys():
                    organ_list = TEMPLATE[key]
                    content = 'Task%s|'%(key)
                    for organ in organ_list:
                        dice = dice_list[key][0][organ-1] / dice_list[key][1][organ-1]
                        content += '%s: %.3f, '%(ORGAN_NAME[organ-1], dice)
                    logging.info('val_dice {}'.format(content))
                    print(content)
           
                if total_count > 0:
                    avg_dice = total_dice / total_count
                
                    print("all average Dice: %.3f" % avg_dice)
                    logging.info("all average Dice: %.3f" % avg_dice)
                if avg_dice > best_dice:
                    best_dice = avg_dice
                    torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'best_omni_seg_%.4f.pth'%avg_dice))
                    
            
            
            
        if (epoch+1) % 5 == 0 and args.local_rank==0:
                checkpoint = {
                        'model': model.state_dict(),

                        "epoch": epoch
                    }
                torch.save(checkpoint, osp.join(args.snapshot_dir, 'checkpoint_omni_unet' + '_e'+ str(epoch) + '.pth'))

            
            

    end = timeit.default_timer()
    print(end - start)
   
if __name__ == '__main__':
    main()


