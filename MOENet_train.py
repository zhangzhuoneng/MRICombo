import argparse
import os, sys
sys.path.append("..")
sys.path.append("../../")
sys.path.append(".")
import torch
import numpy as np
import torch.backends.cudnn as cudnn
import os.path as osp
from torch.utils.tensorboard import SummaryWriter
import logging
from network.OmniNet import omni_seg_cls
from MOE_dataset_seg import UnisegDataset,tr_seg_collate
from MOE_dataset_cls import UniclsDataset, tr_cls_collate,val_cls_collate
import random
import timeit
from loss_functions import omni_loss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader, WeightedRandomSampler
from utils import ChinaTimeFormatter, adjust_learning_rate, weight_base_init_new, WeightedRandomSamplerDDP
from utils import ChinaTimeFormatter, adjust_learning_all_rate, WeightedRandomSamplerDDP,new_dice,Hd_95
from utils import TEMPLATE,ORGAN_NAME,get_key_task
from monai.inferers import sliding_window_inference
from sklearn.metrics import accuracy_score
from torch.nn.parallel import DistributedDataParallel 
from sklearn.metrics import accuracy_score
from PCGrad import PCGrad
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
   

start = timeit.default_timer()

def get_arguments():
    parser = argparse.ArgumentParser(description="UniMRINet")
    parser.add_argument('--dist', dest='dist', type=bool, default=False,
                        help='distributed training or not')
    #parser.add_argument("--data_dir", type=str, default="/root/dataspace/Medical_image_database/MRI/segmentation/")#NAS root
    #parser.add_argument("--data_dir", type=str, default="/root/localspace/zzn/UniMRINet/dataset/")#local root
    parser.add_argument("--data_dir", type=str, default="/data/zzn/UniMRINet/dataset/")#local root
    parser.add_argument("--log_dir", type=str, default='./log/log_omni_MRICombo')
    parser.add_argument("--tensorboard_log_name", type=str, default='/omni_seg_cls_MRICombo_lb_4experts_0.05_400_0.5cls_weight')
    parser.add_argument("--snapshot_dir", type=str, default='./snapshots/omni_seg_cls_MRICombo_lb_4experts_0.05_400_0.5cls_weight')
    parser.add_argument("--cls_weight", type=float, default=0.5)
    parser.add_argument('--backbone_name', default='MRICombo', help='backbone unet,swinunetr') 
    parser.add_argument("--train_seg_list", type=str, default= "/data/zzn/UniMRINet/dataset/segmentation/seg_train_1.txt" )
    parser.add_argument("--val_seg_list", type=str, default= "/data/zzn/UniMRINet/dataset/segmentation/seg_val_1.txt" )
    parser.add_argument("--train_cls_list", type=str, default="/data/zzn/UniMRINet/dataset/classification/cls_train_new.txt")
    parser.add_argument("--val_cls_list", type=str, default="/data/zzn/UniMRINet/dataset/classification/cls_test_new.txt" )
    parser.add_argument("--reload_path", type=str, default='/data/zzn/UniMRINet/code/MRICombo/snapshots/omni_seg_cls_MRICombo_lb_4experts_0.05_400/checkpoint_omni_unet_e44.pth')
    parser.add_argument('--trans_encoding', default='word_embedding', 
                        help='the type of encoding: rand_embedding or word_embedding')
    parser.add_argument('--word_embedding', default='../pretrained_weights/sequence_cancer_encoding.pth', 
                        help='The path of word embedding')
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
        # Distributed training
        # print("a")
        args.local_rank = int(os.environ['LOCAL_RANK'])  
        torch.cuda.set_device(args.local_rank)
        world_size = int(os.environ['WORLD_SIZE'])
        torch.distributed.init_process_group(backend='nccl')
        torch.distributed.barrier()
    else:
        # Single GPU training
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.cuda.set_device(0)

    # model = omni_seg_cls(
    # img_size=(args.roi_x,args.roi_y,args.roi_z),
    # seg_in_channels=4,
    # cls_in_channels=7,
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

   
    model.train()
  
    device = torch.device('cuda:{}'.format(args.local_rank))
    model.to(device)
 
    if args.dist:
        # model = DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DistributedDataParallel(model, device_ids=[args.local_rank],find_unused_parameters=True)
        
        # model._set_static_graph()
      
    else:
        model = torch.nn.DataParallel(model).cuda()
    
   
  
    # model = DDP(model, device_ids=[args.local_rank])
    # model._set_static_graph()
    # optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=0.99, nesterov=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    # optimizer = PCGrad(optimizer)
    
    
    # load checkpoint...
    if args.reload_from_checkpoint:
        if os.path.exists(args.reload_path):
            checkpoint = torch.load(args.reload_path, map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint['model'])
            # optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch']
            print('loading from checkpoint: {}'.format(args.reload_path))
            
        else:
            print('File not exists in the reload path: {}'.format(args.reload_path))

    # seg_loss
    loss_seg_DICE = omni_loss.DiceLoss(num_classes=args.seg_classes).to(device)
    loss_seg_CE = omni_loss.CELoss(num_classes=args.seg_classes).to(device)


    # cls_loss
    # classes_weights = torch.tensor([0.0528, 0.3361, 0.3870, 0.2241], device=device)
    # loss_cls = CrossEntropyLoss(weight=classes_weights).to(device)
    loss_cls = CrossEntropyLoss(ignore_index=-1).to(device)


    #seg dataset and loader
    train_seg_dataset = UnisegDataset(args.data_dir, args.train_seg_list, split="train",
                                crop_size=(args.roi_x,args.roi_y,args.roi_z), scale=args.random_scale, mirror=args.random_mirror)
    seg_sample_weight = weight_base_init_new(train_seg_dataset,'seg')

    if args.dist:
         # Distributed training
        weighted_seg_sampler = WeightedRandomSamplerDDP(
                                        data_set = train_seg_dataset,
                                        weights = seg_sample_weight,
                                        num_replicas = world_size,
                                        rank = args.local_rank,
                                        num_samples=len(seg_sample_weight),
                                        replacement=True)
    else:
        # Single GPU training
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
                                # sampler = weighted_seg_sampler,
                                # collate_fn=tr_seg_collate
                                )
    
    #cls dataset and loader
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
    for epoch in range(args.start_epoch+1,args.num_epochs):
        # if epoch < args.start_epoch:
        #     continue
        if args.dist:
            weighted_seg_sampler.set_epoch(epoch)
            weighted_cls_sampler.set_epoch(epoch)

        epoch_seg_loss = []
        epoch_cls_loss = []
        start_epoch_time = timeit.default_timer()
        lr=adjust_learning_all_rate(optimizer, epoch, args.learning_rate, args.num_epochs, args.power)
        for (seg_batch, cls_batch) in zip(train_seg_loader, train_cls_loader):
            # seg task
            seg_x1 = torch.from_numpy(seg_batch['x1']).cuda()
            seg_x2 = torch.from_numpy(seg_batch['x2']).cuda()
            seg_x3 = torch.from_numpy(seg_batch['x3']).cuda()
            seg_x4 = torch.from_numpy(seg_batch['x4']).cuda()
            seg_x5 = torch.from_numpy(seg_batch['x5']).cuda()
            seg_x6 = torch.from_numpy(seg_batch['x6']).cuda()
            seg_x7 = torch.from_numpy(seg_batch['x7']).cuda()
            seg_x8 = torch.from_numpy(seg_batch['x8']).cuda()
            sequence_seg_code = torch.from_numpy(seg_batch['sequence_code']).cuda()
            seg_labels = torch.from_numpy(seg_batch['label']).cuda()
            seg_volumeName = seg_batch['name']
            
            # task and dataset prompt
            seg_region_ids = torch.from_numpy(seg_batch['region_ids']).cuda()
            seg_task_ids = torch.from_numpy(seg_batch['task_ids']).cuda()

            # cls task
            if epoch >= 250:
                cls_x1 = torch.from_numpy(cls_batch['x1']).cuda()
                cls_x2 = torch.from_numpy(cls_batch['x2']).cuda()
                cls_x3 = torch.from_numpy(cls_batch['x3']).cuda()
                cls_x4 = torch.from_numpy(cls_batch['x4']).cuda()
                cls_x5 = torch.from_numpy(cls_batch['x5']).cuda()
                cls_x6 = torch.from_numpy(cls_batch['x6']).cuda()
                cls_x7 = torch.from_numpy(cls_batch['x7']).cuda()
                cls_x8 = torch.from_numpy(cls_batch['x8']).cuda()
                sequence_cls_code = torch.from_numpy(cls_batch['sequence_code']).cuda()
                cls_labels = torch.from_numpy(cls_batch['label']).cuda()
                cls_volumeName = cls_batch['name']
                
                 # task and dataset prompt
                cls_region_ids = torch.from_numpy(cls_batch['region_ids']).cuda()
                cls_task_ids = torch.from_numpy(cls_batch['task_ids']).cuda()


           
                seg_inputs = (seg_x1, seg_x2, seg_x3, seg_x4, seg_x5, seg_x6, seg_x7, seg_x8)
                cls_inputs = (cls_x1, cls_x2, cls_x3, cls_x4, cls_x5, cls_x6, cls_x7, cls_x8)
                
                # use model
                seg_preds, cls_preds = model(
                    seg_inputs=seg_inputs,
                    cls_inputs=cls_inputs,
                    seg_sequence_code=sequence_seg_code,
                    cls_sequence_code=sequence_cls_code,
                    names=cls_volumeName,
                    seg_region_ids=seg_region_ids,
                    cls_region_ids=cls_region_ids
                )

                # seg loss
                term_seg_Dice = loss_seg_DICE.forward(seg_preds, seg_labels, seg_volumeName, TEMPLATE)
                term_seg_BCE = loss_seg_CE.forward(seg_preds, seg_labels, seg_volumeName, TEMPLATE)
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
                # cls loss
                avg_ce_loss = total_loss / cls_labels.shape[0]
                epoch_cls_loss.append(float(avg_ce_loss))
                
               
                all_loss = term_all_seg + args.cls_weight*avg_ce_loss
                # expert balance loss
                lb_loss = (model.module.last_lb_loss if hasattr(model, 'module') else model.last_lb_loss)
                all_loss = all_loss + args.lb_coeff * lb_loss
            else:
               
                seg_inputs = (seg_x1, seg_x2, seg_x3, seg_x4, seg_x5, seg_x6, seg_x7, seg_x8)
                seg_preds, _ = model(
                    seg_inputs=seg_inputs,
                    cls_inputs=None,
                    seg_sequence_code=sequence_seg_code,
                    cls_sequence_code=None,
                    names=seg_volumeName,
                    seg_region_ids=seg_region_ids,
                    cls_region_ids=None
                )
                term_seg_Dice = loss_seg_DICE.forward(seg_preds, seg_labels, seg_volumeName, TEMPLATE)
                term_seg_BCE = loss_seg_CE.forward(seg_preds, seg_labels, seg_volumeName, TEMPLATE)
                term_all_seg = term_seg_Dice + term_seg_BCE
                epoch_seg_loss.append(float(term_all_seg))
                lb_loss = (model.module.last_lb_loss if hasattr(model, 'module') else model.last_lb_loss)
                all_loss = term_all_seg + args.lb_coeff * lb_loss
                epoch_cls_loss.append(0.0)  

          
            optimizer.zero_grad()
            all_loss.backward()
            optimizer.step()
            
        epoch_seg_loss = np.mean(epoch_seg_loss)
        all_tr_seg_loss.append(epoch_seg_loss)
        end_epoch_time = timeit.default_timer()
       
        writer.add_scalar('train_seg_loss', epoch_seg_loss, epoch)

        cur_lb = (model.module.last_lb_loss if hasattr(model, 'module') else model.last_lb_loss)
        writer.add_scalar('train_lb_loss', float(cur_lb.detach().cpu()), epoch)

   
        logging.info('epoch_{} lr = {:.7f}'.format(epoch, lr))
     
       
        epoch_cls_loss = np.mean(epoch_cls_loss)
        all_tr_cls_loss.append(epoch_cls_loss) 
        end_epoch_time = timeit.default_timer()   
        epoch_time = end_epoch_time-start_epoch_time

        # writer.add_scalar('info/lr', lr, epoch)
        writer.add_scalar('train_cls_loss', epoch_cls_loss, epoch) 

        logging.info('train--seg loss = {:.3f}, train--cls loss = {:.3f}, time = {:.3f} seconds'\
        .format(epoch_seg_loss.item(),epoch_cls_loss.item(),epoch_time))
        
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
                # dataset_names = ["LLD"]
              
                for dataset_name in dataset_names:
                    all_labels[dataset_name] = []
                    all_preds[dataset_name] = []
                    all_probs[dataset_name] = []
                
                model.eval()
                with torch.no_grad():
                    for index, batch in enumerate(val_cls_loader):
                       
                        cls_x1 = torch.from_numpy(batch['x1']).cuda()
                        cls_x2 = torch.from_numpy(batch['x2']).cuda()
                        cls_x3 = torch.from_numpy(batch['x3']).cuda()
                        cls_x4 = torch.from_numpy(batch['x4']).cuda()
                        cls_x5 = torch.from_numpy(batch['x5']).cuda()
                        cls_x6 = torch.from_numpy(batch['x6']).cuda()
                        cls_x7 = torch.from_numpy(batch['x7']).cuda()
                        cls_x8 = torch.from_numpy(batch['x8']).cuda()
                        # cls_x7 = torch.zeros_like(cls_x1).cuda()  
                        # cls_x8 = torch.zeros_like(cls_x1).cuda() 
                        # x7 = torch.from_numpy(batch['x7']).cuda()
                        # x8 = torch.from_numpy(batch['x8']).cuda()
                        sequence_cls_code = torch.from_numpy(batch['sequence_code']).cuda()
                        labels = torch.from_numpy(batch['label']).cuda()
                        labels = labels.long()
                        volumeName = batch['name']
                        cls_region_ids = torch.from_numpy(batch['region_ids']).cuda()
                        cls_task_ids = torch.from_numpy(batch['task_ids']).cuda()
                    
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
                        
                        
                       
      
                        N = labels.shape[0]
                        total_loss = 0
                        # # print( volumeName,N)
                        for i in range(N):
                            # print
                            dataset_name = volumeName[i][:3]  
                            sample_pred = cls_preds[i]  
                            sample_label = labels[i].unsqueeze(0)   
                            # print(i,dataset_name,sample_pred,sample_label)
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
                                # print(i,roc_auc_score(labels_bin[:, i], probs[:, i]))
                                auc_scores.append(roc_auc_score(labels_bin[:, i], probs[:, i]))
                            auc = np.mean(auc_scores)
                        else:  # 二分类数据集
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

                  
                    # if (args.local_rank == 0):
                    logging.info('val-- cls loss = {:.4f} cls acc = {:.4f}, cls auc = {:.4f}, time = {:.3f} seconds'.format(
                        cls_loss, cls_acc, cls_auc, epoch_time))

                   
                    if not hasattr(args, 'best_acc'):
                        args.best_acc = 0.0
                        args.best_epoch = 0

                  
                    is_best = cls_acc > args.best_acc

                   
                    if is_best:
                        args.best_acc = cls_acc
                        args.best_epoch = epoch
                        logging.info(f'新的最佳acc: {args.best_acc:.4f}, 在epoch {args.best_epoch}')
                        
                       
                        if args.local_rank == 0: 
                            torch.save(model.state_dict(),os.path.join(args.snapshot_dir, 'best_model.pth'))  
                    if epoch % 5 == 0:      
                        torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'omni_cls'+'_e'+ str(epoch)+'.pth'))
            
                
        if epoch % 20 == 1:
            print('seg  evaling..........')
        
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
            

            model.eval()
            dice_list = {}
          
            total_dice = 0.0
            total_count = 0
            for key in TEMPLATE.keys():
                dice_list[key] = np.zeros((2,args.seg_classes)) # 1st row for dice, 2nd row for count
            with torch.no_grad():
                for index, batch in enumerate(val_seg_loader):
        
                    x1,x2,x3,x4,x5,x6,x7,x8, name, label, mask_code, affine,seg_region_ids,task_ids= batch
                    
                    # print(x1.shape)
                    mask_code = mask_code.to(device)
        
                    x1 = x1.to(device)
                    x2 = x2.to(device)
                    x3 = x3.to(device)
                    x4 = x4.to(device)
                    x5 = x5.to(device)
                    x6 = x6.to(device)
                    x7 = x7.to(device)
                    x8 = x8.to(device)
                   
                 
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
                    # print( template_key)
                    organ_list = TEMPLATE[template_key]
                    # print( organ_list)
                    end = timeit.default_timer()
                    for organ in organ_list:
                    
                        
                        if np.sum(label_binary[:, organ - 1, :, :, :]) == 0:
                            continue
                        val_dice = new_dice(pred_binary[:,organ-1,:,:,:], label_binary[:,organ-1,:,:,:])
                        hd95_distance = Hd_95(pred_binary[:,organ-1,:,:,:], label_binary[:,organ-1,:,:,:])
                        # print('%s: %s dice = %.3f hd_95 = %.2f s'%(name[0],ORGAN_NAME[organ - 1],val_dice, hd95_distance))
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
                        # 'optimizer': optimizer.state_dict(),
                        "epoch": epoch
                    }
                torch.save(checkpoint, osp.join(args.snapshot_dir, 'checkpoint_omni_unet' + '_e'+ str(epoch) + '.pth'))

            
            

    end = timeit.default_timer()
    print(end - start)
   
if __name__ == '__main__':
    main()
# multi-gpu
# NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=4,5,6 python -m torch.distributed.launch --nproc_per_node=3 --master_port 6535 train.py --dist True
