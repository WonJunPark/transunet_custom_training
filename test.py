import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.dataset_synapse import Synapse_dataset
from utils import test_single_volume
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from glob import glob
from utils2.dataset import BasicDataset
from losses import dice_coeff
import torch.nn.functional as F
from utils import DiceLoss
from torchvision import transforms
from sklearn.metrics import confusion_matrix
from utils2.colors import get_colors
import cv2
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--volume_path', type=str,
                    default='../data/Synapse/test_vol_h5', help='root dir for validation volume data')  # for acdc volume_path=root_dir
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=3, help='output channel of network')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')

parser.add_argument('--max_iterations', type=int,default=20000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=100, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=7,
                    help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=512, help='input patch size of network input')
parser.add_argument('--is_savenii', action="store_true", help='whether to save results during inference')

parser.add_argument('--n_skip', type=int, default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16', help='select one vit model')

parser.add_argument('--test_save_dir', type=str, default='../predictions', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')
args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def total_metric(pred, label):
    intersection = (pred*label).sum()
    dice_coef = (2*intersection)/(pred.sum() + label.sum())
    jaccard_coef = intersection/((pred.sum() + label.sum())-intersection)

    pred = pred.flatten()
    label = label.flatten()

    # print(confusion_matrix(pred, label, labels=[1, 0]))
    # https://sites.google.com/site/torajim/articles/performance_measure

    conf = confusion_matrix(label, pred, labels=[1, 0])
    TP = conf[0, 0]
    FN = conf[0, 1]
    FP = conf[1, 0]
    TN = conf[1, 1]

    dice_coef = 2 * TP / (2 * TP + FP + FN + 1.)
    jaccard_coef = TP / (TP + FP + FN + 1.)
    acc = (TP + TN) / (TP + TN + FP + FN + 1.)
    sensitivity = TP / (TP + FN + 1.)
    specificity = TN / (FP + TN + 1.)
    precision = TP / (TP + FP + 1.)

    return round(dice_coef*100, 2), round(acc*100,2), round(sensitivity*100,2), \
           round(specificity*100,2), round(precision*100,2), round(jaccard_coef*100,2)

def inference(args, model, test_save_path=None):
    # db_test = args.Dataset(base_dir=args.volume_path, split="test_vol", list_dir=args.list_dir)
    # testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    # logging.info("{} test iterations per epoch".format(len(testloader)))

    data_path = '../sample_preprocessing/test/input/'
    label_path = '../sample_preprocessing/test/label/'

    data_file_list = sorted(os.listdir(data_path))
    label_file_list = sorted(os.listdir(label_path))

    # img_path_list = sorted(glob('../sample_preprocessing/test/input/*/*.png'))
    # mask_path_list = sorted(glob('../sample_preprocessing/test/label/*/*.png'))
    #
    # dataset = BasicDataset(img_path_list, mask_path_list)
    # test_loader = DataLoader(dataset,
    #                           batch_size=1,
    #                           shuffle=True,
    #                           num_workers=8,
    #                           pin_memory=True)

    model.eval()
    ivh_dice_t, acc_t, sensitivity_t, specificity_t, precision_t, ivh_jaccard_t = 0, 0, 0, 0, 0, 0
    ich_dice_t, acc2_t, sensitivity2_t, specificity2_t, precision2_t, ich_jaccard_t = 0, 0, 0, 0, 0, 0
    ivh_c = 0
    ich_c = 0
    count = 0

    for d in tqdm(range(len(data_file_list))):
    # for d in range(1):
        ct_path = data_path + data_file_list[d]
        mask_path = label_path + label_file_list[d]

        ct_path2 = sorted(os.listdir(ct_path))
        mask_path2 = sorted(os.listdir(mask_path))


        for i in range(len(ct_path2)):
            img_path2 = ct_path + '/' + ct_path2[i]
            label_path2 = mask_path + '/' + mask_path2[i]

            img = cv2.imread(img_path2)
            img2 = torch.from_numpy(BasicDataset.preprocess(img, 1))
            img2 = img2.unsqueeze(0)
            img2 = img2.to(device=device, dtype=torch.float32)

            with torch.no_grad():
                outputs = model(img2)

                outputs = F.softmax(outputs, dim=1)
                # logging.info('ivh : %f / ich : %f' % (outputs[:,1,:,:].max(),outputs[:,2,:,:].max()))
                outputs = outputs.squeeze(0)

                tf = transforms.Compose(
                        [
                            transforms.ToPILImage(),
                            transforms.Resize(512, 512),
                            transforms.ToTensor()
                        ]
                )

                masks = []
                for prob in outputs:
                    prob = tf(prob.cpu())
                    mask = prob.squeeze().cpu().numpy()
                    mask = mask > 0.4 #out_threshold
                    masks.append(mask)


            label = cv2.imread(label_path2)
            label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
            h, w = label.shape

            ivh = np.where(label == 1, 1, 0)
            ich = np.where(label == 2, 1, 0)

            if ivh.sum() != 0:
                ivh_dice, acc, sensitivity, specificity, precision, ivh_jaccard \
                    = total_metric(masks[1], ivh)
                ivh_dice_t += ivh_dice
                acc_t += acc
                sensitivity_t += sensitivity
                specificity_t += specificity
                precision_t += precision
                ivh_jaccard_t += ivh_jaccard
                ivh_c += 1

            if ich.sum() != 0:
                ich_dice, acc2, sensitivity2, specificity2, precision2, ich_jaccard \
                    = total_metric(masks[2], ich)

                ich_dice_t += ich_dice
                acc2_t += acc2
                sensitivity2_t += sensitivity2
                specificity2_t += specificity2
                precision2_t += precision2
                ich_jaccard_t += ich_jaccard
                ich_c += 1

            output_path = '../sample_preprocessing/test/output/' + data_file_list[d][:3]

            if not os.path.exists(output_path):
                os.makedirs(output_path)

            colors = get_colors(n_classes=3)
            w, h = 512, 512

            img_mask = np.zeros([h, w, 3], np.uint8)
            for idx in range(0, len(masks)):
                image_idx = Image.fromarray((masks[idx] * 255).astype(np.uint8))
                # plt.imshow(image_idx)
                # plt.show()
                array_img = np.asarray(image_idx)
                img_mask[np.where(array_img == 255)] = colors[idx]

            img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            img_mask = cv2.cvtColor(np.asarray(img_mask), cv2.COLOR_RGB2BGR)
            output = cv2.addWeighted(img, 0.6, img_mask, 0.4, 0)

            cv2.imwrite(output_path + '/{:03d}.png'.format(i+1), output)

    print("========IVH========")
    print("dice : {}".format(round(ivh_dice_t / ivh_c, 2)))
    print("jacard : {}".format(round(ivh_jaccard_t / ivh_c, 2)))
    print("sensitivity : {}".format(round(sensitivity_t / ivh_c, 2)))
    print("specificity : {}".format(round(specificity_t / ivh_c, 2)))
    print("precision : {}".format(round(precision_t / ivh_c, 2)))

    print("========ICH========")
    print("dice : {}".format(round(ich_dice_t / ich_c, 2)))
    print("jacard : {}".format(round(ich_jaccard_t / ich_c, 2)))
    print("sensitivity : {}".format(round(sensitivity2_t / ich_c, 2)))
    print("specificity : {}".format(round(specificity2_t / ich_c, 2)))
    print("precision : {}".format(round(precision2_t / ich_c, 2)))

    #
    # for i_batch, sampled_batch in tqdm(enumerate(test_loader)):
    #     h, w = sampled_batch["image"].size()[2:]
    #     # image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
    #     # metric_i = test_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
    #     #                               test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing)
    #
    #     image_batch, label_batch = sampled_batch['image'], sampled_batch['mask']
    #     image_batch, label_batch = image_batch.cuda().float(), label_batch.cuda().float()
    #
    #     with torch.no_grad():
    #         outputs = model(image_batch)
    #         outputs = F.softmax(outputs, dim=1)
    #         outputs = torch.squeeze(outputs)
    #
    #         tf = transforms.Compose(
    #                 [
    #                     transforms.ToPILImage(),
    #                     transforms.Resize(512, 512),
    #                     transforms.ToTensor()
    #                 ]
    #         )
    #
    #         masks = []
    #         for prob in outputs:
    #             prob = tf(prob.cpu())
    #             mask = prob.squeeze().cpu().numpy()
    #             mask = mask > 0.3 #out_threshold
    #             masks.append(mask)
    #
    #     label_batch = torch.squeeze(label_batch).cpu().numpy()
    #
    #     ivh = np.where(label_batch == 1, 1, 0)
    #     ich = np.where(label_batch == 2, 1, 0)
    #
    #     if ivh.sum() != 0:
    #         ivh_dice, acc, sensitivity, specificity, precision, ivh_jaccard \
    #             = total_metric(masks[1], ivh)
    #         ivh_dice_t += ivh_dice
    #         acc_t += acc
    #         sensitivity_t += sensitivity
    #         specificity_t += specificity
    #         precision_t += precision
    #         ivh_jaccard_t += ivh_jaccard
    #         ivh_c += 1
    #
    #     if ich.sum() != 0:
    #         ich_dice, acc2, sensitivity2, specificity2, precision2, ich_jaccard \
    #             = total_metric(masks[2], ich)
    #
    #         ich_dice_t += ich_dice
    #         acc2_t += acc2
    #         sensitivity2_t += sensitivity2
    #         specificity2_t += specificity2
    #         precision2_t += precision2
    #         ich_jaccard_t += ich_jaccard
    #         ich_c += 1
    #
    #     output_path = './sample_preprocessing/test/output/'
    #
    #     if not os.path.exists(output_path):
    #         os.makedirs(output_path)
    #
    #     colors = get_colors(n_classes=3)
    #     w, h = 512, 512
    #
    #     img_mask = np.zeros([h, w, 3], np.uint8)
    #     for idx in range(0, len(masks)):
    #         image_idx = Image.fromarray((masks[idx] * 255).astype(np.uint8))
    #         # plt.imshow(image_idx)
    #         # plt.show()
    #         array_img = np.asarray(image_idx)
    #         img_mask[np.where(array_img == 255)] = colors[idx]
    #
    #     img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    #     img_mask = cv2.cvtColor(np.asarray(img_mask), cv2.COLOR_RGB2BGR)
    #     output = cv2.addWeighted(img, 0.6, img_mask, 0.4, 0)
    #
    #     cv2.imwrite(output_path + '/{:03d}.png'.format(count + 1), output)
    #     count += 1
    #
    #
    # print("========IVH========")
    # print("dice : {}".format(round(ivh_dice_t / ivh_c, 2)))
    # print("jacard : {}".format(round(ivh_jaccard_t / ivh_c, 2)))
    # print("sensitivity : {}".format(round(sensitivity_t / ivh_c, 2)))
    # print("specificity : {}".format(round(specificity_t / ivh_c, 2)))
    # print("precision : {}".format(round(precision_t / ivh_c, 2)))
    #
    # print("========ICH========")
    # print("dice : {}".format(round(ich_dice_t / ich_c, 2)))
    # print("jacard : {}".format(round(ich_jaccard_t / ich_c, 2)))
    # print("sensitivity : {}".format(round(sensitivity2_t / ich_c, 2)))
    # print("specificity : {}".format(round(specificity2_t / ich_c, 2)))
    # print("precision : {}".format(round(precision2_t / ich_c, 2)))



    #
    #     metric_list += np.array(metric_i)
    #     logging.info('idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
    #
    # metric_list = metric_list / len(dataset)
    #
    #
    # for i in range(1, args.num_classes):
    #     logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
    # performance = np.mean(metric_list, axis=0)[0]
    # mean_hd95 = np.mean(metric_list, axis=0)[1]
    # logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
    # return "Testing Finished!"


if __name__ == "__main__":

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_config = {
        'Synapse': {
            'Dataset': Synapse_dataset,
            'volume_path': '../data/Synapse/test_vol_h5',
            'list_dir': './lists/lists_Synapse',
            'num_classes': 3,
            'z_spacing': 1,
        },
    }
    dataset_name = args.dataset
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.volume_path = dataset_config[dataset_name]['volume_path']
    args.Dataset = dataset_config[dataset_name]['Dataset']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.z_spacing = dataset_config[dataset_name]['z_spacing']
    args.is_pretrain = True

    # name the same snapshot defined in train script!
    args.exp = 'TU_' + dataset_name + str(args.img_size)
    snapshot_path = "./model/{}/{}".format(args.exp, 'TU')
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    snapshot_path = snapshot_path + '_vitpatch' + str(args.vit_patches_size) if args.vit_patches_size!=16 else snapshot_path
    snapshot_path = snapshot_path + '_epo' + str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    if dataset_name == 'ACDC':  # using max_ep
        # och instead of iteration to control training duration
        snapshot_path = snapshot_path + '_' + str(args.max_iterations)[0:2] + 'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_'+str(args.img_size)
    snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed!=1234 else snapshot_path

    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)

    if args.vit_name.find('R50') !=-1:
        config_vit.patches.grid = (int(args.img_size/args.vit_patches_size), int(args.img_size/args.vit_patches_size))

    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()

    snapshot = os.path.join(snapshot_path, 'best_model.pth')
    # snapshot = os.path.join(snapshot_path, 'epoch_99.pth')


    if not os.path.exists(snapshot): snapshot = snapshot.replace('best_model', 'epoch_'+str(args.max_epochs-1))
    # snapshot = './model/TU_Synapse512/TU_pretrain_R50-ViT-B_16_skip3_epo100_bs7_512/211125_epoch_99.pth'
    snapshot = './model/TU_Synapse512/TU_pretrain_R50-ViT-B_16_skip3_epo100_bs7_512/epoch_49.pth'

    net.load_state_dict(torch.load(snapshot))
    # net = nn.DataParallel(net)

    snapshot_name = snapshot_path.split('/')[-1]

    log_folder = './test_log/test_log_' + args.exp
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/'+snapshot_name+".txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    if args.is_savenii:
        args.test_save_dir = '../predictions'
        test_save_path = os.path.join(args.test_save_dir, args.exp, snapshot_name)
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None
    inference(args, net, test_save_path)


