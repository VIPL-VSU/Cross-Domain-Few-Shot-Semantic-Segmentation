""" ProtoNet with/without attention learner for Few-shot 3D Point Cloud Semantic Segmentation


"""
import torch
from torch import optim
from torch.nn import functional as F

from models.protonet_QGPA import ProtoNetAlignQGPASR
from models.coseg import COSeg
from models.DPA import DynamicPrototypeAdaptation
from models.protonet import ProtoNet
from utils.checkpoint_util import load_pretrain_checkpoint, load_model_checkpoint
import torchprofile
from utils.logger import init_logger

class ProtoLearner(object):

    def __init__(self, args, mode='train'):

        # init model and optimizer
        # if args.use_transformer or args.use_o2s:
        #     self.model = ProtoNetAlignQGPASR(args)
        # else:
        #     self.model = ProtoNet(args)
        # self.model = ProtoNetAlignQGPASR(args)
        if args.coseg:
            self.model = COSeg(args)
        elif args.DPA:
            self.model = DynamicPrototypeAdaptation(args)
        else:
            self.model = ProtoNetAlignQGPASR(args)
        print(self.model)
        if torch.cuda.is_available():
            self.model.cuda()

        if mode == 'train':
            if args.use_attention:
                if args.use_transformer:
                    self.optimizer = torch.optim.Adam(
                    [#{'params': self.model.encoder.parameters(), 'lr': 0.0001},
                     {'params': self.model.base_learner.parameters()},
                     {'params': self.model.transformer.parameters(), 'lr': args.trans_lr},
                     {'params': self.model.att_learner.parameters()},
                     # {'params': self.model.obj_encoder.parameters()},
                     # {'params': self.model.obj_base_learner.parameters()},
                     # {'params': self.model.obj_att_learner.parameters()},
                     ], lr=args.lr)
                elif args.use_o2s:
                    self.optimizer = torch.optim.Adam(
                        [#{'params': self.model.encoder.parameters(), 'lr': 0.0001},
                         {'params': self.model.base_learner.parameters()},
                         {'params': self.model.transformer.parameters(), 'lr': args.trans_lr},
                         {'params': self.model.o2s_module.parameters(), 'lr': args.trans_lr},
                         {'params': self.model.att_learner.parameters()},
                         # {'params': self.model.obj_encoder.parameters()},
                         # {'params': self.model.obj_base_learner.parameters()},
                         # {'params': self.model.obj_att_learner.parameters()},
                         ], lr=args.lr)
                else:
                    self.optimizer = torch.optim.Adam(
                    [{'params': self.model.encoder.parameters(), 'lr': 0.0001},
                     {'params': self.model.base_learner.parameters()},
                     {'params': self.model.att_learner.parameters()}
                     ], lr=args.lr)
            else:
                self.optimizer = torch.optim.Adam(
                    [{'params': self.model.encoder.parameters(), 'lr': 0.0001},
                     {'params': self.model.base_learner.parameters()},
                     {'params': self.model.linear_mapper.parameters()}], lr=args.lr)
            #set learning rate scheduler
            self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=args.step_size,
                                                          gamma=args.gamma)
            # load pretrained model for point cloud encoding
            self.model = load_pretrain_checkpoint(self.model, args.pretrain_checkpoint_path)
        elif mode == 'test':
            # Load model checkpoint
            self.model = load_model_checkpoint(self.model, args.model_checkpoint_path, mode='test')
        else:
            raise ValueError('Wrong GMMLearner mode (%s)! Option:train/test' %mode)

    def train(self, data, sampled_classes):
        """
        Args:
            data: a list of torch tensors wit the following entries.
            - support_x: support point clouds with shape (n_way, k_shot, in_channels, num_points)
            - support_y: support masks (foreground) with shape (n_way, k_shot, num_points)
            - query_x: query point clouds with shape (n_queries, in_channels, num_points)
            - query_y: query labels with shape (n_queries, num_points)
        """

        [support_x, query_x, query_y] = data
        self.model.train()

        query_logits, loss, _ = self.model(support_x, query_x, query_y, 'train')

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()

        query_pred = F.softmax(query_logits, dim=1).argmax(dim=1)
        correct = torch.eq(query_pred, query_y).sum().item()  # including background class
        accuracy = correct / (query_y.shape[0]*query_y.shape[1])

        return loss, accuracy

    def test(self, data):
        """
        Args:
            support_x: support point clouds with shape (n_way, k_shot, in_channels, num_points)
            support_y: support masks (foreground) with shape (n_way, k_shot, num_points), each point \in {0,1}.
            query_x: query point clouds with shape (n_queries, in_channels, num_points)
            query_y: query labels with shape (n_queries, num_points), each point \in {0,..., n_way}
        """
        [support_x, query_x, query_y] = data
        self.model.eval()

        with torch.no_grad():
            logits, loss, fg_prototypes = self.model(support_x, query_x, query_y, 'test')
            #print(logits)
            pred = F.softmax(logits, dim=1).argmax(dim=1)

            # bg_id = query_y.view(-1) == 1
            # fg_id = torch.logical_not(bg_id)
            # bg_pred_score_bg = logits.view(-1, logits.shape[1])[bg_id, 1]
            # fg_pred_score_bg = logits.view(-1, logits.shape[1])[fg_id, 1]
            # print('bg pred bg score: max: {} | min: {} | mean: {}'.format(bg_pred_score_bg.max(0)[0], bg_pred_score_bg.min(0)[0], bg_pred_score_bg.mean(0)))
            # print('fg pred bg score: max: {} | min: {} | mean: {}'.format(fg_pred_score_bg.max(0)[0], fg_pred_score_bg.min(0)[0], fg_pred_score_bg.mean(0)))

            # pred_score = F.softmax(logits, dim=1).max(dim=1)[0].view(-1)
            # correct_id = torch.eq(pred, query_y).view(-1)
            # wrong_id = torch.logical_not(correct_id).view(-1)
            # print(correct_id.sum(), wrong_id.sum())
            # wrong_score_max = pred_score[wrong_id].max(0)[0]
            # wrong_score_min = pred_score[wrong_id].min(0)[0]
            # wrong_score_mean = pred_score[wrong_id].mean(0)
            # correct_score_max = pred_score[correct_id].max(0)[0]
            # correct_score_min = pred_score[correct_id].min(0)[0]
            # correct_score_mean = pred_score[correct_id].mean(0)
            # print('wrong max: {} | min: {} | mean: {}'.format(wrong_score_max,wrong_score_min,wrong_score_mean))
            # print('correct max: {} | min: {} | mean: {}'.format(correct_score_max, correct_score_min, correct_score_mean))

            correct = torch.eq(pred, query_y).sum().item()
            query_y = query_y.view(-1)
            accuracy = correct / (query_y.shape[0])

        # 使用 torchprofile 来统计 FLOPs
        flops = torchprofile.profile_macs(self.model, (support_x, query_x, query_y, 'test'))

        # 计算参数数量
        params = sum(p.numel() for p in self.model.parameters())

        print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")
        print(f"Parameters: {params / 1e6:.2f} M")  

        return pred, loss, accuracy