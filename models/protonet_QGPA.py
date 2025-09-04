""" Prototypical Network 


"""
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.dgcnn import DGCNN
from models.dgcnn_new import DGCNN_semseg
from models.attention import SelfAttention, QGPA, O2S
from models.gmmn import GMMNnetwork


class BaseLearner(nn.Module):
    """The class for inner loop."""
    def __init__(self, in_channels, params):
        super(BaseLearner, self).__init__()

        self.num_convs = len(params)
        self.convs = nn.ModuleList()

        for i in range(self.num_convs):
            if i == 0:
                in_dim = in_channels
            else:
                in_dim = params[i-1]
            self.convs.append(nn.Sequential(
                              nn.Conv1d(in_dim, params[i], 1),
                              nn.BatchNorm1d(params[i])))

    def forward(self, x):
        for i in range(self.num_convs):
            x = self.convs[i](x)
            if i != self.num_convs-1:
                x = F.relu(x)
        return x


class ProtoNetAlignQGPASR(nn.Module):
    def __init__(self, args):
        super(ProtoNetAlignQGPASR, self).__init__()
        self.n_way = args.n_way
        self.k_shot = args.k_shot
        self.dist_method = 'cosine'
        if args.use_mahalanobis:
            self.dist = 'mahalanobis'#'cosine' or 'mahalanobis'
        else:
            self.dist = 'cosine'
        self.in_channels = args.pc_in_dim
        self.n_points = args.pc_npts
        self.use_attention = args.use_attention
        self.use_align = args.use_align
        self.use_linear_proj = args.use_linear_proj
        self.use_supervise_prototype = args.use_supervise_prototype
        self.mdp = args.use_multiple_discriminative_prototypes
        self.obj = args.obj
        self.o2s = args.use_o2s
        self.mode = args.phase
        if args.use_high_dgcnn:
            self.encoder = DGCNN_semseg(args.edgeconv_widths, args.dgcnn_mlp_widths, args.pc_in_dim, k=args.dgcnn_k, return_edgeconvs=True)
            self.obj_encoder = DGCNN_semseg(args.edgeconv_widths, args.dgcnn_mlp_widths, args.pc_in_dim, k=args.dgcnn_k, return_edgeconvs=True)
        else:
            self.encoder = DGCNN(args.edgeconv_widths, args.dgcnn_mlp_widths, args.pc_in_dim, k=args.dgcnn_k, return_edgeconvs=True)
            self.obj_encoder = DGCNN(args.edgeconv_widths, args.dgcnn_mlp_widths, args.pc_in_dim, k=args.dgcnn_k, return_edgeconvs=True)
        self.base_learner = BaseLearner(args.dgcnn_mlp_widths[-1], args.base_widths)
        self.obj_base_learner = BaseLearner(args.dgcnn_mlp_widths[-1], args.base_widths)

        for p in self.encoder.parameters():
            p.requires_grad=False

        if self.use_attention:
            self.att_learner = SelfAttention(args.dgcnn_mlp_widths[-1], args.output_dim)
            self.obj_att_learner = SelfAttention(args.dgcnn_mlp_widths[-1], args.output_dim)
        else:
            self.linear_mapper = nn.Conv1d(args.dgcnn_mlp_widths[-1], args.output_dim, 1, bias=False)
            self.obj_linear_mapper = nn.Conv1d(args.dgcnn_mlp_widths[-1], args.output_dim, 1, bias=False)

        if self.use_linear_proj:
            self.conv_1 = nn.Sequential(nn.Conv1d(args.train_dim, args.train_dim, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(args.train_dim),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.use_transformer = args.use_transformer

        self.transformer = QGPA()
        if self.o2s:
            self.o2s_module = O2S()

        self.n_classes = self.n_way + 1

        if args.dataset == 'scannet':
            self.base_classes = 10
        elif args.dataset == 's3dis':
            self.base_classes = 6
        elif args.dataset == 'scannet_modelnet':
            self.base_classes = 7
        elif args.dataset == 'scannet_sunrgbd':
            self.base_classes = 8
        elif args.dataset == 's3dis_modelnet':
            self.base_classes = 8
        elif args.dataset == 's3dis_sunrgbd':
            self.base_classes = 6

        in_dim = args.dgcnn_mlp_widths[-1]
        for edgeconv_width in args.edgeconv_widths:
            in_dim += edgeconv_width[-1]
        # self.segmenter = nn.Sequential(
        #                     nn.Conv1d(in_dim, 256, 1, bias=False),
        #                     nn.BatchNorm1d(256),
        #                     nn.LeakyReLU(0.2),
        #                     nn.Conv1d(256, 128, 1),
        #                     nn.BatchNorm1d(128),
        #                     nn.LeakyReLU(0.2),
        #                     nn.Dropout(0.3),
        #                     nn.Conv1d(128, self.base_classes+1, 1)
        #                  )
        # self.threshold_l = 1/self.n_way - 0.1
        # self.threshold_h = 1/self.n_way + 0.1

    def forward(self, support_x, query_x, query_y, mode):
        """
        Args:
            support_x: support point clouds with shape (n_way, k_shot, in_channels, num_points) [2, 9, 2048]
            support_y: support masks (foreground) with shape (n_way, k_shot, num_points) [2, 1, 2048]
            query_x: query point clouds with shape (n_queries, in_channels, num_points) [2, 9, 2048]
            query_y: query labels with shape (n_queries, num_points), each point \in {0,..., n_way} [2, 2048]
        Return:
            query_pred: query point clouds predicted similarity, shape: (n_queries, n_way+1, num_points)
        """
        support_x = support_x.view(self.n_way * self.k_shot, self.in_channels, self.n_points)
        query_x = query_x.view(-1, self.in_channels, self.n_points)
        query_y = query_y.view(-1, self.n_points)
        if self.obj:
            support_feat, _ = self.get_obj_Features(support_x)
        else:
            support_feat, _ = self.getFeatures(support_x)
        support_feat = support_feat.view(self.n_way, self.k_shot, -1, self.n_points)
        query_feat, xyz = self.getFeatures(query_x)  # (n_queries, feat_dim, num_points)

        # refine_support_feat = self.o2s_module(query_feat, support_feat)
        # refine_support_feat = refine_support_feat.view(-1, self.n_way, self.k_shot, support_feat.shape[2], self.n_points)#(n_query, n_way, k_shot, dim, point)
        # prototypes, covs = self.get_obj_Prototypes(refine_support_feat)

        #prototypes, covs = self.get_obj_Prototypes(support_feat.unsqueeze(0).repeat(query_y.shape[0],1,1,1,1))
        #print(prototypes.shape, covs.shape)



        fg_mask = torch.ones((self.n_way, self.k_shot, self.n_points)).cuda()
        bg_mask = torch.logical_not(fg_mask).cuda()
        support_mask_feat = self.getMaskedFeatures(support_feat, fg_mask)
        prototypes = [support_mask_feat[way, ...].sum(dim=0) / self.k_shot for way in range(self.n_way)]
        #prototypes = self.get_obj_Prototypes(support_mask_feat)
        bg_prototypes = torch.zeros(prototypes[0].shape).cuda()
        prototypes = [bg_prototypes]+prototypes
        prototypes = torch.stack(prototypes, dim=0)
        # print(prototypes.shape) #[n_way+1, feat_dim]

        '''
        fg_mask = torch.ones((self.n_way, self.k_shot, self.n_points)).cuda()
        bg_mask = torch.logical_not(fg_mask).cuda()
        #support_mask_feat = self.getMaskedFeatures(support_feat, fg_mask)

        #prototypes, covs = self.get_obj_Prototypes(support_feat) #[n_way, feat_dim] [n_way, feat_dim*feat_dim]
        prototypes = self.get_Multiple_obj_Prototypes(support_feat)
        mean_prototypes = [prototype.mean(0) for prototype in prototypes]
        #bg_prototypes, bg_cov = self.get_bg_Prototypes(query_feat, prototypes)
        bg_prototypes = torch.zeros(prototypes[0].shape[1]).cuda()
        #bg_cov = torch.eye(prototypes[0].shape[0]).cuda()
        # bg_labels = torch.zeros(1,self.n_classes).cuda()
        # bg_labels[0,0] = 1
        #prototypes = [bg_prototypes.unsqueeze(0)] + prototypes
        mean_prototypes = [bg_prototypes] + mean_prototypes
        mean_prototypes = torch.stack(mean_prototypes, dim=0)
        prototypes = torch.cat(prototypes, dim=0) #[n_way*n_shot, feat_dim]

        # covs = [bg_cov] + covs
        # covs = torch.stack(covs,  dim=0) #[n_way+1, feat_dim, feat_dim]
        #print(len(prototypes), prototypes[0].shape, len(covs), covs[0].shape)
        #print(prototypes[0], prototypes[1])
        # labels = [bg_labels] + labels
        # labels = torch.cat(labels, dim=0) #[n_way*k_shot+1, n_classes]
        '''
        multiple_discriminative_loss = 0
        if self.mdp:
            multiple_discriminative_loss = self.multiple_discriminative_Loss(prototypes, support_feat)

        self_regulize_loss = 0
        if self.use_supervise_prototype:
            self_regulize_loss = self.sup_regulize_Loss(prototypes, support_feat, fg_mask, bg_mask)

        if self.o2s:
            # prototypes_all = prototypes.unsqueeze(0).repeat(query_feat.shape[0], 1, 1) #[n_query, n_way*k_shot, feat_dim]
            # #support_feat_ = support_feat.mean(1)
            # prototypes_all_post = self.o2s_module(query_feat, support_feat, prototypes_all) #[n_query, n_way*k_shot, out_channel]
            # prototypes_all_post = prototypes_all_post.reshape(query_feat.shape[0], self.n_way, self.k_shot, -1)

            if self.dist == 'cosine':
                prototypes_all_post = prototypes_all_post.mean(2).squeeze(2) #[n_query, n_way, out_channel]
                bg_prototypes_all_post = torch.zeros(prototypes_all_post[:,0,:].shape).unsqueeze(1).cuda()
                prototypes_all_post = torch.cat([prototypes_all_post,bg_prototypes_all_post], dim=1)
                query_pred = self.cosine_distance(prototypes_all_post, query_feat)  # [n_query, n_way+1, num_points]
            elif self.dist == 'mahalanobis':
                query_pred = self.mahalanobis_distance(prototypes, covs, query_feat)
            # prototypes_new = torch.chunk(prototypes_all_post, prototypes_all_post.shape[1], dim=1)
            # similarity = [self.calculateSimilarity_trans(query_feat, prototype.squeeze(1), self.dist_method) for prototype in prototypes_new]
            # query_pred = torch.stack(similarity, dim=1) #[n_query, n_way+1, num_points]
            #query_pred = F.softmax(query_pred, dim=1)
            # query_pred_max_score = query_pred.max(1)[0]
            # wrong_id = query_pred_max_score <= self.threshold_h
            # bg_label = torch.zeros(self.n_classes).cuda()
            # bg_label[0] = 1
            # query_pred.transpose(1,2)[wrong_id] = bg_label
            #print(query_pred.shape)

            #query_pred = torch.matmul(query_pred.transpose(1,2), labels).transpose(1,2) #[n_query, n_way+1, num_points]
        elif self.use_transformer:
            prototypes_all = prototypes.unsqueeze(0).repeat(query_feat.shape[0], 1, 1)
            support_feat_ = support_feat.mean(1)
            #support_feat_ = support_feat.view(support_feat.shape[0]*support_feat.shape[1], support_feat.shape[2], support_feat.shape[3])
            #print(query_feat.shape, support_feat_.shape, prototypes_all.shape) #[n_query, feat_dim, num_point], [n_way, feat_dim, num_point], [n_query, n_way+1, feat_dim]
            prototypes_all_post = self.transformer(query_feat, support_feat_, prototypes_all) #[n_query, n_way+1, feat_dim]
            if self.dist == 'cosine':
                query_pred = self.cosine_distance(prototypes_all_post, query_feat)  # [n_query, n_way+1, num_points]
            elif self.dist == 'mahalanobis':
                query_pred = self.mahalanobis_distance(prototypes_all_post, covs, query_feat)
            #print(query_pred)
            # prototypes_new = torch.chunk(prototypes_all_post, prototypes_all_post.shape[1], dim=1)
            # similarity = [self.calculateSimilarity_trans(query_feat, prototype.squeeze(1), self.dist_method) for prototype in prototypes_new]
            # query_pred = torch.stack(similarity, dim=1) #[n_query, n_way+1, num_points]
            #query_pred = F.softmax(query_pred, dim=1)
            # query_pred_max_score = query_pred.max(1)[0]
            # wrong_id = query_pred_max_score <= self.threshold_h
            # bg_label = torch.zeros(self.n_classes).cuda()
            # bg_label[0] = 1
            # query_pred.transpose(1, 2)[wrong_id] = bg_label
            #query_pred = torch.matmul(query_pred.transpose(1, 2), labels).transpose(1, 2)  # [n_query, n_way+1, num_points]
        else:
            prototypes_all_post = prototypes.unsqueeze(0).repeat(query_feat.shape[0], 1, 1)
            if self.dist == 'cosine':
                query_pred = self.cosine_distance(prototypes_all_post, query_feat)  # [n_query, n_way+1, num_points]
            elif self.dist == 'mahalanobis':
                query_pred = self.mahalanobis_distance(prototypes_all_post, covs, query_feat)

            # similarity = [self.calculateSimilarity(query_feat, prototype, self.dist_method) for prototype in prototypes]
            # query_pred = torch.stack(similarity, dim=1)
            # query_pred = torch.matmul(query_pred.transpose(1, 2), labels).transpose(1, 2)

        if self.mode == 'genmptieval':
            query_basepred = self.get_base_Features(query_x) #(n_query, base_class, num_point)
            query_new_pred = query_pred.max(1)[1].view(-1) #(num_fg_query, 0-nway)
            query_basepred = query_basepred.max(1)[1].view(-1)#(num_fg_query, 0-train_class

            #print(query_basepred.shape, query_new_pred.shape)
            gen_query_pred = torch.zeros((query_basepred.shape[0], self.n_classes+self.base_classes)).cuda()
            for idx in range(query_basepred.shape[0]):
                if query_new_pred[idx] == 0:
                    gen_query_pred[idx][query_basepred[idx] + 2] = 1
                else:
                    gen_query_pred[idx][query_new_pred[idx]] = 1

            query_pred = gen_query_pred.view(query_y.shape[0], query_y.shape[1], -1).transpose(1, 2)
        else:
            query_pred = query_pred
        # print(query_pred.shape, query_y.shape)
        loss = F.cross_entropy(query_pred, query_y)



        align_loss = 0
        # if self.use_align:
        #     align_loss_epi = self.alignLoss_trans(query_feat, query_pred, support_feat, fg_mask, bg_mask)
        #     align_loss += align_loss_epi

        #prototypes_all_post = prototypes_all_post.clone().detach()

        #print(loss, align_loss, multiple_discriminative_loss)
        return query_pred, loss + align_loss + multiple_discriminative_loss+self_regulize_loss , prototypes

    def forward_test_semantic(self, support_x, support_y, query_x, query_y, embeddings=None):
        """
        Args:
            support_x: support point clouds with shape (n_way, k_shot, in_channels, num_points) [2, 9, 2048]
            support_y: support masks (foreground) with shape (n_way, k_shot, num_points) [2, 1, 2048]
            query_x: query point clouds with shape (n_queries, in_channels, num_points) [2, 9, 2048]
            query_y: query labels with shape (n_queries, num_points), each point \in {0,..., n_way} [2, 2048]
        Return:
            query_pred: query point clouds predicted similarity, shape: (n_queries, n_way+1, num_points)
        """

        query_feat, xyz = self.getFeatures(query_x)

        # prototype learning
        if self.use_transformer:
            prototypes_all_post = embeddings
            prototypes_new = torch.chunk(prototypes_all_post, prototypes_all_post.shape[1], dim=1)
            similarity = [self.calculateSimilarity_trans(query_feat, prototype.squeeze(1), self.dist_method) for prototype in prototypes_new]
            query_pred = torch.stack(similarity, dim=1)
            loss = self.computeCrossEntropyLoss(query_pred, query_y)

        return query_pred, loss

    def multiple_discriminative_Loss(self, prototypes, support_feats):
        '''
        Compute the loss of multiple prototypes self-discriminative supervise
        :param prototypes: (n_way+1, feat_dim)
        :param support_feats: (n_way, k_shot, feat_dim, n_points)
        :return: loss results
        '''
        loss = 0
        for way in range(self.n_way):
            for shot in range(self.k_shot):
                obj_feats = support_feats[way, shot].unsqueeze(0)
                supp_dist = [self.calculateSimilarity(obj_feats, prototype, self.dist_method) for prototype in prototypes]
                supp_pred = torch.stack(supp_dist, dim=1).squeeze(0).transpose(0,1) #[num_point, n_way+1]
                #supp_pred = torch.matmul(supp_pred.transpose(0,1), labels) #[num_point, n_way+1]

                supp_label = torch.zeros(supp_pred.shape[0], device=supp_pred.device).long()
                supp_label[:] = way + 1

                loss = loss + F.cross_entropy(supp_pred, supp_label) / self.n_way / self.k_shot
        return loss


    def sup_regulize_Loss(self, prototype_supp, supp_fts, fore_mask, back_mask):
        """
        Compute the loss for the prototype suppoort self alignment branch

        Args:
            prototypes: embedding features for query images
                expect shape: N x C x num_points
            supp_fts: embedding features for support images
                expect shape: (Wa x Shot) x C x num_points
            fore_mask: foreground masks for support images
                expect shape: (way x shot) x num_points
            back_mask: background masks for support images
                expect shape: (way x shot) x num_points
        """
        n_ways, n_shots = self.n_way, self.k_shot

        # Compute the support loss
        loss = 0
        for way in range(n_ways):
            prototypes = [prototype_supp[0], prototype_supp[way + 1]]
            # if self.contrast:
            #     prototypes = torch.zeros(prototype_supp.shape).cuda()
            #     prototypes.copy_(prototype_supp)
            #     tmp = prototypes[way+1]
            #     prototypes[way+1] = prototypes[1]
            #     prototypes[1] = tmp
            # else:
            #     prototypes = [prototype_supp[0], prototype_supp[way + 1]]

            for shot in range(n_shots):
                img_fts = supp_fts[way, shot].unsqueeze(0) #[1, C, num_point]

                supp_dist = [self.calculateSimilarity(img_fts, prototype, self.dist_method) for prototype in prototypes]
                supp_pred = torch.stack(supp_dist, dim=1)
                #print(supp_pred.shape, fore_mask.shape)
                # Construct the support Ground-Truth segmentation
                supp_label = torch.full_like(fore_mask[way, shot], 255, device=img_fts.device).long()

                supp_label[fore_mask[way, shot] == 1] = 1
                supp_label[back_mask[way, shot] == 1] = 0
                # Compute Loss

                loss = loss + F.cross_entropy(supp_pred, supp_label.unsqueeze(0), ignore_index=255) / n_shots / n_ways
        return loss

    def getFeatures(self, x):
        """
        Forward the input data to network and generate features
        :param x: input data with shape (B, C_in, L)
        :return: features with shape (B, C_out, L)
        """
        if self.use_attention:
            feat_level1, feat_level2, xyz = self.encoder(x)
            feat_level3 = self.base_learner(feat_level2)
            att_feat = self.att_learner(feat_level2)
            if self.use_linear_proj:
                return self.conv_1(torch.cat((feat_level1[0], feat_level1[1], feat_level1[2], att_feat, feat_level3), dim=1)), xyz
            else:
                return torch.cat((feat_level1[0], feat_level1[1], feat_level1[2], att_feat, feat_level3), dim=1), xyz
        else:
            # return self.base_learner(self.encoder(x))
            feat_level1, feat_level2 = self.encoder(x)
            feat_level3 = self.base_learner(feat_level2)
            map_feat = self.linear_mapper(feat_level2)
            return torch.cat((feat_level1, map_feat, feat_level3), dim=1)

    def get_base_Features(self, pc):
        num_points = pc.shape[2]
        edgeconv_feats, point_feat, _ = self.encoder(pc)
        global_feat = point_feat.max(dim=-1, keepdim=True)[0]
        edgeconv_feats.append(global_feat.expand(-1, -1, num_points))
        pc_feat = torch.cat(edgeconv_feats, dim=1)

        logits = self.segmenter(pc_feat)
        return logits

    def get_obj_Features(self, x):
        """
        Forward the input data to network and generate features
        :param x: input data with shape (B, C_in, L)
        :return: features with shape (B, C_out, L)
        """
        # feat_level1, feat_level2 = self.encoder(x)
        # feat_level3 = self.base_learner(feat_level2)
        # return torch.cat((feat_level1, feat_level3), dim=1)
        if self.use_attention:
            feat_level1, feat_level2, xyz = self.obj_encoder(x)
            feat_level3 = self.obj_base_learner(feat_level2)
            att_feat = self.obj_att_learner(feat_level2)
            if self.use_linear_proj:
                return self.conv_1(torch.cat((feat_level1[0], feat_level1[1], feat_level1[2], att_feat, feat_level3), dim=1)), xyz
            else:
                return torch.cat((feat_level1[0], feat_level1[1], feat_level1[2], att_feat, feat_level3), dim=1), xyz
            #return torch.cat((feat_level1, att_feat, feat_level3), dim=1)
        else:
            # return self.base_learner(self.encoder(x))
            feat_level1, feat_level2 = self.obj_encoder(x)
            feat_level3 = self.obj_base_learner(feat_level2)
            map_feat = self.obj_linear_mapper(feat_level2)
            return torch.cat((feat_level1, map_feat, feat_level3), dim=1)

    def getMaskedFeatures(self, feat, mask):
        """
        Extract foreground and background features via masked average pooling

        Args:
            feat: input features, shape: (n_way, k_shot, feat_dim, num_points)
            mask: binary mask, shape: (n_way, k_shot, num_points)
        Return:
            masked_feat: masked features, shape: (n_way, k_shot, feat_dim)
        """
        mask = mask.unsqueeze(2)
        masked_feat = torch.sum(feat * mask, dim=3) / (mask.sum(dim=3) + 1e-5)
        return masked_feat

    def getPrototype(self, fg_feat, bg_feat):
        """
        Average the features to obtain the prototype

        Args:
            fg_feat: foreground features for each way/shot, shape: (n_way, k_shot, feat_dim)
            bg_feat: background features for each way/shot, shape: (n_way, k_shot, feat_dim)
        Returns:
            fg_prototypes: a list of n_way foreground prototypes, each prototype is a vector with shape (feat_dim,)
            bg_prototype: background prototype, a vector with shape (feat_dim,)
        """
        fg_prototypes = [fg_feat[way, ...].sum(dim=0) / self.k_shot for way in range(self.n_way)]
        bg_prototype = bg_feat.sum(dim=(0,1)) / (self.n_way * self.k_shot)
        return fg_prototypes, bg_prototype

    def get_obj_Prototypes(self, support_feats):
        '''
        :param support_feats: (n_query, n_way, k_shot, feat_dim, n_points)
        :return: prototypes: (n_way, feat_dim)
        '''
        prototypes = []
        covs = []

        for support_feat in support_feats:
            #print(support_feat.shape)
            support_feat = support_feat.view(support_feat.shape[0], support_feat.shape[1]*support_feat.shape[3], support_feat.shape[2]) #[n_way, k_shot*n_points, feat_dim]
            support_cov = [self.compute_covariance_matrix(support_feat[way].squeeze(0)) for way in range(self.n_way)]
            support_prototypes = [support_feat[way, ...].sum(dim=0) / (self.k_shot*self.n_points) for way in range(self.n_way)]
            bg_prototypes = torch.zeros(support_prototypes[0].shape).cuda()
            bg_cov = torch.eye(support_prototypes[0].shape[0]).cuda()
            prototype = [bg_prototypes] + support_prototypes
            cov = [bg_cov] + support_cov
            prototype = torch.stack(prototype, dim=0)
            cov = torch.stack(cov, dim=0)
            prototypes.append(prototype)
            covs.append(cov)
        prototypes = torch.stack(prototypes, dim=0)
        covs = torch.stack(covs, dim=0)

        return prototypes, covs

    def get_Multiple_obj_Prototypes(self, support_feat):
        '''
        :param support_feat: (n_way, k_shot, feat_dim, n_points)
        :return: prototypes: (n_way * k_shot, feat_dim)
                class_labels: (n_way * k_shot, n_classes)
        '''
        support_prototypes = []
        #labels = []
        for i in range(self.n_way):
            support_prototype = [support_feat[i, shot, ...].sum(dim=1).unsqueeze(0) / self.n_points for shot in range(self.k_shot)]
            support_prototype = torch.cat(support_prototype, dim=0)
            support_prototypes.append(support_prototype)

            # construct label matrix
            # class_labels = torch.zeros(support_prototype.shape[0], self.n_classes).cuda()
            # class_labels[:, i + 1] = 1
            # labels.append(class_labels)
        #support_prototypes = torch.cat(support_prototypes, dim=0)
        #labels = torch.cat(labels, dim=0).cuda()
        return support_prototypes

    def get_bg_Prototypes(self, query, prototypes):
        '''
        :param query: (n_query, feat_dim, n_points)
        :param prototypes: (n_way, feat_dim)
        :return: (feat_dim,)
        '''
        query_feat = query.view(query.shape[0]*query.shape[2], query.shape[1])
        #prototypes = torch.cat(prototypes, dim=0)
        supp_dist = [self.calculateSimilarity(query, prototype, self.dist_method) for prototype in prototypes]
        supp_pred = torch.stack(supp_dist, dim=1) #[n_query, n_way, n_points]
        #print(supp_pred.shape)
        supp_pred = F.softmax(supp_pred, dim=1).max(1)[0].view(-1)
        bg_id = supp_pred < self.threshold_l
        bg_prototype = query_feat[bg_id].sum(dim=0) / (bg_id.sum(dim=0)+1e-5)
        bg_cov = self.compute_covariance_matrix(query_feat[bg_id])
        return bg_prototype, bg_cov



    def calculateSimilarity(self, feat,  prototype, method='cosine', scaler=10):
        """
        Calculate the Similarity between query point-level features and prototypes

        Args:
            feat: input query point-level features
                  shape: (n_queries, feat_dim, num_points)
            prototype: prototype of one semantic class
                       shape: (feat_dim,)
            method: 'cosine' or 'euclidean', different ways to calculate similarity
            scaler: used when 'cosine' distance is computed.
                    By multiplying the factor with cosine distance can achieve comparable performance
                    as using squared Euclidean distance (refer to PANet [ICCV2019])
        Return:
            similarity: similarity between query point to prototype
                        shape: (n_queries, 1, num_points)
        """
        if method == 'cosine':
            similarity = F.cosine_similarity(feat, prototype[None, ..., None], dim=1) * scaler
        elif method == 'euclidean':
            similarity = - F.pairwise_distance(feat, prototype[None, ..., None], p=2)**2
        else:
            raise NotImplementedError('Error! Distance computation method (%s) is unknown!' %method)
        return similarity

    def calculateSimilarity_trans(self, feat,  prototype, method='cosine', scaler=10):
        """
        Calculate the Similarity between query point-level features and prototypes

        Args:
            feat: input query point-level features
                  shape: (n_queries, feat_dim, num_points)
            prototype: prototype of one semantic class
                       shape: (feat_dim,)
            method: 'cosine' or 'euclidean', different ways to calculate similarity
            scaler: used when 'cosine' distance is computed.
                    By multiplying the factor with cosine distance can achieve comparable performance
                    as using squared Euclidean distance (refer to PANet [ICCV2019])
        Return:
            similarity: similarity between query point to prototype
                        shape: (n_queries, 1, num_points)
        """
        if method == 'cosine':
            similarity = F.cosine_similarity(feat, prototype[..., None], dim=1) * scaler
        elif method == 'euclidean':
            similarity = - F.pairwise_distance(feat, prototype[..., None], p=2)**2
        else:
            raise NotImplementedError('Error! Distance computation method (%s) is unknown!' %method)
        return similarity

    def computeCrossEntropyLoss(self, query_logits, query_labels):
        """ Calculate the CrossEntropy Loss for query set
        """
        return F.cross_entropy(query_logits, query_labels)

    def alignLoss_trans(self, qry_fts, pred, supp_fts, fore_mask, back_mask):
        """
        Compute the loss for the prototype alignment branch

        Args:
            qry_fts: embedding features for query images
                expect shape: N x C x num_points
            pred: predicted segmentation score
                expect shape: N x (1 + Wa) x num_points
            supp_fts: embedding features for support images
                expect shape: (Wa x Shot) x C x num_points
            fore_mask: foreground masks for support images
                expect shape: (way x shot) x num_points
            back_mask: background masks for support images
                expect shape: (way x shot) x num_points
        """
        n_ways, n_shots = self.n_way, self.k_shot

        # Mask and get query prototype
        pred_mask = pred.argmax(dim=1, keepdim=True)  # N x 1 x H' x W'
        binary_masks = [pred_mask == i for i in range(1 + n_ways)]
        skip_ways = [i for i in range(n_ways) if binary_masks[i + 1].sum() == 0]
        pred_mask = torch.stack(binary_masks, dim=1).float()  # N x (1 + Wa) x 1 x H' x W'

        qry_prototypes = torch.sum(qry_fts.unsqueeze(1) * pred_mask, dim=(0, 3)) / (pred_mask.sum(dim=(0, 3)) + 1e-5)
        # Compute the support loss
        loss = 0
        for way in range(n_ways):
            if way in skip_ways:
                continue
            # Get the query prototypes
            prototypes = [qry_prototypes[0], qry_prototypes[way + 1]]
            for shot in range(n_shots):
                img_fts = supp_fts[way, shot].unsqueeze(0)
                prototypes_all = torch.stack(prototypes, dim=0).unsqueeze(0)
                prototypes_all_post = self.transformer(img_fts, qry_fts.mean(0).unsqueeze(0), prototypes_all)
                prototypes_new = [prototypes_all_post[0, 0], prototypes_all_post[0, 1]]

                supp_dist = [self.calculateSimilarity(img_fts, prototype, self.dist_method) for prototype in prototypes_new]
                supp_pred = torch.stack(supp_dist, dim=1)
                # Construct the support Ground-Truth segmentation
                supp_label = torch.full_like(fore_mask[way, shot], 255, device=img_fts.device).long()

                supp_label[fore_mask[way, shot] == 1] = 1
                supp_label[back_mask[way, shot] == 1] = 0
                # Compute Loss

                loss = loss + F.cross_entropy(supp_pred, supp_label.unsqueeze(0), ignore_index=255) / n_shots / n_ways
        return loss

    def mahalanobis_distance(self, prototypes, support_covs, query_feat):
        '''
        计算马氏距离
        :param prototypes: n_query, n_way, feat_dim
        :param query_feat: n_queries, feat_dim, num_points
        :param support_covs: n_query, n_way, feat_dim, feat_dim
        :return: n_query, n_way+1, num_points
        :param x: nxfeat_dim
        :param mu: 1xfeat_dim
        :param cov: feat_dimxfeat_dim
        :return: nx1
        '''
        similarity = []
        # prototypes_new = torch.chunk(prototypes_all_post, prototypes_all_post.shape[1], dim=1)
        # bg_pred_score = self.calculateSimilarity_trans(query_feat, prototypes_new[0].squeeze(1), self.dist_method) #[n_query, 1, numpoints]
        #bg_pred_score = torch.zeros((query_feat.shape[0],1,query_feat.shape[2])).cuda()

        for i in range(query_feat.shape[0]):
            prototype = prototypes[i] #[n_way, feat_dim]
            covs = support_covs[i] #[n_way, feat_dim, feat_dim]
            x = query_feat[i].view(self.n_points, -1) #[num_points, feat_dim]
            dists = []
            for j in range(self.n_way+1):
                mu = prototype[j] #[1, feat_dim]
                cov = covs[j] #[feat_dim, feat_dim]
                diff = x - mu #[num_points, feat_dim]
                if torch.det(cov) != 0:
                    inv_cov = torch.inverse(cov)
                else:
                    inv_cov = torch.pinverse(cov)

                dist = - torch.sqrt(torch.matmul(torch.matmul(diff.unsqueeze(1), inv_cov), diff.unsqueeze(2)).squeeze()+1) #[num_point, 1]
                #print(j, torch.det(cov), dist)
                dists.append(dist)
            dists = torch.stack(dists, dim=1) #[num_point, n_way+1]
            similarity.append(dists)
        similarity = torch.stack(similarity, dim=0).transpose(1,2) #[n_query, n_way+1, num_point]
        similarity = F.softmax(similarity, dim=1)
        #print(bg_pred_score.shape, similarity.shape)
        #query_pred = torch.cat((bg_pred_score,similarity), dim=1)
        #print(query_pred.shape)

        return similarity

    def cosine_distance(self, prototypes_all_post, query_feat):
        '''
        计算cosine距离
        :param prototypes_all_post: n_query, n_way+1, feat_dim
        :param query_feat: n_queries, feat_dim, num_points
        :return: n_query, n_way+1, num_points
        '''
        prototypes_new = torch.chunk(prototypes_all_post, prototypes_all_post.shape[1], dim=1)
        #print(prototypes_new[0].shape) #[n_query, 1, feat_dim]
        similarity = [self.calculateSimilarity_trans(query_feat, prototype.squeeze(1), self.dist_method) for prototype in prototypes_new]
        query_pred = torch.stack(similarity, dim=1)  # [n_query, n_way+1, num_points]
        return  query_pred

    def compute_covariance_matrix(self, x):
        # Subtract the mean from the data
        mean_x = torch.mean(x, dim=0)
        centered_x = x - mean_x

        # Compute the covariance matrix
        cov_matrix = torch.matmul(centered_x.t(), centered_x) / (x.size(0) - 1)

        # ratio
        #lam = x.shape[0] / (x.shape[0]+1)
        cov_matrix = cov_matrix + torch.eye(x.shape[1], x.shape[1]).cuda()

        return cov_matrix
