"""Self Attention Module


"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, in_channel, out_channel=None, attn_dropout=0.1):
        """
        :param in_channel: previous layer's output feature dimension
        :param out_channel: size of output vector, defaults to in_channel
        """
        super(SelfAttention, self).__init__()
        self.in_channel = in_channel

        if out_channel is not None:
            self.out_channel = out_channel
        else:
            self.out_channel = in_channel

        self.temperature = self.out_channel ** 0.5

        self.q_map = nn.Conv1d(in_channel, self.out_channel, 1, bias=False)
        self.k_map = nn.Conv1d(in_channel, self.out_channel, 1, bias=False)
        self.v_map = nn.Conv1d(in_channel, self.out_channel, 1, bias=False)

        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, x):
        """
        :param x: the feature maps from previous layer,
                      shape: (batch_size, in_channel, num_points)
        :return: y: attentioned features maps,
                        shape： (batch_size, out_channel, num_points)
        """
        q = self.q_map(x)  # (batch_size, out_channel, num_points)
        k = self.k_map(x)  # (batch_size, out_channel, num_points)
        v = self.v_map(x)  # (batch_size, out_channel, num_points)

        attn = torch.matmul(q.transpose(1,2) / self.temperature, k)

        attn = self.dropout(F.softmax(attn, dim=-1))
        y = torch.matmul(attn, v.transpose(1,2)) # (batch_size, num_points, out_channel)

        return y.transpose(1, 2)


class QGPA(nn.Module):
    def __init__(self, attn_dropout=0.1):

        super(QGPA, self).__init__()
        self.in_channel = self.out_channel = 320

        self.temperature = self.out_channel ** 0.5
        self.layer_norm = nn.LayerNorm(self.in_channel)
        proj_dim = 512
        self.q_map = nn.Conv1d(2048, proj_dim, 1, bias=False)
        self.k_map = nn.Conv1d(2048, proj_dim, 1, bias=False)

        self.v_map = nn.Linear(self.in_channel, self.out_channel)
        self.fc = nn.Conv1d(self.in_channel, self.out_channel, 1, bias=False)
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, query, support, prototype):
        '''
        :param query: (n_query, in_channel, num_point)
        :param support: (n_way*k_shot, in_channel, num_point)
        :param prototype: (n_query, n_way+1, in_channel)
        :return:
        '''
        batch, dim = query.shape[0], query.shape[1] #[n_query, dim, num_point]
        way = support.shape[0] + 1
        residual = prototype #[n_query, n_way+1, dim]
        q = self.q_map(query.transpose(1, 2))
        if len(support.shape) == 4: #[n_way, dim, num_point]
            support = support.squeeze()
        support = torch.cat([support.mean(0).unsqueeze(0), support], dim=0)#[n_way+1, dim, num_point]
        k = self.k_map(support.transpose(1, 2))
        v = self.v_map(prototype)
        #print(q.shape, k.shape, v.shape) #[n_query, proj_dim, dim], [n_way+1, proj_dim, dim], #[n_query, n_way+1, dim]
        q = q.view(q.shape[1], q.shape[2] * q.shape[0])
        k = k.view(k.shape[1], k.shape[2] * k.shape[0])

        attn = torch.matmul(q.transpose(0, 1) / self.temperature, k)
        attn = attn.reshape(batch, way, dim, dim) #[n_query, n_way+1, dim, dim]
        attn = F.softmax(attn, dim=-1)
        v = v.unsqueeze(2) #[n_query, n_way+1, 1, dim]
        output = torch.matmul(attn, v.transpose(-2, -1)).squeeze(-1).transpose(1, 2)
        output = self.dropout(self.fc(output)).transpose(1, 2)
        output = self.layer_norm(output + residual)

        return output

class O2S(nn.Module):
    def __init__(self, attn_dropout=0.1):

        super(O2S, self).__init__()
        self.in_channel = self.out_channel = 320

        self.temperature = self.out_channel ** 0.5
        self.layer_norm = nn.LayerNorm(self.in_channel)
        #self.cov_norm = nn.LayerNorm([self.out_channel, self.out_channel])

        self.q_map = nn.Conv1d(self.in_channel, self.out_channel, 1, bias=False)
        self.k_map = nn.Conv1d(self.in_channel, self.out_channel, 1, bias=False)
        self.v_map = nn.Conv1d(self.in_channel, self.out_channel, 1, bias=False)

        self.fc = nn.Conv1d(self.in_channel, self.out_channel, 1, bias=False)
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, query, support):
        '''
        :param query: (n_query, in_channel, num_point)
        :param support: (n_way, k_shot, in_channel, num_point)
        :return: refine_support (n_query, n_way, k_shot*n_point,dim)
        '''
        n_query = query.shape[0]
        n_way = support.shape[0]
        k_shot = support.shape[1]
        dim = support.shape[2]
        n_point = support.shape[3]

        proto = support.mean(1).squeeze(1)
        support = support.view(n_way*k_shot,dim,n_point) #[n_way*k_shot, dim, num_point]

        q = self.q_map(query) #[n_query, dim, point]
        k = self.k_map(proto) #[n_way, dim, point]
        v = self.v_map(support) #[n_way*k_shot, dim, point]

        q = q.view(q.shape[2], q.shape[0] * q.shape[1])  # [point, n_query * out_dim]
        k = k.view(k.shape[2], k.shape[0] * k.shape[1])  # [point, n_way * out_dim]

        attn = torch.matmul(q.transpose(0, 1) / self.temperature, k)
        attn = attn.reshape(n_query, n_way, self.out_channel,self.out_channel)  # [n_query, n_way, dim, dim]
        attn = F.softmax(attn, dim=-1)

        v_all = v.unsqueeze(0).repeat(n_query,1,1,1)
        v_all = v_all.reshape(n_query, n_way, k_shot*n_point, dim, 1)

        output = torch.matmul(attn.unsqueeze(2),v_all).squeeze(-1).view(n_query, dim, -1) #[n_query, dim, k_shot*n_point*n_way]
        output = self.dropout(self.fc(output)).transpose(1,2)
        output = self.layer_norm(output)

        return output

'''
        origin_p = prototype
        #print(covs.shape)
        #support = torch.cat([support.mean(0).unsqueeze(0), support], dim=0)#[n_way+1, dim, num_point]
        support = support.view(support.shape[0]*support.shape[1], support.shape[2], support.shape[3]) #[n_way*k_shot, dim, num_point]
        #support = torch.cat([torch.zeros(support[0].shape).unsqueeze(0).cuda(), support], dim=0) #[n_way*k_shot+1, dim, num_point]

        q = self.q_map(query)
        k = self.k_map(support)
        v = self.v_map(prototype.transpose(1, 2)) #[n_query, out_channel, k_shot*n_way]
        #covs = self.v_map(self.v_map(covs[]).transpose(1,3).transpose(1,2) #[n_query, n_way+1, out_channel, out_channel]

        q = q.view(q.shape[2], q.shape[0] * q.shape[1]) #[num_point, n_query * out_channel]
        k = k.view(k.shape[2], k.shape[0] * k.shape[1]) #[num_point, n_way+1 * out_channel]

        attn = torch.matmul(q.transpose(0,1) / self.temperature, k)
        attn = attn.reshape(n_query, support.shape[0], self.out_channel, self.out_channel) #[n_query, n_way*k_shot, out_channel, out_channel]
        attn = F.softmax(attn, dim=-1)

        v = v.unsqueeze(3) #[n_query, out_channel, n_way*k_shot+1, 1]
        output = torch.matmul(attn, v.transpose(1, 2)).squeeze(-1).transpose(1, 2)
        #covs = torch.matmul(torch.matmul(attn, covs),attn.transpose(2,3)) #[n_query, n_way+1, out_channel, out_channel]
        output = self.dropout(self.fc(output)).transpose(1, 2)
        #covs = self.dropout(self.fc(self.fc(covs.transpose(1,2)).transpose(1,3)).transpose(1,3).transpose(1,2))
        output = self.layer_norm(output + origin_p) #[n_query, n_way+1, out_channel]
        #covs = self.cov_norm(covs)
'''

class PrototypeRectification(nn.Module):
    def __init__(self, attn_dropout=0.1):

        super(PrototypeRectification, self).__init__()
        self.in_channel = self.out_channel = 320

        self.temperature = self.out_channel ** 0.5
        self.layer_norm = nn.LayerNorm(self.in_channel)
        proj_dim = 1024
        self.q_map = nn.Conv1d(2048, proj_dim, 1, bias=False)
        self.k_map = nn.Conv1d(2048, proj_dim, 1, bias=False)

        self.v_map = nn.Linear(self.in_channel, self.out_channel)
        self.fc = nn.Conv1d(self.in_channel, self.out_channel, 1, bias=False)
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, query, support, prototype):

        batch, dim = query.shape[0], query.shape[1]
        way = support.shape[0] + 1
        residual = prototype
        # print(support.shape)
        q = self.q_map(query.transpose(1, 2))
        if len(support.shape) == 4:
            support = support.squeeze()
        support = torch.cat([support.mean(0).unsqueeze(0), support], dim=0)
        
        k = self.k_map(support.transpose(1, 2))
        v = self.v_map(prototype)
        q = q.view(q.shape[1], q.shape[2] * q.shape[0])
        k = k.view(k.shape[1], k.shape[2] * k.shape[0])

        attn = torch.matmul(q.transpose(0, 1) / self.temperature, k)
        attn = attn.reshape(batch, way, dim, dim)
        attn = F.softmax(attn, dim=-1)
        v = v.unsqueeze(2)
        output = torch.matmul(attn, v.transpose(-2, -1)).squeeze(-1).transpose(1, 2)
        output = self.dropout(self.fc(output)).transpose(1, 2)
        output = self.layer_norm(output + residual)

        return output


class CrossAttention(nn.Module):
    def __init__(self, attn_dropout=0.1):

        super(CrossAttention, self).__init__()
        self.in_channel = self.out_channel = 320

        self.temperature = self.out_channel ** 0.5
        self.layer_norm = nn.LayerNorm(self.in_channel)

        self.q_map = nn.Linear(self.in_channel, self.out_channel)
        self.k_map = nn.Conv1d(self.in_channel, self.in_channel, 1, bias=False)

        self.v_map = nn.Conv1d(self.in_channel, self.in_channel, 1, bias=False)
        self.fc = nn.Linear(self.in_channel, self.out_channel)
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, query, prototype):

        residual = prototype
        q = self.q_map(prototype)  # [b, n-class, d]

        k = self.k_map(query)  # [b, d, 2048]
        v = self.v_map(query)  # [b, d, 2048]

        attn = torch.matmul(q / self.temperature, k)  # [b, n-class, 2048]
        attn = F.softmax(attn, dim=-1)
        output = torch.matmul(attn, v.transpose(-2, -1))  # [b, n-class, d]
        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output




