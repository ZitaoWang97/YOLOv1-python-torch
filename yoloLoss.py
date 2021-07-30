import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class yoloLoss(nn.Module):
    def __init__(self, S, B, l_coord, l_noobj): # 7,2,5,0.5
        super(yoloLoss, self).__init__()
        self.S = S
        self.B = B
        self.l_coord = l_coord
        self.l_noobj = l_noobj

    def compute_iou(self, box1, box2):
        '''Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].
        Args:
          box1: (tensor) bounding boxes, sized [N,4].
          box2: (tensor) bounding boxes, sized [M,4].
        Return:
          (tensor) iou, sized [N,M].
        '''
        N = box1.size(0)
        M = box2.size(0)

        lt = torch.max(
            box1[:, :2].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:, :2].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        rb = torch.min(
            box1[:, 2:].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:, 2:].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        wh = rb - lt  # [N,M,2]
        wh[wh < 0] = 0  # clip at 0
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N,]
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M,]
        area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
        area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

        iou = inter / (area1 + area2 - inter)
        return iou

    def forward(self, pred_tensor, target_tensor):
        '''
        pred_tensor: (tensor) size(batchsize,S,S,Bx5+20=30) [batch_size,7,7,30]
        target_tensor: (tensor) size(batchsize,S,S,30)
        '''
        N = pred_tensor.size()[0] # N:batch_size
        coo_mask = target_tensor[:, :, :, 4] > 0  # size(N,7,7),7*7:49个grid，有obj的显示true
        noo_mask = target_tensor[:, :, :, 4] == 0 #   ...                   ，noobj的网格显示true
        coo_mask = coo_mask.unsqueeze(-1).expand_as(target_tensor)  # N,7,7,1 --> N,7,7,30
        noo_mask = noo_mask.unsqueeze(-1).expand_as(target_tensor)  # N,7,7,1 --> N,7,7,30

        coo_pred = pred_tensor[coo_mask].view(-1, 30)  # 把pred中对应目标检测的网格中的30个预测值取出来（7*7*N个网格中取）
        box_pred = coo_pred[:, :10].contiguous().view(-1, 5)  # [N*obj_grid*2,5]
        class_pred = coo_pred[:, 10:]  # [N*obj_grid,20]

        coo_target = target_tensor[coo_mask].view(-1, 30)  # [N*obj_grid,30]
        box_target = coo_target[:, :10].contiguous().view(-1, 5)  # [N*obj_grid*2,5]
        class_target = coo_target[:, 10:]  # [N*obj_grid,20]

        # compute not contain obj loss
        noo_pred = pred_tensor[noo_mask].view(-1, 30)  # [N*noobj_grid,30]
        noo_target = target_tensor[noo_mask].view(-1, 30)  # [N*noobj_grid,30]
        noo_pred_mask = torch.cuda.ByteTensor(noo_pred.size()).bool()  # [N*noobj_grid,30]
        noo_pred_mask.zero_()  # fill tensor with 0
        noo_pred_mask[:, 4] = 1
        noo_pred_mask[:, 9] = 1
        noo_pred_c = noo_pred[noo_pred_mask]  # noo_pred只需要计算confidence的损失,[N*noobj_grid,30] 30维向量中，pred只剩下两个confidence的预测值（i=4&9)
        noo_target_c = noo_target[noo_pred_mask] # target的confidence值为0
        nooobj_loss = F.mse_loss(noo_pred_c, noo_target_c, reduction='sum')

        # compute contain obj loss
        coo_response_mask = torch.cuda.ByteTensor(box_target.size()).bool()  # [N*obj_grid*2,5]
        coo_response_mask.zero_()
        coo_not_response_mask = torch.cuda.ByteTensor(box_target.size()).bool()  # [N*obj_grid*2,5]
        coo_not_response_mask.zero_()
        box_target_iou = torch.zeros(box_target.size()).cuda()  # [N*obj_grid*2,5]
        for i in range(0, box_target.size()[0], 2):  # choose the best iou box,i = range(0,N*obj_grid*2,2)
            box1 = box_pred[i:i + 2,:]  # [2,5]属于同一个grid
            box1_xyxy = Variable(torch.FloatTensor(box1.size()))  # [2,5]
            box1_xyxy[:, :2] = box1[:, :2] / 7. - 0.5 * box1[:, 2:4]
            box1_xyxy[:, 2:4] = box1[:, :2] / 7. + 0.5 * box1[:, 2:4]
            box2 = box_target[i].unsqueeze(0)
            box2_xyxy = Variable(torch.FloatTensor(box2.size()))
            box2_xyxy[:, :2] = box2[:, :2] / 7. - 0.5 * box2[:, 2:4]
            box2_xyxy[:, 2:4] = box2[:, :2] / 7. + 0.5 * box2[:, 2:4]
            iou = self.compute_iou(box1_xyxy[:, :4], box2_xyxy[:, :4])  # [2,1]
            max_iou, max_index = iou.max(0)  # max_index = tensor([0]) or 1
            max_index = max_index.data.cuda()

            coo_response_mask[i + max_index] = 1  # [N*obj_grid*2,5]
            coo_not_response_mask[i + 1 - max_index] = 1  # [N*obj_grid*2,5]

            #####
            # we want the confidence score to equal the
            # intersection over union (IOU) between the predicted box
            # and the ground truth
            #####
            box_target_iou[i + max_index, torch.LongTensor([4]).cuda()] = (max_iou).data.cuda()  # [N*obj_grid*2,5],赋值对应位置的IOU
        box_target_iou = Variable(box_target_iou).cuda()
        # 1.response loss
        box_pred_response = box_pred[coo_response_mask].view(-1, 5)  # [N*obj_grid,5] 对应有物体的grid的IOU较大的那个box的location信息
        box_target_response_iou = box_target_iou[coo_response_mask].view(-1, 5)  # [N*obj_grid*2,5] 实际上预测框和ground truth BB的IOU（online计算）
        box_target_response = box_target[coo_response_mask].view(-1, 5)  # [N*obj_grid*2,5] ground truth的location值（C值就不用了，因为都是1）
        contain_loss = F.mse_loss(box_pred_response[:, 4], box_target_response_iou[:, 4], reduction='sum')
        loc_loss = F.mse_loss(box_pred_response[:, :2], box_target_response[:, :2], reduction='sum') + F.mse_loss(
            torch.sqrt(box_pred_response[:, 2:4]), torch.sqrt(box_target_response[:, 2:4]), reduction='sum')
        # 2.not response loss 所以confidence loss可以分为两部分
        box_pred_not_response = box_pred[coo_not_response_mask].view(-1, 5)
        box_target_not_response = box_target[coo_not_response_mask].view(-1, 5)
        box_target_not_response[:, 4] = 0
        # not_contain_loss = F.mse_loss(box_pred_response[:,4],box_target_response[:,4],size_average=False)

        # I believe this bug is simply a typo
        not_contain_loss = F.mse_loss(box_pred_not_response[:, 4], box_target_not_response[:, 4], reduction='sum')

        # 3.class loss
        class_loss = F.mse_loss(class_pred, class_target, reduction='sum')

        return (
                           self.l_coord * loc_loss + 2 * contain_loss + not_contain_loss + self.l_noobj * nooobj_loss + class_loss) / N




