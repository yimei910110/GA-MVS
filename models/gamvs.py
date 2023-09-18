import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import modules.submodules
from modules.submodules import Get_gradient_depth
from modules.submodules import StageFeatExtNet, CostRegNet, CostRegNetBN, PixelwiseNet, FeatureFetcher, \
    get_pixel_grids, depth_regression, Get_Atten_canny


class UCSNet(nn.Module):

    def __init__(self, cfg):
        super(UCSNet, self).__init__()
        self.cfg = cfg
        self.stage_num = cfg["model"]["stage_num"]
        self.output_channels = cfg["model"]["output_channels"]
        self.depth2stage = cfg["model"]["stage_info"]["depth2stage"]
        self.group_nums = cfg["model"]["group_nums"]
        self.feat_name = cfg["model"].get("feat_name", "StageFeatExtNet")
        # StageFeatExtNet
        self.feat_class = getattr(modules.submodules, self.feat_name)
        self.img_feature = self.feat_class(base_channels=8, stage_num=self.stage_num,
                                           output_channels=self.output_channels)
        self.feature_fetcher = FeatureFetcher()
        self.attention_map= Get_Atten_canny()

        self.lamb=cfg["model"]["lamb"]
        self.softmax = nn.LogSoftmax(dim=1)
        if cfg["model"].get("use_3dbn", True):
            self.cost_network = nn.ModuleDict({
                str(i): CostRegNetBN(self.group_nums[i], 8) for i in range(self.stage_num)
            })
        else:
            self.cost_network = nn.ModuleDict({
                str(i): CostRegNet(self.group_nums[i], 8) for i in range(self.stage_num)
            })

        self.view_weight_nets = nn.ModuleDict({
            str(i): PixelwiseNet(self.group_nums[i]) for i in range(self.stage_num)
        })

    def sequential_wrapping(self, features, current_depths, feature_map_indices_grid, cam_intrinsic, cam_extrinsic,
                            stage_id):
        ref_feature = features[0]
        num_views = len(features)
        B, C, H, W = ref_feature.shape
        depth_num = current_depths.shape[1]

        group_num = self.group_nums[stage_id]
        ref_feature = ref_feature.view(B, group_num, C // group_num, H, W)

        ref_cam_intrinsic = cam_intrinsic[:, 0, :, :].clone()
        R = cam_extrinsic[:, :, :3, :3]
        t = cam_extrinsic[:, :, :3, 3].unsqueeze(-1)
        R_inv = torch.inverse(R)
        uv = torch.matmul(torch.inverse(ref_cam_intrinsic).unsqueeze(1), feature_map_indices_grid)  # (B, 1, 3, FH*FW)

        cam_points = (uv * current_depths.view(B, depth_num, 1, -1))
        world_points = torch.matmul(R_inv[:, 0:1, :, :], cam_points - t[:, 0:1, :, :]).transpose(1, 2).contiguous() \
            .view(B, 3, -1)  # (B, 3, D*FH*FW)

        num_world_points = world_points.size(-1)
        assert num_world_points == H * W * depth_num

        similarity_sum = 0.0
        pixel_wise_weight_sum = 0.0
        for src_idx in range(1, num_views):
            src_fea = torch.unsqueeze(features[src_idx], 1)
            src_cam_intrinsic = cam_intrinsic[:, src_idx:src_idx + 1]
            src_cam_extrinsic = cam_extrinsic[:, src_idx:src_idx + 1]
            warped_volume = self.feature_fetcher(src_fea, world_points, src_cam_intrinsic, src_cam_extrinsic)
            warped_volume = warped_volume.squeeze(1).view(B, C, depth_num, H, W)
            warped_volume = warped_volume.view(B, group_num, C // group_num, depth_num, H, W)
            similarity = (warped_volume * ref_feature.unsqueeze(3)).mean(2)  # B, G, D, H, W
            view_weight = self.view_weight_nets[str(stage_id)](similarity)  # B, 1, H, W

            if self.training:
                similarity_sum = similarity_sum + similarity * view_weight.unsqueeze(1)  # [B, G, Ndepth, H, W]
                pixel_wise_weight_sum = pixel_wise_weight_sum + view_weight.unsqueeze(1)  # [B,1,1,H,W]
            else:
                similarity_sum += similarity * view_weight.unsqueeze(1)
                pixel_wise_weight_sum += view_weight.unsqueeze(1)

            del warped_volume, similarity, view_weight

        similarity = similarity_sum.div_(pixel_wise_weight_sum)

        return similarity

    def forward(self, data_batch, stage_id=None,  is_train=False ,**kwargs):

        img_list = data_batch["imgs"]
        cam_params_list = data_batch["cams"][str(stage_id)]

        cam_extrinsic = cam_params_list[:, :, 0, :3, :4].clone()  # (B, V, 3, 4)
        cam_intrinsic = cam_params_list[:, :, 1, :3, :3].clone()

        num_view = img_list.shape[1]

        img_feature_maps = []
        depth_feature_maps=[]
        ref_img=img_list[:, 0, :, :, :]
        ref_canny_img=data_batch["canny_ref"]
        ref_canny_img = torch.cat((ref_canny_img,ref_img) , dim=1)
        for i in range(num_view):
            curr_img = img_list[:, i, :, :, :]
            curr_feature_maps,depth_feature_att = self.img_feature(curr_img)
            curr_feature_map=curr_feature_maps[str(stage_id)]
            img_feature_maps.append(curr_feature_map)
            depth_feature_maps.append(depth_feature_att.squeeze(1))
        depth_feature_maps1 = torch.stack(depth_feature_maps).permute(1, 0, 2, 3)

        atten = self.attention_map(ref_canny_img)[str(stage_id)].squeeze(1)
        [n1, h1, w1] = atten.data.shape
        attention_one = atten.view(-1)
        maxx = torch.max(attention_one).item()
        minn = torch.min(attention_one).item()
        att_scale = maxx - minn
        min_shape = torch.ones_like(attention_one)
        min_att = min_shape * minn
        attentionmap = attention_one - min_att
        attentionmap = torch.div(attentionmap, att_scale)
        attentionmap= attentionmap.view(n1, h1, w1)

        current_depths = data_batch["depth"]  # (B, D, H, W)

        B, C, H, W = img_feature_maps[0].shape

        feature_map_indices_grid = get_pixel_grids(H, W).view(1, 1, 3, -1).expand(B, 1, 3, -1).to(
            img_feature_maps[0].device)

        cost_img = self.sequential_wrapping(img_feature_maps, current_depths, feature_map_indices_grid,
                                            cam_intrinsic=cam_intrinsic, cam_extrinsic=cam_extrinsic, stage_id=stage_id)

        prob_volume_pre = self.cost_network[str(stage_id)](cost_img)
        prob_volume_pre = torch.squeeze(prob_volume_pre, 1)

        prob_volume = torch.exp(self.softmax(prob_volume_pre))
        depth = depth_regression(prob_volume, depth_values=current_depths)

        samp_variance = (current_depths - depth.unsqueeze(1)) ** 2
        exp_variance = self.lamb * torch.sum(samp_variance * prob_volume, dim=1, keepdim=False) ** 0.5

        num_depth = current_depths.shape[1]
        if is_train :
            return {"pred_depth": depth, "exp_var": exp_variance, "atten": attentionmap,"depth_feature_maps1":depth_feature_maps1}
        else :
            prob_volume_sum4 = 4 * F.avg_pool3d(F.pad(prob_volume.unsqueeze(1), pad=(0, 0, 0, 0, 1, 2)), (4, 1, 1),stride=1, padding=0).squeeze(1)
            depth_index = depth_regression(prob_volume, depth_values=torch.arange(num_depth, device=prob_volume.device,dtype=torch.float)).long()
            depth_index = depth_index.clamp(min=0, max=num_depth - 1)
            prob_conf = torch.gather(prob_volume_sum4, 1, depth_index.unsqueeze(1)).squeeze(1)

            return {"pred_depth": depth, "exp_var": exp_variance, "pred_prob": prob_conf}


def UCSNet_loss(preds, depth_gt, mask, atten, lamb_grad,depth_feature_maps1,source_depth_gts):
    # preds: B, H, W
    # gt_label: B, H, W
    # mask: B, H, W

    grad_loss_func = torch.nn.MSELoss()
    Grad = Get_gradient_depth().cuda()

    N_v=source_depth_gts.shape[1] # 5
    loss_feature=0
    for i in range(N_v):
        source_depth_gt= source_depth_gts[:, i, :, :].unsqueeze(1)
        source_grad_depth_gt=Grad(source_depth_gt)

        source_grad_depth_gt_sig=torch.sigmoid(source_grad_depth_gt).squeeze(1)
        depth_feature_map=depth_feature_maps1[:, i, :, :]
        loss_feature_i=grad_loss_func(depth_feature_map, source_grad_depth_gt_sig)
        loss_feature = loss_feature + loss_feature_i


    depth_grad_gt=Grad(depth_gt.unsqueeze(1)).squeeze(1)
    preds_grad=Grad(preds.unsqueeze(1)).squeeze(1)

    mask = mask > 0.0  # B, H, W

    preds_mask = preds[mask]  # N, C
    depth_gt_mask = depth_gt[mask]  # N

    atten_inv = torch.ones_like(atten) - atten
    atten_inv_mask=atten_inv[mask]
    atten_mask=atten[mask]

    lamb=8
    loss_depth = lamb * F.smooth_l1_loss(preds_mask, depth_gt_mask, reduction='mean')
    loss_depth_atten =lamb_grad* F.smooth_l1_loss(atten_inv_mask*preds_mask, atten_inv_mask*depth_gt_mask, reduction='mean')
    loss_grad = lamb_grad * F.smooth_l1_loss(atten_mask*preds_grad[mask], atten_mask*depth_grad_gt[mask], reduction='mean')

    loss=loss_depth+loss_depth_atten+loss_grad+loss_feature

    return loss,loss_depth,loss_depth_atten,loss_grad,loss_feature