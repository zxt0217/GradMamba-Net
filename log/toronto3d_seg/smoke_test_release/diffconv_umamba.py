import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math
import warnings


class GradientDiffUnit(nn.Module):
    def __init__(self, in_channels, out_channels, k=16):
        super(GradientDiffUnit, self).__init__()
        self.k = k
        self.knn_chunk_size = 1024
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def knn(self, x, k):
        num_points = x.size(2)
        if num_points <= 0:
            raise ValueError("GradientDiffUnit.knn got empty point set.")
        if num_points > 1:
            k = max(1, min(int(k), num_points - 1))
        else:
            k = 1
        # Chunked KNN: avoid allocating full [B, N, N] distance matrix.
        # We process query points in blocks (default 1024) against all keys.
        x_trans = x.transpose(2, 1).contiguous()  # [B, N, C]
        x_norm = torch.sum(x_trans ** 2, dim=-1, keepdim=True)  # [B, N, 1]
        x_norm_t = x_norm.transpose(1, 2)  # [B, 1, N]

        chunk_size = max(1, min(int(self.knn_chunk_size), num_points))
        idx_chunks = []

        for start in range(0, num_points, chunk_size):
            end = min(start + chunk_size, num_points)
            query = x_trans[:, start:end, :]  # [B, M, C]
            query_norm = torch.sum(query ** 2, dim=-1, keepdim=True)  # [B, M, 1]
            inner = torch.matmul(query, x_trans.transpose(1, 2))  # [B, M, N]

            # Use negative squared distance so larger is closer (compatible with topk largest).
            pairwise_distance = -query_norm - x_norm_t + 2 * inner

            if num_points > 1:
                local_rows = torch.arange(end - start, device=x.device)
                global_cols = torch.arange(start, end, device=x.device)
                pairwise_distance[:, local_rows, global_cols] = float("-inf")

            idx_chunks.append(pairwise_distance.topk(k=k, dim=-1)[1])

        idx = torch.cat(idx_chunks, dim=1)
        return idx

    def get_graph_feature(self, x, k=20, idx=None):
        batch_size = x.size(0)
        num_points = x.size(2)
        if num_points <= 0:
            raise ValueError("GradientDiffUnit.get_graph_feature got empty point set.")
        x = x.view(batch_size, -1, num_points)
        if idx is None:
            k = max(1, min(int(k), num_points))
            idx = self.knn(x, k=k)
            k = idx.size(-1)
        else:
            k = idx.size(-1)
        device = x.device

        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)

        _, num_dims, _ = x.size()
        x = x.transpose(2, 1).contiguous()
        feature = x.view(batch_size * num_points, -1)[idx, :]
        feature = feature.view(batch_size, num_points, k, num_dims)
        x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

        feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
        return feature

    def forward(self, x):
        x_graph = self.get_graph_feature(x, k=self.k)
        feat = self.conv(x_graph)
        feat = feat.max(dim=-1)[0]
        return feat


class LGEModule(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(LGEModule, self).__init__()
        self.mid_channel = out_channel // 2
        self.input_map = nn.Conv1d(in_channel, self.mid_channel, 1)
        self.lge_unit = GradientDiffUnit(
            in_channels=self.mid_channel,
            out_channels=out_channel
        )

    def forward(self, xyz, x):
        if x is None or x.shape[1] == 0:
            x = xyz
        x = self.input_map(x)
        feat = self.lge_unit(x)
        return feat


class MVSABlock(nn.Module):

    def __init__(self, dim, num_heads=8, kernel=[(3, 1, 1), (5, 1, 1), (7, 1, 1)],
                 s=[(1, 1, 1)] * 3, pad=[(1, 0, 0), (2, 0, 0), (3, 0, 0)], k1=2, k2=3):
        super(MVSABlock, self).__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.scale = (dim // num_heads) ** -0.5

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.k1, self.k2 = k1, k2

        self.avgpool1 = nn.AvgPool3d(kernel[0], stride=s[0], padding=pad[0])
        self.avgpool2 = nn.AvgPool3d(kernel[1], stride=s[1], padding=pad[1])
        self.avgpool3 = nn.AvgPool3d(kernel[2], stride=s[2], padding=pad[2])
        self.layer_norm = nn.LayerNorm(dim)

        self.attn1 = nn.Parameter(torch.tensor([0.5]))
        self.attn2 = nn.Parameter(torch.tensor([0.5]))

    def forward(self, x, y):
        y_fused = self.avgpool1(y) + self.avgpool2(y) + self.avgpool3(y)
        B, C, D, H, W = x.shape

        y_flat = y_fused.flatten(2).transpose(1, 2)
        y_flat = self.layer_norm(y_flat)
        x_flat = x.flatten(2).transpose(1, 2)

        kv = self.kv(y_flat).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        q = self.q(x_flat).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        N_tokens = attn.shape[-1]
        k1_keep = max(1, N_tokens // self.k1)
        k2_keep = max(1, N_tokens // self.k2)
        neg_inf = torch.finfo(attn.dtype).min

        k1_idx = torch.topk(attn, k=k1_keep, dim=-1)[1]
        mask1 = torch.zeros_like(attn, dtype=torch.bool).scatter_(-1, k1_idx, True)
        attn1 = torch.softmax(attn.masked_fill(~mask1, neg_inf), dim=-1)

        k2_idx = torch.topk(attn, k=k2_keep, dim=-1)[1]
        mask2 = torch.zeros_like(attn, dtype=torch.bool).scatter_(-1, k2_idx, True)
        attn2 = torch.softmax(attn.masked_fill(~mask2, neg_inf), dim=-1)

        out = (attn1 @ v) * self.attn1 + (attn2 @ v) * self.attn2
        out = out.transpose(1, 2).reshape(B, -1, self.dim)
        out = self.proj(out)

        return rearrange(out, 'b (d h w) c -> b c d h w', d=D, h=H, w=W)


class InterChannelShuffleUnit(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, shuffle_groups=2):
        super(InterChannelShuffleUnit, self).__init__()
        self.dwc = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, stride=1,
                      padding=kernel_size // 2, groups=in_channels, bias=False),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.pwc = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.in_channels = in_channels
        self.shuffle_groups = max(1, shuffle_groups)

    def channel_shuffle(self, x, groups):
        batchsize, num_channels, length = x.size()
        if groups <= 1 or num_channels % groups != 0:
            return x
        channels_per_group = num_channels // groups
        x = x.view(batchsize, groups, channels_per_group, length)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batchsize, -1, length)
        return x

    def _resolve_groups(self, num_channels):
        preferred = math.gcd(num_channels, self.shuffle_groups)
        if preferred > 1:
            return preferred
        for g in [4, 3, 5, 2, 6, 7, 8]:
            if g < num_channels and num_channels % g == 0:
                return g
        return 1

    def forward(self, x):
        x = self.dwc(x)
        groups = self._resolve_groups(x.size(1))
        x = self.channel_shuffle(x, groups=groups)
        x = self.pwc(x)
        return x


class ICSRModule(nn.Module):

    def __init__(self, in_channel, mlp, interp_k=3):
        super(ICSRModule, self).__init__()
        self.icsr_blocks = nn.ModuleList()
        last_channel = in_channel
        self.interp_k = max(1, int(interp_k))

        for out_channel in mlp:
            # Using the renamed unit
            self.icsr_blocks.append(InterChannelShuffleUnit(last_channel, out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)

        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape
        if S <= 0:
            raise ValueError("ICSRModule got empty support points.")

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            k = min(self.interp_k, S)
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :k], idx[:, :, :k]
            dists = dists.clamp_min(0.0)
            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, k, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)

        for icsr in self.icsr_blocks:
            new_points = icsr(new_points)

        return new_points



try:
    from mamba_ssm.modules.mamba_simple import Mamba
except ImportError:
    Mamba = None


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        in_dtype = x.dtype
        x_float = x.float()
        weight = self.weight.float()
        rms = x_float.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x_float * torch.rsqrt(rms + self.eps)
        return (x_norm * weight).to(in_dtype)


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        drop_prob = float(drop_prob)
        if drop_prob < 0.0 or drop_prob >= 1.0:
            raise ValueError(f"drop_prob must be in [0, 1), got {drop_prob}")
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob <= 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        return x * random_tensor / keep_prob


class MambaBlock(nn.Module):
    def __init__(self, input_channel, depth, rms_norm, drop_path, fetch_idx, drop_out, drop_path_rate):
        super().__init__()
        self.depth = max(1, int(depth))
        self.fetch_idx = self._resolve_fetch_idx(fetch_idx, self.depth)
        norm_cls = RMSNorm if rms_norm else nn.LayerNorm

        self.mvsa_module = MVSABlock(
            dim=input_channel,
            num_heads=8,
            kernel=[(3, 1, 1), (5, 1, 1), (7, 1, 1)],
            s=[(1, 1, 1)] * 3,
            pad=[(1, 0, 0), (2, 0, 0), (3, 0, 0)]
        )
        self.mvsa_gamma = nn.Parameter(torch.tensor([0.1]))

        self.mamba_layers = nn.ModuleList([
            Mamba(
                d_model=input_channel,
                d_state=16,
                d_conv=4,
                expand=2
            ) for _ in range(self.depth)
        ])
        self.pre_norms = nn.ModuleList([norm_cls(input_channel) for _ in range(self.depth)])
        self.pos_embed = nn.Sequential(
            nn.Linear(3, input_channel // 2),
            nn.GELU(),
            nn.Linear(input_channel // 2, input_channel)
        )
        self.seq_fusion_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_channel, input_channel),
                nn.GELU(),
                nn.Linear(input_channel, input_channel)
            ) for _ in range(self.depth)
        ])
        self.dropouts = nn.ModuleList([
            nn.Dropout(float(drop_out)) if float(drop_out) > 0 else nn.Identity()
            for _ in range(self.depth)
        ])

        start_dp = max(0.0, float(drop_path))
        end_dp = max(0.0, float(drop_path_rate))
        if self.depth == 1:
            dpr = [start_dp]
        else:
            dpr = torch.linspace(start_dp, end_dp, steps=self.depth).tolist()
        self.layer_drop_paths = nn.ModuleList([
            DropPath(rate) if rate > 0 else nn.Identity()
            for rate in dpr
        ])

        self.res_mlp = nn.Sequential(
            nn.Linear(input_channel, input_channel),
            nn.GELU()
        )

    @staticmethod
    def _resolve_fetch_idx(fetch_idx, depth):
        if fetch_idx is None:
            return [depth - 1]
        if isinstance(fetch_idx, (int, float)):
            fetch_idx = [int(fetch_idx)]

        valid = []
        for idx in fetch_idx:
            try:
                idx = int(idx)
            except (TypeError, ValueError):
                continue
            if idx < 0:
                idx = depth + idx
            if 0 <= idx < depth and idx not in valid:
                valid.append(idx)
        return valid if valid else [depth - 1]

    @staticmethod
    def _is_permutation_fast(idx, target_len):
        if target_len <= 0:
            return False
        B = idx.size(0)
        counts = torch.zeros((B, target_len), dtype=torch.int32, device=idx.device)
        ones = torch.ones_like(idx, dtype=torch.int32)
        counts.scatter_add_(1, idx, ones)
        return (counts == 1).all().item()

    def forward(self, pts, series_idx_array, coords=None, prevalidated_idx=False):
        pts_trans = pts.permute(0, 2, 1)
        ori_input = pts_trans
        B, N, _ = pts_trans.shape

        pts_5d = pts.unsqueeze(-1).unsqueeze(-1)
        mvsa_out_5d = self.mvsa_module(pts_5d, pts_5d)
        mvsa_out = mvsa_out_5d.squeeze(-1).squeeze(-1).permute(0, 2, 1)

        coords_trans = None
        if coords is not None:
            coords_candidate = coords.permute(0, 2, 1)
            if coords_candidate.dim() == 3 and coords_candidate.size(0) == B and coords_candidate.size(1) == N and coords_candidate.size(2) >= 3:
                coords_trans = coords_candidate[:, :, :3]
        if coords_trans is None:
            if pts_trans.size(-1) >= 3:
                coords_trans = pts_trans[:, :, :3]
            else:
                coords_trans = pts_trans.new_zeros(B, N, 3)

        def _run_single_scan(sorted_pts, sorted_coords):
            pos = self.pos_embed(sorted_coords)
            layer_outputs = []
            x_seq = sorted_pts + pos
            for i in range(self.depth):
                layer_out = self.mamba_layers[i](self.pre_norms[i](x_seq))
                layer_out = self.seq_fusion_mlps[i](layer_out)
                layer_out = self.dropouts[i](layer_out)
                layer_out = self.layer_drop_paths[i](layer_out)
                if self.depth > 1:
                    x_seq = x_seq + layer_out
                else:
                    # Keep depth=1 behavior aligned with prior implementation.
                    x_seq = layer_out
                layer_outputs.append(x_seq)
            selected_outputs = [layer_outputs[i] for i in self.fetch_idx]
            if len(selected_outputs) == 1:
                return selected_outputs[0]
            return torch.stack(selected_outputs, dim=0).mean(dim=0)

        valid_scan_indices = []
        if series_idx_array is not None:
            idx = series_idx_array
            if idx.dim() == 2:
                idx = idx.unsqueeze(1)
            if idx.dim() == 3 and idx.size(0) == B and idx.size(2) == N:
                idx = idx.to(device=pts_trans.device, dtype=torch.long)
                for s in range(idx.size(1)):
                    idx_s = idx[:, s, :]
                    invalid_idx = ((idx_s < 0) | (idx_s >= N)).any().item()
                    if invalid_idx:
                        continue
                    if not prevalidated_idx:
                        is_perm = self._is_permutation_fast(idx_s, N)
                        if not is_perm:
                            continue
                    valid_scan_indices.append(idx_s)

        scan_outputs = []
        if valid_scan_indices:
            for idx_s in valid_scan_indices:
                idx_expand = idx_s.unsqueeze(-1).expand(-1, -1, pts_trans.size(-1))
                sorted_pts = torch.gather(pts_trans, 1, idx_expand)
                coords_expand = idx_s.unsqueeze(-1).expand(-1, -1, coords_trans.size(-1))
                sorted_coords = torch.gather(coords_trans, 1, coords_expand)
                x_mamba_s = _run_single_scan(sorted_pts, sorted_coords)

                inverse_idx = torch.argsort(idx_s, dim=1)
                inv_expand = inverse_idx.unsqueeze(-1).expand(-1, -1, x_mamba_s.size(-1))
                x_mamba_s = torch.gather(x_mamba_s, 1, inv_expand)
                scan_outputs.append(x_mamba_s)
        else:
            scan_outputs.append(_run_single_scan(pts_trans, coords_trans))

        if len(scan_outputs) == 1:
            x_mamba = scan_outputs[0]
        else:
            x_mamba = torch.stack(scan_outputs, dim=0).mean(dim=0)

        x = ori_input + self.res_mlp(x_mamba) + self.mvsa_gamma * mvsa_out
        return x.permute(0, 2, 1)


def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def query_ball_point(radius, nsample, xyz, new_xyz):
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    if nsample <= 0:
        return torch.empty((B, S, 0), dtype=torch.long, device=device)
    if N <= 0:
        raise ValueError("query_ball_point got empty source xyz points.")

    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N

    nsample_eff = min(nsample, N)
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample_eff]
    valid_mask = group_idx != N
    first_valid_pos = valid_mask.float().argmax(dim=-1, keepdim=True)
    first_valid_idx = torch.gather(group_idx, -1, first_valid_pos)
    fallback_idx = torch.zeros_like(first_valid_idx)
    first_valid_idx = torch.where(valid_mask.any(dim=-1, keepdim=True), first_valid_idx, fallback_idx)
    group_idx = torch.where(valid_mask, group_idx, first_valid_idx.expand_as(group_idx))

    if nsample_eff < nsample:
        pad = group_idx[:, :, :1].expand(-1, -1, nsample - nsample_eff)
        group_idx = torch.cat([group_idx, pad], dim=-1)
    return group_idx


class PointNetSetAbstractionMsgWithLGE(nn.Module):

    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsgWithLGE, self).__init__()
        self.npoint = npoint
        self.in_channel = in_channel
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()

        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            input_dim = 2 * in_channel + 3

            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(input_dim, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                input_dim = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

        # Start from a neutral 0.5 / 0.5 balance to avoid suppressing diff features at init.
        self.diff_feat_logit = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.center_feat_logit = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def forward(self, xyz, points, fps_index):
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
        S = self.npoint

        new_xyz = index_points(xyz, fps_index)
        new_points_ori = index_points(points, fps_index) if points is not None else None
        diff_alpha = torch.sigmoid(self.diff_feat_logit)
        center_alpha = torch.sigmoid(self.center_feat_logit)

        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)

            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            grouped_xyz = grouped_xyz / radius

            if points is not None:
                _, _, D = points.shape
                grouped_points = index_points(points, group_idx)
                center_points_expanded = new_points_ori.view(B, S, 1, D).repeat(1, 1, K, 1)

                # Learnable balance between edge-sensitive diff features and center/context features.
                diff_feat = diff_alpha * (grouped_points - center_points_expanded)
                center_feat = center_alpha * center_points_expanded

                grouped_features_cat = torch.cat([diff_feat, center_feat, grouped_xyz], dim=-1)
            else:
                zeros_feat = grouped_xyz.new_zeros(B, S, K, self.in_channel)
                grouped_features_cat = torch.cat([zeros_feat, zeros_feat, grouped_xyz], dim=-1)

            grouped_features_cat = grouped_features_cat.permute(0, 3, 2, 1)

            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_features_cat = F.relu(bn(conv(grouped_features_cat)), inplace=True)

            new_points_list.append(torch.max(grouped_features_cat, 2)[0])

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat


class PointNetSetAbstraction(nn.Module):
    def __init__(self, in_channel, mlp, group_all=True):
        super(PointNetSetAbstraction, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.group_all = group_all
        last_channel = in_channel + 3
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points):
        if not self.group_all:
            raise NotImplementedError("PointNetSetAbstraction currently supports only group_all=True.")

        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
        new_xyz = torch.zeros(B, 1, C).to(xyz.device)
        grouped_xyz = xyz.view(B, 1, N, C)

        if points is not None:
            new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
        else:
            new_points = grouped_xyz

        new_points = new_points.permute(0, 3, 2, 1)

        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)), inplace=True)

        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points


class DiffMambaNet(nn.Module):

    def __init__(
            self,
            num_classes,
            fps_sampling_nums=[512, 128, 32],
            normal_channel=False,
            strict_indices=False,
            validate_series_permutation=False,
            mamba_depth=1,
            mamba_rms_norm=False,
            mamba_drop_path=0.2,
            mamba_drop_path_rate=0.1,
            mamba_dropout=0.0,
            mamba_fetch_idx=None):
        super(DiffMambaNet, self).__init__()
        self.input_channel = 8 if normal_channel else 3
        self.use_normal = normal_channel
        self.embed_channel = 64
        self.fps_sampling_nums = fps_sampling_nums
        self.strict_indices = strict_indices
        self.validate_series_permutation = validate_series_permutation
        self._index_warning_cache = set()

        # Keep default behavior aligned with previous implementation:
        # depth=1 and no dropout/drop-path unless explicitly enabled.
        self.mamba_depth = int(mamba_depth)
        self.mamba_drop_path_rate = float(mamba_drop_path_rate)
        self.mamba_rms_norm = bool(mamba_rms_norm)
        self.mamba_drop_path = float(mamba_drop_path)
        self.mamba_dropout = float(mamba_dropout)
        self.mamba_fetch_idx = mamba_fetch_idx

        self.lge_stem = LGEModule(
            in_channel=self.input_channel,
            out_channel=self.embed_channel
        )

        self.sa_downsample_level1 = PointNetSetAbstractionMsgWithLGE(
            fps_sampling_nums[0], [0.1, 0.2, 0.4], [32, 64, 128], self.embed_channel,
            [[32, 32, 64], [64, 64, 128], [64, 96, 128]]
        )
        self.mamba_enhance_level1 = MambaBlock(
            320, self.mamba_depth, self.mamba_rms_norm, self.mamba_drop_path,
            self.mamba_fetch_idx, self.mamba_dropout, self.mamba_drop_path_rate
        )

        self.sa_downsample_level2 = PointNetSetAbstractionMsgWithLGE(
            fps_sampling_nums[1], [0.2, 0.4, 0.8], [32, 64, 128], 320,
            [[64, 64, 128], [128, 128, 256], [128, 128, 256]]
        )
        self.mamba_enhance_level2 = MambaBlock(
            640, self.mamba_depth, self.mamba_rms_norm, self.mamba_drop_path,
            self.mamba_fetch_idx, self.mamba_dropout, self.mamba_drop_path_rate
        )

        self.sa_downsample_level3 = PointNetSetAbstractionMsgWithLGE(
            fps_sampling_nums[2], [0.4, 0.8, 1.0], [16, 32, 64], 640,
            [[128, 128, 256], [256, 256, 512], [256, 256, 512]]
        )
        self.mamba_enhance_level3 = MambaBlock(
            1280, self.mamba_depth, self.mamba_rms_norm, self.mamba_drop_path,
            self.mamba_fetch_idx, self.mamba_dropout, self.mamba_drop_path_rate
        )

        self.global_feat_extractor = PointNetSetAbstraction(
            1280, [256, 512, 1024], True
        )

        # 5. Decoder: Inter-channel Shuffled Reconstruction (ICSR)
        self.icsr_upsample_level4_to_3 = ICSRModule(
            in_channel=1024 + 320 + 640 + 1280 + 1280, mlp=[1024, 512], interp_k=3
        )
        self.icsr_upsample_level3_to_2 = ICSRModule(
            in_channel=512 + 640, mlp=[512, 256], interp_k=3
        )
        self.icsr_upsample_level2_to_1 = ICSRModule(
            in_channel=256 + 320, mlp=[256, 128], interp_k=3
        )
        self.icsr_upsample_level1_to_raw = ICSRModule(
            in_channel=128 + 3 + self.embed_channel, mlp=[128, 128], interp_k=2
        )

        # Keep global branch conservative at start, then let training learn useful fusion.
        self.global_feat_weight_level1 = nn.Parameter(torch.tensor([0.0]))
        self.global_feat_weight_level2 = nn.Parameter(torch.tensor([0.0]))
        self.global_feat_weight_level3 = nn.Parameter(torch.tensor([0.0]))

        self.cls_head_conv1 = nn.Conv1d(128, 128, 1)
        self.cls_head_bn1 = nn.BatchNorm1d(128)
        self.cls_head_dropout = nn.Dropout(0.5)
        self.cls_head_conv2 = nn.Conv1d(128, num_classes, 1)

    def _farthest_point_sample(self, source_xyz, sample_len):
        # source_xyz is expected as [B, 3, N] (or [B, C, N] where first 3 are xyz).
        if source_xyz is None or source_xyz.dim() != 3:
            return None
        if source_xyz.size(1) < 3:
            return None

        B, _, N = source_xyz.shape
        sample_len = int(sample_len)
        if sample_len <= 0:
            return torch.empty((B, 0), dtype=torch.long, device=source_xyz.device)
        if N <= 0:
            return None

        xyz = source_xyz[:, :3, :].transpose(1, 2).contiguous()  # [B, N, 3]
        sample_eff = min(sample_len, N)
        centroids = torch.zeros((B, sample_eff), dtype=torch.long, device=source_xyz.device)
        distance = torch.full((B, N), 1e10, device=source_xyz.device)
        batch_indices = torch.arange(B, dtype=torch.long, device=source_xyz.device)

        # Deterministic start: farthest from cloud centroid.
        cloud_center = xyz.mean(dim=1, keepdim=True)
        farthest = torch.sum((xyz - cloud_center) ** 2, dim=-1).max(dim=-1)[1]

        for i in range(sample_eff):
            centroids[:, i] = farthest
            centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
            dist = torch.sum((xyz - centroid) ** 2, dim=-1)
            distance = torch.minimum(distance, dist)
            farthest = distance.max(dim=-1)[1]

        if sample_eff < sample_len:
            pad = centroids[:, -1:].expand(-1, sample_len - sample_eff)
            centroids = torch.cat([centroids, pad], dim=1)
        return centroids

    def _build_default_fps_indices(self, sample_len, source_len, batch_size, device, source_xyz=None):
        if source_len <= 0:
            raise ValueError("Cannot sample from empty point set.")
        if source_xyz is not None:
            fps_idx = self._farthest_point_sample(source_xyz, sample_len)
            if fps_idx is not None:
                return fps_idx
        if sample_len <= source_len:
            base = torch.arange(sample_len, device=device, dtype=torch.long)
        else:
            base = torch.arange(source_len, device=device, dtype=torch.long)
            pad = base[-1:].expand(sample_len - source_len)
            base = torch.cat([base, pad], dim=0)
        return base.unsqueeze(0).repeat(batch_size, 1)

    def _report_index_issue(self, index_type, level, reason):
        msg = f"{index_type} indices at level {level} are invalid ({reason}). Fallback is used."
        if self.strict_indices:
            raise ValueError(msg)
        warn_key = (index_type, int(level), str(reason))
        if warn_key not in self._index_warning_cache:
            warnings.warn(msg, RuntimeWarning, stacklevel=3)
            self._index_warning_cache.add(warn_key)

    @staticmethod
    def _is_permutation_fast(idx, target_len):
        # O(B * N) check without sorting:
        # valid permutation iff each position in [0, N-1] appears exactly once.
        B = idx.size(0)
        counts = torch.zeros((B, target_len), dtype=torch.int32, device=idx.device)
        ones = torch.ones_like(idx, dtype=torch.int32)
        counts.scatter_add_(1, idx, ones)
        return (counts == 1).all().item()

    @staticmethod
    def _is_permutation_strict(idx, target_len):
        expected = torch.arange(target_len, device=idx.device, dtype=torch.long).unsqueeze(0)
        return (torch.sort(idx, dim=1)[0] == expected).all().item()

    def _map_absolute_fps_indices(self, idx, source_origin_idx, original_len):
        # idx: [B, sample_len] absolute ids in the original cloud.
        # source_origin_idx: [B, source_len] absolute ids for current source set.
        if original_len <= 0:
            return None
        B, source_len = source_origin_idx.shape
        if ((source_origin_idx < 0) | (source_origin_idx >= original_len)).any().item():
            return None

        lut = torch.full((B, original_len), -1, dtype=torch.long, device=source_origin_idx.device)
        rel = torch.arange(source_len, dtype=torch.long, device=source_origin_idx.device).unsqueeze(0).expand(B, -1)
        lut.scatter_(1, source_origin_idx, rel)
        mapped = torch.gather(lut, 1, idx)
        if (mapped < 0).any().item():
            return None
        return mapped

    def _get_level_fps_indices(
            self,
            fps_sampling_indices,
            level,
            sample_len,
            source_len,
            batch_size,
            device,
            source_xyz=None,
            source_origin_idx=None,
            original_len=None):
        default_idx = self._build_default_fps_indices(
            sample_len, source_len, batch_size, device, source_xyz=source_xyz
        )
        if fps_sampling_indices is None:
            return default_idx

        idx = None
        if fps_sampling_indices.dim() == 3 and level < fps_sampling_indices.size(1):
            idx = fps_sampling_indices[:, level, :]
        elif fps_sampling_indices.dim() == 2 and level == 0:
            idx = fps_sampling_indices

        if idx is None:
            self._report_index_issue("FPS", level, "missing level or incompatible shape")
            return default_idx
        if idx.size(1) < sample_len:
            self._report_index_issue("FPS", level, "insufficient index length")
            return default_idx

        idx = idx[:, :sample_len].to(device=device, dtype=torch.long)
        if ((idx >= 0) & (idx < source_len)).all().item():
            return idx

        if source_origin_idx is not None and original_len is not None:
            if ((idx >= 0) & (idx < original_len)).all().item():
                mapped_idx = self._map_absolute_fps_indices(
                    idx, source_origin_idx.to(device=device, dtype=torch.long), original_len
                )
                if mapped_idx is not None:
                    return mapped_idx

        self._report_index_issue("FPS", level, "out-of-range values or unmappable absolute ids")
        return default_idx

    def _get_level_mamba_series_indices(self, mamba_series_indices, level, target_len, device):
        if mamba_series_indices is None:
            return None

        idx = None
        if mamba_series_indices.dim() == 4 and level < mamba_series_indices.size(1):
            # [B, L, S, N] -> take current level and keep all scan directions.
            idx = mamba_series_indices[:, level, :, :]
        elif mamba_series_indices.dim() == 3:
            if mamba_series_indices.size(1) == len(self.fps_sampling_nums) and level < mamba_series_indices.size(1):
                # Backward-compatible shape: [B, L, N], single scan direction.
                idx = mamba_series_indices[:, level, :].unsqueeze(1)
            else:
                # Single-level multi-direction shape: [B, S, N].
                idx = mamba_series_indices
        elif mamba_series_indices.dim() == 2:
            idx = mamba_series_indices.unsqueeze(1)

        if idx is None:
            self._report_index_issue("Mamba series", level, "missing level or incompatible shape")
            return None
        if idx.dim() != 3:
            self._report_index_issue("Mamba series", level, "expected [B, S, N] after parsing")
            return None
        if idx.size(2) < target_len:
            self._report_index_issue("Mamba series", level, "insufficient index length")
            return None

        idx = idx[:, :, :target_len].to(device=device, dtype=torch.long)
        invalid_idx = ((idx < 0) | (idx >= target_len)).any().item()
        if invalid_idx:
            self._report_index_issue("Mamba series", level, "contains out-of-range values")
            return None

        # Always validate permutations to avoid inverse-index mismatch.
        # Use fast check during training by default; use strict sort-based check
        # when explicitly requested or in eval mode.
        use_strict_check = self.strict_indices or self.validate_series_permutation or (not self.training)
        for s in range(idx.size(1)):
            idx_s = idx[:, s, :]
            if use_strict_check:
                is_permutation = self._is_permutation_strict(idx_s, target_len)
            else:
                is_permutation = self._is_permutation_fast(idx_s, target_len)
            if not is_permutation:
                self._report_index_issue("Mamba series", level, f"direction {s} is not a valid permutation")
                return None
        return idx

    def forward(self, point_cloud, fps_sampling_indices=None, mamba_series_indices=None):
        B, _, N = point_cloud.shape
        point_coords = point_cloud[:, :3, :]
        feats_in = point_cloud if self.use_normal else None
        origin_idx_level0 = torch.arange(N, device=point_cloud.device, dtype=torch.long).unsqueeze(0).repeat(B, 1)

        initial_embed_feats = self.lge_stem(point_coords, feats_in)
        fps_idx_level1 = self._get_level_fps_indices(
            fps_sampling_indices, level=0, sample_len=self.fps_sampling_nums[0],
            source_len=N, batch_size=B, device=point_cloud.device,
            source_xyz=point_coords,
            source_origin_idx=origin_idx_level0, original_len=N
        )
        origin_idx_level1 = torch.gather(origin_idx_level0, 1, fps_idx_level1)

        downsampled_coords1, downsampled_feats1 = self.sa_downsample_level1(
            point_coords, initial_embed_feats,
            fps_idx_level1
        )
        mamba_idx_level1 = self._get_level_mamba_series_indices(
            mamba_series_indices, level=0, target_len=downsampled_feats1.size(2), device=point_cloud.device
        )
        enhanced_feats1 = self.mamba_enhance_level1(
            downsampled_feats1, mamba_idx_level1, downsampled_coords1, prevalidated_idx=True
        )
        global_feats_level1 = self.global_feat_weight_level1 * torch.max(enhanced_feats1, 2)[0]
        fps_idx_level2 = self._get_level_fps_indices(
            fps_sampling_indices, level=1, sample_len=self.fps_sampling_nums[1],
            source_len=downsampled_coords1.size(2), batch_size=B, device=point_cloud.device,
            source_xyz=downsampled_coords1,
            source_origin_idx=origin_idx_level1, original_len=N
        )
        origin_idx_level2 = torch.gather(origin_idx_level1, 1, fps_idx_level2)

        downsampled_coords2, downsampled_feats2 = self.sa_downsample_level2(
            downsampled_coords1, enhanced_feats1,
            fps_idx_level2
        )
        mamba_idx_level2 = self._get_level_mamba_series_indices(
            mamba_series_indices, level=1, target_len=downsampled_feats2.size(2), device=point_cloud.device
        )
        enhanced_feats2 = self.mamba_enhance_level2(
            downsampled_feats2, mamba_idx_level2, downsampled_coords2, prevalidated_idx=True
        )
        global_feats_level2 = self.global_feat_weight_level2 * torch.max(enhanced_feats2, 2)[0]
        fps_idx_level3 = self._get_level_fps_indices(
            fps_sampling_indices, level=2, sample_len=self.fps_sampling_nums[2],
            source_len=downsampled_coords2.size(2), batch_size=B, device=point_cloud.device,
            source_xyz=downsampled_coords2,
            source_origin_idx=origin_idx_level2, original_len=N
        )

        downsampled_coords3, downsampled_feats3 = self.sa_downsample_level3(
            downsampled_coords2, enhanced_feats2,
            fps_idx_level3
        )
        mamba_idx_level3 = self._get_level_mamba_series_indices(
            mamba_series_indices, level=2, target_len=downsampled_feats3.size(2), device=point_cloud.device
        )
        enhanced_feats3 = self.mamba_enhance_level3(
            downsampled_feats3, mamba_idx_level3, downsampled_coords3, prevalidated_idx=True
        )
        global_feats_level3 = self.global_feat_weight_level3 * torch.max(enhanced_feats3, 2)[0]

        global_coords, global_feats = self.global_feat_extractor(downsampled_coords3, enhanced_feats3)
        global_feats_level4 = global_feats.view(B, 1024)

        fused_global_feats = torch.cat(
            (global_feats_level1, global_feats_level2, global_feats_level3, global_feats_level4),
            dim=-1
        ).unsqueeze(-1)

        upsampled_feats3 = self.icsr_upsample_level4_to_3(
            downsampled_coords3, global_coords, enhanced_feats3, fused_global_feats
        )
        upsampled_feats2 = self.icsr_upsample_level3_to_2(
            downsampled_coords2, downsampled_coords3, enhanced_feats2, upsampled_feats3
        )
        upsampled_feats1 = self.icsr_upsample_level2_to_1(
            downsampled_coords1, downsampled_coords2, enhanced_feats1, upsampled_feats2
        )
        raw_res_feats = self.icsr_upsample_level1_to_raw(
            point_coords, downsampled_coords1,
            torch.cat([point_coords, initial_embed_feats], 1),
            upsampled_feats1
        )

        cls_feat = F.relu(self.cls_head_bn1(self.cls_head_conv1(raw_res_feats)))
        cls_feat = self.cls_head_dropout(cls_feat)
        seg_pred = self.cls_head_conv2(cls_feat)
        seg_pred = F.log_softmax(seg_pred, dim=1)
        seg_pred = seg_pred.permute(0, 2, 1)

        return seg_pred


class get_model(nn.Module):
    def __init__(self, num_classes, fps_n_list, normal_channel=True):
        super(get_model, self).__init__()
        self.net = DiffMambaNet(
            num_classes=num_classes,
            fps_sampling_nums=fps_n_list,
            normal_channel=normal_channel
        )

    def forward(self, xyz, fps_index_array, series_idx_arrays):
        return self.net(xyz, fps_index_array, series_idx_arrays)


class get_loss(nn.Module):
    def __init__(self, ignore_index=-1):
        super(get_loss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, pred, gold, weight=None, smoothing=False):
        gold = gold.contiguous().view(-1)

        if smoothing:
            eps = 0.2
            n_class = pred.size(1)
            one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_prb = F.log_softmax(pred, dim=1)
            loss = -(one_hot * log_prb).sum(dim=1).mean()
        else:
            loss = F.cross_entropy(pred, gold, weight=weight, ignore_index=self.ignore_index, reduction='mean')

        return loss


class get_loss_weighted(nn.Module):
    def __init__(self, ignore_index=-1, fence_class_idx=7, fence_focal_gamma=2.0, fence_focal_weight=0.0):
        super(get_loss_weighted, self).__init__()
        self.ignore_index = ignore_index
        self.fence_class_idx = int(fence_class_idx)
        self.fence_focal_gamma = float(fence_focal_gamma)
        self.fence_focal_weight = float(fence_focal_weight)

    def forward(self, pred, target, weight):
        target = target.view(-1)
        valid_mask = target != self.ignore_index
        if not valid_mask.any():
            return pred.sum() * 0.0

        pred_valid = pred[valid_mask]
        target_valid = target[valid_mask]
        ce_loss = F.cross_entropy(pred_valid, target_valid, weight=weight, reduction='mean')

        if self.fence_focal_weight <= 0.0:
            return ce_loss

        fence_mask = target_valid == self.fence_class_idx
        if not fence_mask.any():
            return ce_loss

        pred_fence = pred_valid[fence_mask]
        target_fence = target_valid[fence_mask]
        log_probs = F.log_softmax(pred_fence, dim=1)
        log_pt = log_probs.gather(1, target_fence.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()
        focal_term = -((1.0 - pt).clamp_min(1e-6) ** self.fence_focal_gamma) * log_pt
        if weight is not None and weight.numel() > self.fence_class_idx:
            focal_term = focal_term * weight[self.fence_class_idx].detach()
        fence_focal_loss = focal_term.mean()

        return ce_loss + self.fence_focal_weight * fence_focal_loss

