import torch
import torch.nn as nn


class MotionEncoder(nn.Module):
    def __init__(self, in_dim=6, hidden=128):
        super().__init__()
        self.lstm = nn.LSTM(
            in_dim,
            hidden,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return out[:, -1]


class InteractionTransformer(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        possible_heads = [8, 4, 2, 1]
        nhead = 1
        for h in possible_heads:
            if in_dim % h == 0:
                nhead = h
                break

        layer = nn.TransformerEncoderLayer(
            d_model=in_dim,
            nhead=nhead,
            dim_feedforward=256,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, 2)
        self.fc = nn.Linear(in_dim, 128)

    def forward(self, x):
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.fc(x)


class StyleEncoder(nn.Module):
    def __init__(self, long_dim, lat_dim):
        super().__init__()
        self.long_net = nn.Sequential(
            nn.Linear(long_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        self.lat_net = nn.Sequential(
            nn.Linear(lat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )

    def forward(self, long, lat):
        long_f = self.long_net(long)
        lat_f = self.lat_net(lat)
        return torch.cat([long_f, lat_f], dim=-1)


class IntentHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        return self.net(x)


def physics_cv(last_state, T_f, dt=0.04):
    x = last_state[:, 0]
    y = last_state[:, 1]
    vx = last_state[:, 2]
    vy = last_state[:, 3]

    traj = []

    for t in range(1, T_f + 1):
        xt = x + vx * dt * t
        yt = y + vy * dt * t
        traj.append(torch.stack([xt, yt], dim=-1))

    traj = torch.stack(traj, dim=1)
    return traj


class MultiModalDecoder(nn.Module):
    def __init__(self, in_dim, T_f, K=3):
        super().__init__()
        self.T_f = T_f
        self.K = K

        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, K * T_f * 2)
        )

        self.prob_head = nn.Linear(in_dim, K)

    def forward(self, x):
        traj = self.net(x)
        traj = traj.view(-1, self.K, self.T_f, 2)

        prob_logits = self.prob_head(x)
        prob = torch.softmax(prob_logits, dim=-1)

        return traj, prob, prob_logits


class ResFeatureFusion(nn.Module):
    def __init__(self, fusion_dim, interaction_dim=128, style_dim=128):
        super().__init__()

        self.interaction_proj = nn.Linear(interaction_dim, fusion_dim) if interaction_dim != fusion_dim else nn.Identity()
        self.style_proj = nn.Linear(style_dim, fusion_dim) if style_dim != fusion_dim else nn.Identity()

        self.fusion_weight = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.Sigmoid()
        )

    def forward(self, fusion_feature, interaction_feat, style_feat):
        interaction_proj = self.interaction_proj(interaction_feat)
        style_proj = self.style_proj(style_feat)

        residual = interaction_proj + style_proj

        concat_feat = torch.cat([fusion_feature, residual], dim=-1)
        weight = self.fusion_weight(concat_feat)

        res_feature = fusion_feature + weight * residual

        return res_feature


class MultiBranchFusion(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_dim)

        self.gate_layer = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.GELU(),
            nn.Linear(input_dim * 2, input_dim),
            nn.Sigmoid()
        )

        self.transform = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        self.se_layer = nn.Sequential(
            nn.Linear(input_dim, input_dim // 4),
            nn.GELU(),
            nn.Linear(input_dim // 4, input_dim),
            nn.Sigmoid()
        )

        self.output_proj = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.layer_norm(x)

        gate = self.gate_layer(x)
        x_gated = x * gate

        x_transformed = self.transform(x_gated)

        se_weight = self.se_layer(x_transformed)
        x_se = x_transformed * se_weight

        x_fused = x_se + x_gated

        output = self.output_proj(x_fused)

        return output


class PRISM(nn.Module):
    def __init__(
            self,
            motion_dim,
            interaction_dim,
            long_dim,
            lat_dim,
            T_f,
            K=3,
            fusion_output_dim=512,
    ):
        super().__init__()

        self.T_f = T_f
        self.K = K

        self.motion_encoder = MotionEncoder(motion_dim)
        self.interaction_encoder = InteractionTransformer(interaction_dim)
        self.style_encoder = StyleEncoder(long_dim, lat_dim)

        self.motion_dim_out = 256
        self.interaction_dim_out = 128
        self.style_dim_out = 128

        concat_dim = self.motion_dim_out + self.interaction_dim_out + self.style_dim_out

        self.fusion_layer = MultiBranchFusion(
            input_dim=concat_dim,
            output_dim=fusion_output_dim
        )

        self.decoder = MultiModalDecoder(concat_dim, T_f, K)
        self.intent_head = IntentHead(fusion_output_dim)
        self.res_fusion = ResFeatureFusion(
            fusion_dim=fusion_output_dim,
            interaction_dim=self.interaction_dim_out,
            style_dim=self.style_dim_out
        )

    def layer1_input(self, hist, interaction, s_long, s_lat):
        return hist, interaction, s_long, s_lat

    def layer2_encode(self, hist, interaction, s_long, s_lat):
        motion_feat = self.motion_encoder(hist)
        interaction_feat = self.interaction_encoder(interaction)
        style_feat = self.style_encoder(s_long, s_lat)
        return motion_feat, interaction_feat, style_feat

    def layer3_concat(self, motion_feat, interaction_feat, style_feat):
        concat_feat = torch.cat([motion_feat, interaction_feat, style_feat], dim=-1)
        return concat_feat

    def layer4_fusion(self, concat_feat):
        fusion_feat = self.fusion_layer(concat_feat)
        return fusion_feat

    def layer5_decode(self, fusion_feat, interaction_feat, style_feat, concat_feat):
        residual, prob, prob_logits = self.decoder(concat_feat)
        res_feature = self.res_fusion(fusion_feat, interaction_feat, style_feat)
        intent_logits = self.intent_head(res_feature)
        return residual, prob, prob_logits, intent_logits

    def layer6_output(self, residual, prob, prob_logits, intent_logits, hist_last):
        physics = physics_cv(hist_last, self.T_f).unsqueeze(1)
        pred = physics + residual
        return pred, prob, intent_logits, prob_logits

    def forward(self, hist, interaction, s_long, s_lat, hist_last=None):
        hist, interaction, s_long, s_lat = self.layer1_input(hist, interaction, s_long, s_lat)
        motion_feat, interaction_feat, style_feat = self.layer2_encode(hist, interaction, s_long, s_lat)
        concat_feat = self.layer3_concat(motion_feat, interaction_feat, style_feat)
        fusion_feat = self.layer4_fusion(concat_feat)
        residual, prob, prob_logits, intent_logits = self.layer5_decode(fusion_feat, interaction_feat, style_feat, concat_feat)
        pred, prob, intent_logits, prob_logits = self.layer6_output(residual, prob, prob_logits, intent_logits, hist_last)
        return pred, prob, intent_logits, prob_logits