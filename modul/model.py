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


class HighDSOTAXR(nn.Module):
    def __init__(
            self,
            motion_dim,
            interaction_dim,
            long_dim,
            lat_dim,
            T_f,
            K=3,
            use_motion=True,
            use_interaction=True,
            use_style=True,
            use_intent=True,
            use_physics=True,
            use_multimodal=True,
    ):
        super().__init__()
        self.T_f = T_f
        self.K = K if use_multimodal else 1
        self.use_motion = use_motion
        self.use_interaction = use_interaction
        self.use_style = use_style
        self.use_intent = use_intent
        self.use_physics = use_physics
        self.use_multimodal = use_multimodal

        feature_dims = []

        if use_motion:
            self.motion = MotionEncoder(motion_dim)
            feature_dims.append(256)
        else:
            self.motion = None

        if use_interaction:
            self.interaction = InteractionTransformer(interaction_dim)
            feature_dims.append(128)
        else:
            self.interaction = None

        if use_style:
            self.style = StyleEncoder(long_dim, lat_dim)
            feature_dims.append(128)
        else:
            self.style = None

        fusion_dim = sum(feature_dims) if feature_dims else 256

        if use_intent:
            self.intent = IntentHead(fusion_dim)
        else:
            self.intent = None

        decoder_in_dim = fusion_dim
        self.decoder = MultiModalDecoder(decoder_in_dim, T_f, self.K)

    def forward(self, hist, interaction, s_long, s_lat):
        features = []

        if self.use_motion and self.motion is not None:
            m = self.motion(hist)
            features.append(m)

        if self.use_interaction and self.interaction is not None:
            i = self.interaction(interaction)
            features.append(i)

        if self.use_style and self.style is not None:
            s = self.style(s_long, s_lat)
            features.append(s)

        if features:
            feat = torch.cat(features, dim=-1)
        else:
            if not hasattr(self, 'fallback_fc'):
                self.fallback_fc = nn.Linear(hist.shape[-1], 256).to(hist.device)
            feat = self.fallback_fc(hist[:, -1, :])

        if self.use_intent and self.intent is not None:
            intent_logits = self.intent(feat)
        else:
            intent_logits = None

        residual, prob, prob_logits = self.decoder(feat)

        if self.use_physics:
            physics = physics_cv(hist, self.T_f)
            physics = physics.unsqueeze(1)
            pred = physics + residual
        else:
            pred = residual

        return pred, prob, intent_logits, prob_logits


class LSTMAutoRegressive(nn.Module):
    def __init__(self, motion_dim=6, hidden_dim=128, num_layers=2, T_f=125, dt=0.04):
        super().__init__()
        self.motion_dim = motion_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.T_f = T_f
        self.dt = dt

        self.encoder_lstm = nn.LSTM(
            input_size=motion_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )

        self.decoder_lstm = nn.LSTM(
            input_size=2,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )

        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, hist):
        batch_size = hist.size(0)

        encoder_out, (h_n, c_n) = self.encoder_lstm(hist)

        h_dec = h_n
        c_dec = c_n

        last_pos = hist[:, -1, :2]
        current_input = last_pos.unsqueeze(1)

        predictions = []

        for t in range(self.T_f):
            decoder_out, (h_dec, c_dec) = self.decoder_lstm(current_input, (h_dec, c_dec))
            next_pos = self.output_head(decoder_out.squeeze(1))
            predictions.append(next_pos.unsqueeze(1))
            current_input = next_pos.unsqueeze(1)

        pred = torch.cat(predictions, dim=1)

        return pred


class HighDSOTAXR2(nn.Module):
    def __init__(
            self,
            motion_dim,
            interaction_dim,
            long_dim,
            lat_dim,
            T_f,
            K=3,
            use_motion=True,
            use_interaction=True,
            use_style=True,
            use_intent=True,
            use_physics=True,
            use_multimodal=True,
    ):
        super().__init__()
        self.T_f = T_f
        self.K = K if use_multimodal else 1
        self.use_motion = use_motion
        self.use_interaction = use_interaction
        self.use_style = use_style
        self.use_intent = use_intent
        self.use_physics = use_physics
        self.use_multimodal = use_multimodal

        self.is_pure_lstm = (
                use_motion and
                not use_interaction and
                not use_style and
                not use_intent and
                not use_physics and
                not use_multimodal
        )

        if self.is_pure_lstm:
            self.lstm_predictor = LSTMAutoRegressive(
                motion_dim=motion_dim,
                hidden_dim=128,
                num_layers=2,
                T_f=T_f,
                dt=0.04
            )
            self.motion = None
            self.interaction = None
            self.style = None
            self.intent = None
            self.decoder = None
        else:
            self.lstm_predictor = None

            feature_dims = []

            if use_motion:
                self.motion = MotionEncoder(motion_dim)
                feature_dims.append(256)
            else:
                self.motion = None

            if use_interaction:
                self.interaction = InteractionTransformer(interaction_dim)
                feature_dims.append(128)
            else:
                self.interaction = None

            if use_style:
                self.style = StyleEncoder(long_dim, lat_dim)
                feature_dims.append(128)
            else:
                self.style = None

            fusion_dim = sum(feature_dims) if feature_dims else 256

            if use_intent:
                self.intent = IntentHead(fusion_dim)
            else:
                self.intent = None

            decoder_in_dim = fusion_dim
            self.decoder = MultiModalDecoder(decoder_in_dim, T_f, self.K)

    def forward(self, hist, interaction, s_long, s_lat):
        if self.is_pure_lstm and self.lstm_predictor is not None:
            pred = self.lstm_predictor(hist)
            pred = pred.unsqueeze(1)
            prob = torch.ones(pred.size(0), 1, device=pred.device)
            intent_logits = None
            prob_logits = None
            return pred, prob, intent_logits, prob_logits

        features = []

        if self.use_motion and self.motion is not None:
            m = self.motion(hist)
            features.append(m)

        if self.use_interaction and self.interaction is not None:
            i = self.interaction(interaction)
            features.append(i)

        if self.use_style and self.style is not None:
            s = self.style(s_long, s_lat)
            features.append(s)

        if features:
            feat = torch.cat(features, dim=-1)
        else:
            if not hasattr(self, 'fallback_fc'):
                self.fallback_fc = nn.Linear(hist.shape[-1], 256).to(hist.device)
            feat = self.fallback_fc(hist[:, -1, :])

        if self.use_intent and self.intent is not None:
            intent_logits = self.intent(feat)
        else:
            intent_logits = None

        residual, prob, prob_logits = self.decoder(feat)

        if self.use_physics:
            physics = physics_cv(hist, self.T_f)
            physics = physics.unsqueeze(1)
            pred = physics + residual
        else:
            pred = residual

        return pred, prob, intent_logits, prob_logits


def physics_cv(hist, T_f, dt=0.04):
    pos = hist[:, -1, :2]
    vel = hist[:, -1, 2:4]

    traj = []
    for t in range(1, T_f + 1):
        traj.append(pos + vel * (t * dt))

    traj = torch.stack(traj, dim=1)
    return traj


def physics_cv_from_last(last_state, T_f, dt=0.04):
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
        physics = physics_cv_from_last(hist_last, self.T_f).unsqueeze(1)
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