import random
import matplotlib
matplotlib.use('Agg')
from multiprocessing.dummy import Pool
from os import cpu_count
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import torch
from tqdm import tqdm
import pickle
import json
import time
import gc
from pathlib import Path
import os
import csv
import numpy as np


def plot_trajectory_compared(hist, future, future_pred, scaler=None, save_dir=None, filename=None):
    if torch.is_tensor(hist):
        hist = hist.detach().cpu().numpy()
    if torch.is_tensor(future):
        future = future.detach().cpu().numpy()
    if torch.is_tensor(future_pred):
        future_pred = future_pred.detach().cpu().numpy()

    hist_ = hist[:, :, :2]
    means = scaler.scaler.mean_[:2]
    scales = scaler.scaler.scale_[:2]
    position = hist_ * scales + means

    if len(hist.shape) == 3:
        hist = position[0]
    if len(future.shape) == 3:
        future = future[0]
    if len(future_pred.shape) == 3:
        future_pred = future_pred[0]

    plt.figure(figsize=(10, 8))

    plt.plot(hist[:, 0], hist[:, 1], 'b-', linewidth=2, label='History')
    plt.scatter(hist[0, 0], hist[0, 1], c='blue', s=50, label='Start')
    plt.scatter(hist[-1, 0], hist[-1, 1], c='purple', s=80, marker='s', label='Current')

    plt.plot(future[:, 0], future[:, 1], 'g-', linewidth=2, label='Ground Truth')
    plt.scatter(future[-1, 0], future[-1, 1], c='green', s=100, marker='*', label='GT End')

    plt.plot(future_pred[:, 0], future_pred[:, 1], 'r--', linewidth=2, label='Prediction')
    plt.scatter(future_pred[-1, 0], future_pred[-1, 1], c='red', s=100, marker='*', label='Pred End')

    ade = np.mean(np.sqrt(np.sum((future_pred - future) ** 2, axis=1)))
    fde = np.sqrt(np.sum((future_pred[-1] - future[-1]) ** 2))
    plt.title(f'Trajectory Comparison (ADE: {ade:.3f}m, FDE: {fde:.3f}m)')

    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        if filename is None:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'trajectory_{timestamp}.png'
        elif not filename.endswith('.png'):
            filename += '.png'

        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.close()

    return save_path if save_dir else None


def plot_prediction_vs_gt(
        csv_path="trajectory_prediction_results.csv",
        batch_id=0,
        sample_id=0,
        save_fig=False,
        fig_path="trajectory_plot.png"
):
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.read_csv(csv_path)

    df_sample = df[
        (df["batch_id"] == batch_id) &
        (df["sample_id"] == sample_id)
    ]

    pred_x = df_sample["pred_x"].values
    pred_y = df_sample["pred_y"].values
    gt_x = df_sample["gt_x"].values
    gt_y = df_sample["gt_y"].values
    error = df_sample["error"].values

    plt.figure(figsize=(6, 6))

    plt.plot(
        gt_x[:125],
        gt_y[:125],
        'o-',
        label="Ground Truth",
        linewidth=2
    )

    plt.plot(
        pred_x[:125],
        pred_y[:125],
        's--',
        label="Prediction",
        linewidth=2
    )

    plt.scatter(
        gt_x[0],
        gt_y[0],
        marker='*',
        s=150,
        label="Start"
    )

    plt.scatter(
        gt_x[124],
        gt_y[124],
        marker='X',
        s=120,
        label="GT End"
    )

    plt.scatter(
        pred_x[124],
        pred_y[124],
        marker='D',
        s=120,
        label="Pred End"
    )

    plt.title(
        f"Trajectory Prediction (Batch {batch_id}, Sample {sample_id})\n"
        f"Mean Error = {error.mean():.3f} m"
    )

    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")

    if save_fig:
        plt.savefig(fig_path, dpi=300)
    plt.show()


def save_prediction_results(
        pred, future,
        batch_id,
        scene_ids,
        batch_data,
        vehicle_ids,
        segment_ids,
        frame_nums,
        init_num, scene_idx, epoch,
        save_path="trajectory_prediction_results.csv"
):
    pred_np = pred.detach().cpu().numpy()
    future_np = future.detach().cpu().numpy()

    batch_size = pred_np.shape[0]
    T = pred_np.shape[1]

    if not isinstance(scene_ids, (list, np.ndarray)):
        scene_ids = scene_ids.cpu().numpy() if torch.is_tensor(scene_ids) else [scene_ids]
    if not isinstance(vehicle_ids, (list, np.ndarray)):
        vehicle_ids = vehicle_ids.cpu().numpy() if torch.is_tensor(vehicle_ids) else [vehicle_ids]
    if not isinstance(segment_ids, (list, np.ndarray)):
        segment_ids = segment_ids.cpu().numpy() if torch.is_tensor(segment_ids) else [segment_ids]
    if not isinstance(frame_nums, (list, np.ndarray)):
        frame_nums = frame_nums.cpu().numpy() if torch.is_tensor(frame_nums) else [frame_nums]

    file_exists = os.path.exists(save_path)
    with open(save_path, "a" if file_exists else "w", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "init_num",
                "scene_idx",
                "epoch",
                "batch_id",
                "scene_id",
                "vehicle_id",
                "segment_id",
                "frame_idx",
                "time_step",
                "pred_x",
                "pred_y",
                "gt_x",
                "gt_y",
                "error"
            ])

        for i in range(batch_size):
            scene_id = scene_ids[i] if i < len(scene_ids) else "unknown"
            vehicle_id = vehicle_ids[i] if i < len(vehicle_ids) else "unknown"
            segment_id = segment_ids[i] if i < len(segment_ids) else "unknown"
            start_frame = frame_nums[i] if i < len(frame_nums) else 0

            for t in range(T):
                pred_x = pred_np[i, t, 0]
                pred_y = pred_np[i, t, 1]
                gt_x = future_np[i, t, 0]
                gt_y = future_np[i, t, 1]

                current_frame = start_frame + t
                error = np.sqrt((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2)

                writer.writerow([
                    init_num,
                    scene_idx,
                    epoch,
                    batch_id,
                    scene_id,
                    vehicle_id,
                    segment_id,
                    current_frame,
                    t,
                    f"{pred_x:.6f}",
                    f"{pred_y:.6f}",
                    f"{gt_x:.6f}",
                    f"{gt_y:.6f}",
                    f"{error:.6f}"
                ])


def visualize_style_control(
    model,
    dataset,
    device,
    sample_idx=0,
    T_pred=30,
    lat_scale_list=[0.5, 1.0, 1.5]
):
    model.eval()

    sample = dataset[sample_idx]

    hist_motion = torch.tensor(sample["hist_motion"]).unsqueeze(0).to(device)
    style_long  = torch.tensor(sample["style_long"]).unsqueeze(0).to(device)
    style_lat   = torch.tensor(sample["style_lat"]).unsqueeze(0).to(device)
    gt_future   = sample["future"]

    with torch.no_grad():
        z_motion = model.motion_enc(hist_motion)
        z_long   = model.long_enc(style_long)
        z_lat    = model.lat_enc(style_lat)

        preds = []

        for scale in lat_scale_list:
            z_lat_mod = z_lat * scale
            traj = model.decoder(
                z_motion, z_long, z_lat_mod, T_pred
            )
            preds.append(traj.squeeze(0).cpu().numpy())

    plt.figure(figsize=(6, 6))

    hist_xy = hist_motion.squeeze(0).cpu().numpy()
    plt.plot(hist_xy[:, 0], hist_xy[:, 1],
             "ko-", label="History")

    plt.plot(gt_future[:, 0], gt_future[:, 1],
             "k--", linewidth=2, label="GT Future")

    colors = ["blue", "green", "red"]
    for pred, scale, c in zip(preds, lat_scale_list, colors):
        plt.plot(pred[:, 0], pred[:, 1],
                 color=c,
                 linewidth=2,
                 label=f"Pred (lat x {scale})")

    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.title("Style-Controlled Trajectory Prediction")

    plt.show()


class FeatureScaler:
    def __init__(self):
        self.motion_mean = None
        self.motion_std = None
        self.long_mean = None
        self.long_std = None
        self.lat_mean = None
        self.lat_std = None

    def fit(self, samples):
        motion = np.concatenate([s["hist_motion"] for s in samples], axis=0)
        long_f = np.concatenate([s["style_long"] for s in samples], axis=0)
        lat_f = np.concatenate([s["style_lat"] for s in samples], axis=0)

        self.motion_mean = motion.mean(axis=0)
        self.motion_std = motion.std(axis=0) + 1e-6

        self.long_mean = long_f.mean(axis=0)
        self.long_std = long_f.std(axis=0) + 1e-6

        self.lat_mean = lat_f.mean(axis=0)
        self.lat_std = lat_f.std(axis=0) + 1e-6

    def transform(self, sample):
        sample["hist_motion"] = (
            sample["hist_motion"] - self.motion_mean
        ) / self.motion_std

        sample["style_long"] = (
            sample["style_long"] - self.long_mean
        ) / self.long_std

        sample["style_lat"] = (
            sample["style_lat"] - self.lat_mean
        ) / self.lat_std

        return sample


def visualize_val_prediction(
        hist, future, pred,
        save_path=None,
        title="Val Trajectory Prediction"
):
    hist = hist.cpu().numpy()
    future = future.cpu().numpy()
    pred = pred.cpu().numpy()

    plt.figure(figsize=(5, 5))

    plt.plot(hist[:, 0], hist[:, 1], "k-", label="History")
    plt.plot(future[:, 0], future[:, 1], "g-", label="GT Future")
    plt.plot(pred[:, 0], pred[:, 1], "r--", label="Pred Future")

    plt.legend()
    plt.axis("equal")
    plt.grid(True)
    plt.title(title)

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()
    plt.close()


def add_features(df:pd.DataFrame):
    return df


def get_scene_lr_config(scene_idx, base_lr):
    if scene_idx == 0:
        return dict(
            base_lr=base_lr,
            warmup_epochs=5
        )
    else:
        return dict(
            base_lr=base_lr * 0.3,
            warmup_epochs=1
        )


def init_prism_weights(model,
                       weight_scale=1.0,
                       bias_scale=0.0,
                       decoder_last_std=0.01,
                       intent_last_std=0.01,
                       lstm_forget_bias=1.0,
                       use_large_noise=False,
                       large_noise_std=1.0):
    if use_large_noise:
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=large_noise_std)
                if module.bias is not None:
                    nn.init.normal_(module.bias, mean=0.0,
                                    std=large_noise_std) if bias_scale > 0 else nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.LSTM):
                for param_name, param in module.named_parameters():
                    if 'weight' in param_name:
                        nn.init.normal_(param, mean=0.0, std=large_noise_std)
                    elif 'bias' in param_name:
                        nn.init.constant_(param, 0.0)
                        if 'bias_ih' in param_name:
                            hidden_size = module.hidden_size
                            param.data[hidden_size:2 * hidden_size] = lstm_forget_bias
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
    else:
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=0, mode='fan_in', nonlinearity='relu')
                module.weight.data *= weight_scale
                if module.bias is not None:
                    if bias_scale > 0:
                        nn.init.normal_(module.bias, mean=0.0, std=bias_scale)
                    else:
                        nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.LSTM):
                for param_name, param in module.named_parameters():
                    if 'weight_ih' in param_name:
                        nn.init.xavier_uniform_(param)
                        param.data *= weight_scale
                    elif 'weight_hh' in param_name:
                        nn.init.orthogonal_(param)
                        param.data *= weight_scale
                    elif 'bias_ih' in param_name:
                        nn.init.constant_(param, 0.0)
                        param.data[module.hidden_size:2 * module.hidden_size] = lstm_forget_bias
                        if bias_scale > 0:
                            param.data += torch.randn_like(param.data) * bias_scale
                    elif 'bias_hh' in param_name:
                        nn.init.constant_(param, 0.0)
                        if bias_scale > 0:
                            param.data += torch.randn_like(param.data) * bias_scale
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)

    if hasattr(model, 'decoder') and hasattr(model.decoder, 'net'):
        last_linear = model.decoder.net[-1]
        if isinstance(last_linear, nn.Linear):
            if use_large_noise:
                nn.init.normal_(last_linear.weight, mean=0.0, std=large_noise_std)
            else:
                nn.init.normal_(last_linear.weight, mean=0.0, std=decoder_last_std)
            nn.init.constant_(last_linear.bias, 0.0)

    if hasattr(model, 'intent_head') and hasattr(model.intent_head, 'net'):
        intent_last = model.intent_head.net[-1]
        if isinstance(intent_last, nn.Linear):
            if use_large_noise:
                nn.init.normal_(intent_last.weight, mean=0.0, std=large_noise_std)
            else:
                nn.init.normal_(intent_last.weight, mean=0.0, std=intent_last_std)
            nn.init.constant_(intent_last.bias, 0.0)



def build_samples_from_df(df, T_h=20, T_f=30, motion_cols=None, style_long_cols=None, style_lat_cols=None):
    samples = []

    for vid in df["id"].unique():
        ego_df = df[df["id"] == vid].reset_index(drop=True)

        if len(ego_df) < T_h + T_f:
            continue

        for i in range(T_h, len(ego_df) - T_f):
            hist_df = ego_df.iloc[i - T_h:i]
            future_df = ego_df.iloc[i:i + T_f]

            lane_ids = future_df["lane_id"].values
            if np.all(lane_ids == lane_ids[0]):
                intent = 0
            elif lane_ids[-1] < lane_ids[0]:
                intent = 1
            else:
                intent = 2

            samples.append({
                "hist_motion": hist_df[motion_cols].values.astype(np.float32),
                "style_long": hist_df[style_long_cols].values.astype(np.float32),
                "style_lat": hist_df[style_lat_cols].values.astype(np.float32),
                "intent": intent,
                "future": future_df[["ego_x", "ego_y"]].values.astype(np.float32),
                "vehicle_id": vid
            })

    return samples


def _build_samples_for_one_vehicle(args):
    ego_df, T_h, T_f, motion_cols, style_long_cols, style_lat_cols, interaction_cols, scaler, balance_per_vehicle = args

    samples = []

    if len(ego_df) < T_h + T_f:
        return None

    for i in range(T_h, len(ego_df) - T_f):
        hist_df = ego_df.iloc[i - T_h:i]
        future_df = ego_df.iloc[i:i + T_f]

        lane_ids = future_df["lane_id"].values
        if np.all(lane_ids == lane_ids[0]):
            intent = 0
            continue
        else:
            try:
                first_lane = float(lane_ids[0])
                last_lane = float(lane_ids[-1])

                if last_lane < first_lane:
                    intent = 1
                elif last_lane > first_lane:
                    intent = 2
                else:
                    intent = 0
            except (ValueError, TypeError):
                if lane_ids[-1] < lane_ids[0]:
                    intent = 1
                else:
                    intent = 2

        temp_df = future_df.copy()
        future = scaler.inverse_transform(temp_df)[["ego_x", "ego_y"]].values.astype(np.float32)

        samples.append({
            "hist_motion": hist_df[motion_cols].values.astype(np.float32),
            "style_long": hist_df[style_long_cols].values.astype(np.float32),
            "style_lat": hist_df[style_lat_cols].values.astype(np.float32),
            "interaction": hist_df[interaction_cols].values.astype(np.float32),
            "intent": intent,
            "future": future,
            "vehicle_id": ego_df["id"].iloc[0]
        })

    if len(samples) > 0 and balance_per_vehicle:
        follow_samples = [s for s in samples if s["intent"] == 0]
        left_lane_change = [s for s in samples if s["intent"] == 1]
        right_lane_change = [s for s in samples if s["intent"] == 2]

        n_follow = len(follow_samples)
        n_left = len(left_lane_change)
        n_right = len(right_lane_change)

        max_lane_change = max(n_left, n_right)

        print(f"Vehicle {ego_df['id'].iloc[0]}: follow={n_follow}, left={n_left}, right={n_right}")

        if n_follow > 0 and max_lane_change > 0:
            n_balanced = min(n_follow, max_lane_change)

            if n_follow > n_balanced:
                follow_samples = random.sample(follow_samples, n_balanced)

            lane_change_samples = []
            if n_balanced > 0:
                if n_left > 0 and n_right > 0:
                    left_ratio = n_left / (n_left + n_right)
                    n_left_sample = int(n_balanced * left_ratio)
                    n_right_sample = n_balanced - n_left_sample

                    if n_left_sample > 0:
                        lane_change_samples.extend(random.sample(left_lane_change, min(n_left_sample, n_left)))
                    if n_right_sample > 0:
                        lane_change_samples.extend(random.sample(right_lane_change, min(n_right_sample, n_right)))
                elif n_left > 0:
                    lane_change_samples = random.sample(left_lane_change, n_balanced)
                elif n_right > 0:
                    lane_change_samples = random.sample(right_lane_change, n_balanced)

            balanced_samples = follow_samples + lane_change_samples
            random.shuffle(balanced_samples)

            print(f"  Balanced: follow={len(follow_samples)}, lane_change={len(lane_change_samples)}, total={len(balanced_samples)}")

            return balanced_samples
        else:
            print(f"  Vehicle has only one type, unchanged: total={len(samples)}")
            return samples

    return samples


def build_samples_from_df_parallel(
        df,
        T_h=20,
        T_f=30,
        num_workers=None ,
        motion_cols=None,
        style_long_cols=None,
        style_lat_cols=None,
        interaction_cols = None,
        scaler=None,
        balance_per_vehicle=False
):
    if num_workers is None:
        num_workers = max(cpu_count() - 1, 1)

    samples = []

    vehicle_groups = [
        df[df["id"] == vid].reset_index(drop=True)
        for vid in df["id"].unique()
    ]

    task_args = [
        (
            ego_df,
            T_h,
            T_f,
            motion_cols,
            style_long_cols,
            style_lat_cols,
            interaction_cols,
            scaler,
            balance_per_vehicle
        )
        for ego_df in vehicle_groups
    ]

    with Pool(processes=num_workers) as pool:
        for vehicle_samples in tqdm(
                pool.imap_unordered(_build_samples_for_one_vehicle, task_args),
                total=len(task_args),
                desc="Building samples (parallel)"
        ):
            if vehicle_samples is not None:
                samples.extend(vehicle_samples)

    return samples


def load_scene_samples(
        df,
        T_h,
        T_f,
        style_lat_cols=None,
        style_long_cols=None,
        motion_cols=None,
        interaction_cols = None,
        scaler=None,
        use_cache=True,
        cache_dir="./scene_cache",
        force_recreate=False,
        verbose=True,
        scene_name=None,
        balance_globally=False,
):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    config = {
        'scene_name': scene_name,
        'T_h': T_h,
        'T_f': T_f,
        'interaction_cols': sorted(interaction_cols) if interaction_cols else None,
        'style_lat_cols': sorted(style_lat_cols) if style_lat_cols else None,
        'style_long_cols': sorted(style_long_cols) if style_long_cols else None,
        'motion_cols': sorted(motion_cols) if motion_cols else None,
        'df_shape': [len(df), len(df.columns)],
        'df_columns': sorted(df.columns.tolist()),
    }

    if scaler is not None and hasattr(scaler.scaler, 'mean_') and scaler.scaler.mean_ is not None:
        config['scaler_mean_sum'] = float(np.sum(scaler.scaler.mean_))
        config['scaler_mean_shape'] = scaler.scaler.mean_.shape[0]
        config['scaler_scale_sum'] = float(np.sum(scaler.scaler.scale_))

    cache_key = get_cache_key(config)

    if scene_name:
        safe_scene_name = "".join(c for c in scene_name if c.isalnum() or c in ('-', '_')).rstrip()
        cache_file = cache_dir / f"samples_{safe_scene_name}_{cache_key}.pkl"
        meta_file = cache_dir / f"samples_{safe_scene_name}_{cache_key}_meta.json"
    else:
        cache_file = cache_dir / f"samples_{cache_key}.pkl"
        meta_file = cache_dir / f"samples_{cache_key}_meta.json"

    if verbose:
        print(f"\n🔑 Scene name: {scene_name if scene_name else 'Not specified'}")
        print(f"🔑 Cache key: {cache_key}")
        print(f"📁 Cache file: {cache_file}")

    if use_cache and not force_recreate and cache_file.exists():
        try:
            if verbose:
                file_size = cache_file.stat().st_size / (1024 * 1024)
                print(f"✅ Cache file found! (Size: {file_size:.2f} MB)")
                print(f"⏳ Loading...")

            start_time = time.time()
            with open(cache_file, 'rb') as f:
                samples = pickle.load(f)
            load_time = time.time() - start_time

            if verbose:
                print(f"✨ Successfully loaded {len(samples)} samples from cache (Time: {load_time:.2f}s)")
                print(f"💾 Cache source: {cache_file}")

            del df
            gc.collect()

            return samples

        except Exception as e:
            print(f"⚠️ Cache loading failed: {e}")
            print("🔄 Rebuilding dataset...")

    if verbose:
        print("🔄 Cache not found, building dataset...")
        start_time = time.time()

    samples = build_samples_from_df_parallel(
        df,
        T_h=T_h,
        T_f=T_f,
        num_workers=None,
        style_lat_cols=style_lat_cols,
        style_long_cols=style_long_cols,
        motion_cols=motion_cols,
        interaction_cols = interaction_cols,
        scaler=scaler
    )

    if verbose:
        build_time = time.time() - start_time
        print(f"✅ Dataset building completed: {len(samples)} samples (Time: {build_time:.2f}s)")

    if use_cache:
        try:
            if verbose:
                print(f"💾 Saving to cache: {cache_file}")

            save_start = time.time()
            with open(cache_file, 'wb') as f:
                pickle.dump(samples, f, protocol=pickle.HIGHEST_PROTOCOL)
            save_time = time.time() - save_start

            metadata = {
                'cache_key': cache_key,
                'scene_name': scene_name,
                'num_samples': len(samples),
                'created_time': time.strftime('%Y-%m-%d %H:%M:%S'),
                'config': config,
                'build_time': build_time,
                'save_time': save_time
            }
            with open(meta_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            if verbose:
                file_size = cache_file.stat().st_size / (1024 * 1024)
                print(f"✅ Cache saved successfully (Size: {file_size:.2f} MB)")

        except Exception as e:
            print(f"⚠️ Cache save failed: {e}")

    del df
    gc.collect()

    return samples


def load_scene_samples1(
        df,
        T_h,
        T_f,
        style_lat_cols=None,
        style_long_cols=None,
        motion_cols=None,
        interaction_cols = None,
        scaler=None,
        use_cache=True,
        cache_dir="./scene_cache",
        force_recreate=False,
        verbose=True,
        scene_name=None
):
    samples = build_samples_from_df_parallel(
        df,
        T_h=T_h,
        T_f=T_f,
        num_workers=None,
        style_lat_cols=style_lat_cols,
        style_long_cols=style_long_cols,
        motion_cols=motion_cols,
        interaction_cols=interaction_cols,
        scaler=scaler
    )

    del df
    gc.collect()

    return samples


def get_cache_key(config):
    import hashlib
    import json

    config_str = json.dumps(config, sort_keys=True, default=str)
    return hashlib.md5(config_str.encode()).hexdigest()


def _get_samples_shape(samples):
    if not samples:
        return {}

    sample = samples[0]
    shapes = {}

    for key in ['hist_motion', 'style_long', 'style_lat', 'future']:
        if key in sample:
            value = sample[key]
            if hasattr(value, 'shape'):
                shapes[key] = list(value.shape)
            elif isinstance(value, (list, tuple)):
                shapes[key] = [len(value)]
            else:
                shapes[key] = str(type(value))

    return shapes


def inspect_scene_cache(cache_dir="./scene_cache"):
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        print(f"Cache directory does not exist: {cache_dir}")
        return

    cache_files = list(cache_dir.glob("samples_*.pkl"))
    meta_files = list(cache_dir.glob("samples_*_meta.json"))

    print(f"\n📁 Scene cache directory: {cache_dir.absolute()}")
    print(f"📊 Statistics:")
    print(f"  - Cache files: {len(cache_files)}")
    print(f"  - Meta files: {len(meta_files)}")

    if cache_files:
        print(f"\n📋 Cache file details:")
        total_size = 0
        for cache_file in sorted(cache_files, key=lambda x: x.stat().st_mtime, reverse=True):
            size_mb = cache_file.stat().st_size / (1024 * 1024)
            modified = time.strftime('%Y-%m-%d %H:%M:%S',
                                     time.localtime(cache_file.stat().st_mtime))
            total_size += size_mb

            meta_file = cache_dir / cache_file.name.replace('.pkl', '_meta.json')
            if meta_file.exists():
                with open(meta_file, 'r') as f:
                    meta = json.load(f)
                print(f"  📄 {cache_file.name}")
                print(f"    Size: {size_mb:.2f} MB, Modified: {modified}")
                print(f"    Samples: {meta.get('num_samples', 'N/A')}")
                print(f"    Created: {meta.get('created_time', 'N/A')}")
            else:
                print(f"  📄 {cache_file.name} (no metadata)")
                print(f"    Size: {size_mb:.2f} MB, Modified: {modified}")

        print(f"\n📦 Total cache size: {total_size:.2f} MB")


def clean_scene_cache(cache_dir="./scene_cache", keep_last=5, older_than_days=None):
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        return

    cache_files = list(cache_dir.glob("samples_*.pkl"))
    cache_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    to_delete = set()

    if len(cache_files) > keep_last:
        to_delete.update(cache_files[keep_last:])

    if older_than_days:
        current_time = time.time()
        for f in cache_files:
            file_time = f.stat().st_mtime
            if (current_time - file_time) > (older_than_days * 24 * 3600):
                to_delete.add(f)

    for f in to_delete:
        print(f"🧹 Deleting: {f.name}")
        f.unlink()

        meta_file = cache_dir / f.name.replace('.pkl', '_meta.json')
        if meta_file.exists():
            meta_file.unlink()

    print(f"✅ Cleanup complete, deleted {len(to_delete)} cache files")


def get_scene_cache_info(cache_dir="./scene_cache"):
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        return {"exists": False}

    cache_files = list(cache_dir.glob("samples_*.pkl"))

    info = {
        "exists": True,
        "directory": str(cache_dir.absolute()),
        "num_caches": len(cache_files),
        "total_size_mb": sum(f.stat().st_size for f in cache_files) / (1024 * 1024),
        "caches": []
    }

    for cache_file in sorted(cache_files, key=lambda x: x.stat().st_mtime, reverse=True)[:3]:
        meta_file = cache_dir / cache_file.name.replace('.pkl', '_meta.json')
        cache_info = {
            "filename": cache_file.name,
            "size_mb": cache_file.stat().st_size / (1024 * 1024),
            "modified": time.strftime('%Y-%m-%d %H:%M:%S',
                                      time.localtime(cache_file.stat().st_mtime))
        }

        if meta_file.exists():
            with open(meta_file, 'r') as f:
                meta = json.load(f)
            cache_info["num_samples"] = meta.get("num_samples")
            cache_info["created_time"] = meta.get("created_time")

        info["caches"].append(cache_info)

    return info