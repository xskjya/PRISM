import csv
import glob
import os
import random
import datetime
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from modul.Dataset import DrivingDataset, collate_fn
from modul.build_data import create_fixed_datasets
from modul.plot_tool import plot_forgetting_curve
from modul.tool import init_weights, plot_trajectory_compared
from modul.warmupScheduler import WarmupScheduler
from modul.model import PRISM

def train_epoch(
        model,
        loader,
        optimizer,
        device,
        T_f,
        epoch,
        traj_loss_weight=1.0,
        intent_loss_weight=0.3,
        prob_loss_weight=0.1,
        grad_clip=1.0,
        return_lr=False,
        scaler1=None,
        diversity_weight=0.1,
        min_modal_distance=2.0,
):
    model.train()

    total_loss = 0.0
    total_traj = 0.0
    total_int = 0.0
    total_prob = 0.0
    total_distance = 0.0

    n_batches = 0
    current_lr = optimizer.param_groups[0]["lr"]

    pbar = tqdm(loader, desc=f"Train Epoch {epoch}", leave=False)

    for batch_idx, batch in enumerate(pbar):
        hist, s_long, s_lat, intent, future, interaction, hist_last = batch

        hist = torch.stack(hist).to(device)
        s_long = torch.stack(s_long).to(device)
        s_lat = torch.stack(s_lat).to(device)
        intent = torch.stack(intent).to(device)
        future = torch.stack(future).to(device)
        interaction = torch.stack(interaction).to(device)
        hist_last = torch.stack(hist_last).to(device)

        if future.shape[-1] > 2:
            future = future[..., :2]


        with torch.amp.autocast(device_type=device, enabled=(scaler1 is not None)):
            pred, prob, intent_logits, prob_logits = model(
                hist, interaction, s_long, s_lat, hist_last=hist_last
            )

        B, K, T, _ = pred.shape

        diff = pred - future.unsqueeze(1)
        traj_error = torch.norm(diff, dim=-1).mean(dim=-1)

        min_error, min_index = torch.min(traj_error, dim=1)
        batch_indices = torch.arange(B, device=device)
        best_pred = pred[batch_indices, min_index]

        traj_loss = F.smooth_l1_loss(best_pred, future, reduction='mean')

        avg_modal_distance = torch.tensor(0.0, device=device)
        modal_diversity_loss = torch.tensor(0.0, device=device)

        if K > 1 and diversity_weight > 0:
            endpoints = pred[:, :, -1, :]
            endpoint_dists = torch.cdist(endpoints, endpoints)
            mask = 1 - torch.eye(K, device=device)
            mask = mask.unsqueeze(0).expand(B, -1, -1)
            avg_endpoint_dist = (endpoint_dists * mask).sum() / (mask.sum() + 1e-6)

            traj_flat = pred.reshape(B, K, -1)
            traj_dists = torch.cdist(traj_flat, traj_flat)
            avg_traj_dist = (traj_dists * mask).sum() / (mask.sum() + 1e-6)


            avg_modal_distance = (avg_endpoint_dist + avg_traj_dist) / 2

            modal_diversity_loss = torch.relu(min_modal_distance - avg_modal_distance)

            extra_push = -0.01 * torch.tanh(avg_modal_distance / min_modal_distance)
            modal_diversity_loss = modal_diversity_loss + extra_push

        prob_loss = F.cross_entropy(prob_logits, min_index)

        intent_loss = F.cross_entropy(intent_logits, intent)

        loss = (
                traj_loss_weight * traj_loss
                + intent_loss_weight * intent_loss
                + diversity_weight * modal_diversity_loss
        )

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"⚠️ Warning: Loss is NaN/Inf at batch {batch_idx}, skipping...")
            continue

        optimizer.zero_grad()

        if scaler1 is not None:
            scaler1.scale(loss).backward()
            if grad_clip is not None:
                scaler1.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler1.step(optimizer)
            scaler1.update()
        else:
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += loss.item()
        total_traj += traj_loss.item()
        total_int += intent_loss.item()
        total_prob += prob_loss.item()
        total_distance += avg_modal_distance.item()

        n_batches += 1

        pbar.set_postfix({
            "loss": f"{loss.item():.3f}",
            "traj": f"{traj_loss.item():.3f}",
            "dist": f"{avg_modal_distance.item():.3f}",
            "lr": f"{current_lr:.6f}"
        })


    if n_batches == 0:
        print(f"⚠️ Warning: No valid batches in epoch {epoch}")
        avg_loss = float('nan')
        avg_traj = float('nan')
        avg_int = float('nan')
        avg_prob = float('nan')
        avg_distance = float('nan')
    else:
        avg_loss = total_loss / n_batches
        avg_traj = total_traj / n_batches
        avg_int = total_int / n_batches
        avg_prob = total_prob / n_batches
    outputs = (avg_loss, avg_traj, avg_int, avg_prob)
    if return_lr:
        return outputs + (current_lr,)
    else:
        return outputs


def eval_epoch(
        model,
        loader,
        device,
        T_f,
        compute_fde=False,
        eval_horizons=(1, 2, 3, 4),
        fps=25,
        scaler_dict=None,
        scene_name=None,
        return_detailed_metrics=False,
        trajectory_compared_dir="trajectory_compared_dir",
):
    model.eval()

    correct = 0
    intent_total = 0

    ae_sum = 0.0
    ae_count = 0

    lateral_ae_sum = 0.0
    longitudinal_ae_sum = 0.0
    error_count = 0

    ae_horizon_sum = {h: 0.0 for h in eval_horizons}
    ae_horizon_count = {h: 0 for h in eval_horizons}

    lateral_horizon_sum = {h: 0.0 for h in eval_horizons}
    longitudinal_horizon_sum = {h: 0.0 for h in eval_horizons}

    fde_sum = 0.0
    fde_count = 0
    n_batches = 0

    all_modes_ade_sum = []
    K = None

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            hist, s_long, s_lat, intent, future, interaction, hist_last, scene_info = batch

            hist = torch.stack(hist).to(device)
            s_long = torch.stack(s_long).to(device)
            s_lat = torch.stack(s_lat).to(device)
            intent = torch.stack(intent).to(device)
            future = torch.stack(future).to(device)
            interaction = torch.stack(interaction).to(device)
            hist_last = torch.stack(hist_last).to(device)

            if future.shape[-1] > 2:
                future = future[..., :2]

            pred, prob, intent_logits, prob_logits = model(
                hist, interaction, s_long, s_lat, hist_last=hist_last
            )

            if K is None:
                K = pred.shape[1]
                all_modes_ade_sum = [0.0] * K

            for k in range(K):
                mode_pred = pred[:, k, :, :]
                mode_ade = torch.norm(mode_pred - future, dim=-1).mean().item()
                all_modes_ade_sum[k] += mode_ade * future.shape[0]

            if batch_idx == 0:
                first_batch_errors = []
                for k in range(K):
                    mode_pred = pred[:, k, :, :]
                    mode_ade = torch.norm(mode_pred - future, dim=-1).mean().item()
                    first_batch_errors.append(mode_ade)

            traj_error = torch.norm(
                pred - future.unsqueeze(1),
                dim=-1
            )

            traj_error = traj_error.mean(dim=-1)
            _, best_k = torch.min(traj_error, dim=1)
            batch_indices = torch.arange(pred.size(0), device=pred.device)
            pred_best = pred[batch_indices, best_k]

            pred_intent = intent_logits.argmax(dim=-1)
            correct += (pred_intent == intent).sum().item()
            intent_total += intent.size(0)

            longitudinal_error = torch.abs(pred_best[..., 0] - future[..., 0])
            lateral_error = torch.abs(pred_best[..., 1] - future[..., 1])

            ae_per_step = (longitudinal_error + lateral_error) / 2

            ae_sum += ae_per_step.sum().item()
            ae_count += ae_per_step.numel()

            lateral_ae_sum += lateral_error.sum().item()
            longitudinal_ae_sum += longitudinal_error.sum().item()
            error_count += lateral_error.numel()

            for h in eval_horizons:
                t = min(int(h * fps), T_f)

                ae_h_per_step = (
                    torch.abs(pred_best[:, :t, 0] - future[:, :t, 0]) +
                    torch.abs(pred_best[:, :t, 1] - future[:, :t, 1])
                ) / 2
                ae_horizon_sum[h] += ae_h_per_step.sum().item()
                ae_horizon_count[h] += ae_h_per_step.numel()

                lateral_h_per_step = torch.abs(pred_best[:, :t, 1] - future[:, :t, 1])
                lateral_horizon_sum[h] += lateral_h_per_step.sum().item()

                longitudinal_h_per_step = torch.abs(pred_best[:, :t, 0] - future[:, :t, 0])
                longitudinal_horizon_sum[h] += longitudinal_h_per_step.sum().item()

            if compute_fde:
                fde = torch.norm(pred_best[:, -1] - future[:, -1], dim=-1)
                fde_sum += fde.sum().item()
                fde_count += fde.numel()

            n_batches += 1
            intent_cpu = intent.cpu().numpy()
            lane_change_indices = np.where((intent_cpu == 1) | (intent_cpu == 2))[0]

            if len(lane_change_indices) > 0:
                first_lane_change_idx = lane_change_indices[0]

                if first_lane_change_idx < len(scene_info):
                    sample_scene_info = scene_info[first_lane_change_idx]
                else:
                    sample_scene_info = {}

                file_num = sample_scene_info.get('file_num', -1)
                intent_value = intent_cpu[first_lane_change_idx]
                intent_type = "left" if intent_value == 1 else "right"

                if scaler_dict is not None and file_num in scaler_dict:
                    scene_scaler = scaler_dict[file_num]
                else:
                    scene_scaler = None

                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{scene_name}_{intent_type}_{timestamp}.png" if scene_name else f"trajectory_{intent_type}_{timestamp}.png"

                hist_sample = hist[first_lane_change_idx:first_lane_change_idx + 1]
                future_sample = future[first_lane_change_idx:first_lane_change_idx + 1]
                pred_sample = pred_best[first_lane_change_idx:first_lane_change_idx + 1]

                save_path = plot_trajectory_compared(
                    hist_sample,
                    future_sample,
                    pred_sample,
                    scene_scaler,
                    save_dir=trajectory_compared_dir,
                    filename=filename
                )

    intent_acc = correct / intent_total if intent_total > 0 else 0.0
    ae_avg = ae_sum / ae_count if ae_count > 0 else 0.0

    lateral_ae_avg = lateral_ae_sum / error_count if error_count > 0 else 0.0
    longitudinal_ae_avg = longitudinal_ae_sum / error_count if error_count > 0 else 0.0


    total_samples = ae_count // T_f if ae_count > 0 else 0
    if total_samples > 0:
        avg_mode_ades = [s / total_samples for s in all_modes_ade_sum]
        if len(avg_mode_ades) > 1:
            ade_std = np.std(avg_mode_ades)

    ae_horizon_avg = {}
    for h in eval_horizons:
        if ae_horizon_count[h] > 0:
            ae_horizon_avg[f"AE@{h}s"] = ae_horizon_sum[h] / ae_horizon_count[h]
        else:
            ae_horizon_avg[f"AE@{h}s"] = 0.0

    lateral_horizon_avg = {}
    longitudinal_horizon_avg = {}
    for h in eval_horizons:
        if ae_horizon_count[h] > 0:
            lateral_horizon_avg[f"LateralAE@{h}s"] = lateral_horizon_sum[h] / ae_horizon_count[h]
            longitudinal_horizon_avg[f"LongAE@{h}s"] = longitudinal_horizon_sum[h] / ae_horizon_count[h]

    if compute_fde:
        fde_avg = fde_sum / fde_count if fde_count > 0 else 0.0

        if return_detailed_metrics:
            return intent_acc, ae_avg, ae_horizon_avg, fde_avg, {
                'lateral_ae': lateral_ae_avg,
                'longitudinal_ae': longitudinal_ae_avg,
                'lateral_horizon': lateral_horizon_avg,
                'longitudinal_horizon': longitudinal_horizon_avg
            }
        else:
            return intent_acc, ae_avg, ae_horizon_avg, fde_avg
    else:
        if return_detailed_metrics:
            return intent_acc, ae_avg, ae_horizon_avg, {
                'lateral_ae': lateral_ae_avg,
                'longitudinal_ae': longitudinal_ae_avg,
                'lateral_horizon': lateral_horizon_avg,
                'longitudinal_horizon': longitudinal_horizon_avg
            }
        else:
            return intent_acc, ae_avg, ae_horizon_avg


if __name__ == "__main__":
    name = "experiment"

    motion_cols = [
        "ego_x", "ego_y",
        "ego_vx", "ego_vy",
        "ego_ax", "ego_ay",
        "ego_delta"
    ]


    interaction_cols = [
        "has_preceding",
        "preceding_gap",
        "ttc_preceding",


        "has_following",
        "following_gap",
        "ttc_following",

        "has_left_vehicle",
        "left_lat_gap",
        "left_gap",
        "left_delta_v",

        "has_right_vehicle",
        "right_lat_gap",
        "right_gap",
        "right_delta_v",
    ]


    style_long_cols = [
        "v_tilde",
        "delta_v_flow",
        "thw",
        "acc_norm",
    ]


    style_lat_cols = [
        "max_vy",
        "vy_sign_change",
        "intent_lead_time",
        "lane_change_duration",
    ]


    BINARY_FEATURES = [
        'brake_event',
        'is_unlimited',
        'vy_sign_change',
        'has_preceding',
        'has_following',
        'has_left_vehicle',
        'has_right_vehicle',
    ]

    normal_cols = [col for col in (motion_cols + interaction_cols + style_lat_cols + style_long_cols)
                   if col not in BINARY_FEATURES]
    motion_dim = len(motion_cols)
    long_dim = len(style_long_cols)
    lat_dim = len(style_lat_cols)


    data_dir = "./data/feature"
    save_dir     = Path("models")
    log_dir      = Path(f"logs")
    cache_dir    = Path(f"scene_cache")
    trajectory_compared_dir = Path(f"trajectory_compared_val/{name}")
    test_trajectory_compared_dir = Path(f"trajectory_compared_test/{name}")


    base_lr = 3e-4
    batch_size = 1000  #
    fps = 25
    K = 3

    T_h = fps * 3
    T_f = fps * 5
    eval_horizons = [1, 2, 3, 4]

    max_epochs = 200
    patience = 20
    warmup_epochs = 5
    warmup_strategy = "linear"


    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15

    balance_per_vehicle = True
    max_samples_per_vehicle = 10


    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    trajectory_compared_dir.mkdir(parents=True, exist_ok=True)

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


    device = "cuda" if torch.cuda.is_available() else "cpu"
    pin_memory = (device == "cuda")

    all_files = sorted(
        glob.glob(os.path.join(data_dir, "*_tracks_extract_feature.csv"))
    )


    train_samples, val_samples, test_samples, scaler_dict = create_fixed_datasets(
        all_files=all_files,
        T_h=T_h,
        T_f=T_f,
        style_lat_cols=style_lat_cols,
        style_long_cols=style_long_cols,
        motion_cols=motion_cols,
        interaction_cols=interaction_cols,
        normal_cols=normal_cols,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        cache_dir=cache_dir,
        use_cache=True,
        force_recreate=False,
        balance_per_vehicle=balance_per_vehicle ,
        max_samples_per_vehicle = max_samples_per_vehicle
    )


    train_dataset = DrivingDataset(
        train_samples,
        return_scene_info=False
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=True
    )

    val_dataset = DrivingDataset(
        val_samples,
        return_scene_info=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )

    test_dataset = DrivingDataset(
        test_samples,
        return_scene_info=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )


    resume_path = os.path.join(save_dir, f"resume_training.pth")
    start_epoch = 0
    global_step = 0

    if device == "cuda":
        scaler1 = torch.amp.GradScaler('cuda')
    else:
        scaler1 = None

    model = PRISM(
        motion_dim=motion_dim,
        long_dim=long_dim,
        lat_dim=lat_dim,
        interaction_dim=len(interaction_cols),
        T_f=T_f,
        K=K
    ).to(device)

    init_weights(model, mode="kaiming")

    optimizer = optim.Adam(model.parameters(), lr=base_lr)

    main_scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
    )

    warmup_scheduler = WarmupScheduler(
        optimizer=optimizer,
        warmup_epochs=warmup_epochs,
        base_lr=base_lr,
        warmup_strategy=warmup_strategy
    )

    best_val_ade = float("inf")
    best_state = None
    validation_history = []
    patience_counter = 0


    if os.path.exists(resume_path):
        try:
            checkpoint = torch.load(resume_path, map_location=device,weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            if 'main_scheduler_state_dict' in checkpoint:
                main_scheduler.load_state_dict(checkpoint['main_scheduler_state_dict'])
            if 'warmup_scheduler_state_dict' in checkpoint:
                warmup_scheduler.load_state_dict(checkpoint['warmup_scheduler_state_dict'])

            start_epoch = checkpoint.get('epoch', 0) + 1
            global_step = checkpoint.get('global_step', 0)
            patience_counter = checkpoint.get('patience_counter', 0)

            best_val_ade = checkpoint.get('best_val_ade', float("inf"))
            best_state = checkpoint.get('best_state', None)
            validation_history = checkpoint.get('validation_history', [])

        except Exception as e:
            start_epoch = 0
            global_step = 0
            patience_counter = 0
            best_val_ade = float("inf")
            best_state = None
            validation_history = []

    writer = SummaryWriter(
        log_dir=os.path.join(log_dir, f"TensorBoard_{name}"),
        purge_step=global_step
    )

    csv_path = os.path.join(log_dir, f"training_log.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow([
                "epoch",
                "train_loss", "traj_loss", "intent_loss",
                "current_lr", "val_ade", "val_fde", "intent_acc"
            ])


    prev_lr = None

    for epoch in range(start_epoch, max_epochs):
        if epoch < warmup_epochs:
            current_lr = warmup_scheduler.step(epoch)
        else:
            current_lr = optimizer.param_groups[0]["lr"]

        prev_lr = current_lr

        if epoch < 10:
            diversity_weight = 0.01
            min_modal_distance = 1.0
        elif epoch < 30:
            diversity_weight = 0.05
            min_modal_distance = 1.5
        else:
            diversity_weight = 0.1
            min_modal_distance = 2.0

        train_loss, traj_loss, intent_loss, prob_loss = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            T_f=T_f,
            epoch=epoch,
            scaler1=scaler1,
            diversity_weight=diversity_weight,
            min_modal_distance=min_modal_distance,
        )

        intent_acc_val, val_ade, ade_dict, fde_avg, detailed = eval_epoch(
            model=model,
            loader=val_loader,
            device=device,
            T_f=T_f,
            compute_fde=True,
            eval_horizons=eval_horizons,
            fps=fps,
            scaler_dict=scaler_dict,
            scene_name=f"epoch_{epoch}",
            return_detailed_metrics=True,
            trajectory_compared_dir = str(trajectory_compared_dir)
        )

        if epoch >= warmup_epochs:
            main_scheduler.step(val_ade)

        if val_ade < best_val_ade:
            best_val_ade = val_ade
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            torch.save(best_state, os.path.join(save_dir, f"best_model_{epoch}.pth"))
            patience_counter = 0
        else:
            patience_counter += 1

        validation_history.append({
            'epoch': epoch,
            'val_ade': val_ade,
            'val_fde': fde_avg,
            'intent_acc': intent_acc_val,
        })


        global_step += 1
        writer.add_scalar("Loss/train", train_loss, global_step)
        writer.add_scalar("Loss/traj", traj_loss, global_step)
        writer.add_scalar("Loss/intent", intent_loss, global_step)
        writer.add_scalar("LR", current_lr, global_step)
        writer.add_scalar("Val/ADE", val_ade, global_step)
        writer.add_scalar("Val/FDE", fde_avg, global_step)
        writer.add_scalar("Val/Intent_Acc", intent_acc_val, global_step)
        writer.add_scalar("Val/Forgetting", 0, global_step)


        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch,
                train_loss, traj_loss, intent_loss,
                current_lr, val_ade, fde_avg, intent_acc_val
            ])


        if epoch % 10 == 0:
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'main_scheduler_state_dict': main_scheduler.state_dict(),
                'warmup_scheduler_state_dict': warmup_scheduler.state_dict(),
                'epoch': epoch,
                'global_step': global_step,
                'best_val_ade': best_val_ade,
                'best_state': best_state,
                'validation_history': validation_history,
                'patience_counter': patience_counter,
            }
            torch.save(checkpoint, resume_path)


        warmup_flag = "🔥" if epoch < warmup_epochs else "✅"
        ade_str = " | ".join(
            [f"{k}={v:.4f}" for k, v in ade_dict.items()]
        )

        print(
            f"[Epoch {epoch:02d}] "
            f"Loss={train_loss:.4f} | "
            f"Traj={traj_loss:.4f} | "
            f"Val ADE={val_ade:.4f} ({ade_str})| "
            f"Intent={intent_acc_val:.4f} | "
            f"LR={current_lr:.8f}"
        )

        if patience_counter >= patience:
            break

        if epoch >= max_epochs - 1:
            break

    if best_state is not None:
        best_epoch = next((h['epoch'] for h in validation_history if h['val_ade'] == best_val_ade), '?')

    plot_forgetting_curve(validation_history, save_dir, 0)

    torch.save(validation_history, f'{save_dir}/validation_history.pth')

    if os.path.exists(resume_path):
        os.remove(resume_path)

    writer.close()

    if best_state is not None:
        final_path = os.path.join(save_dir, "final_best_model1.pth")
        torch.save(
            {
                "model": best_state,
                "best_val_ade": best_val_ade,
                "T_h": T_h,
                "T_f": T_f,
                "base_lr": base_lr,
                "train_ratio": train_ratio,
                "val_ratio": val_ratio,
                "test_ratio": test_ratio,
            },
            final_path
        )

    if best_state is not None:
        model = PRISM(
            motion_dim=motion_dim,
            long_dim=long_dim,
            lat_dim=lat_dim,
            interaction_dim=len(interaction_cols),
            T_f=T_f,
            K=K
        ).to(device)
        model.load_state_dict(best_state)
        model.eval()

        intent_acc, test_ade, ade_dict, fde_avg, detailed_metrics = eval_epoch(
            model=model,
            loader=test_loader,
            device=device,
            T_f=T_f,
            compute_fde=True,
            eval_horizons=eval_horizons,
            fps=fps,
            scene_name="test_set",
            scaler_dict=scaler_dict,
            return_detailed_metrics=True,
            trajectory_compared_dir = str(test_trajectory_compared_dir)
        )

        test_results = {
            "test_ade": test_ade,
            "test_fde": fde_avg,
            "intent_acc": intent_acc,
            "ade_horizon": ade_dict,
            "lateral_ae": detailed_metrics['lateral_ae'],
            "longitudinal_ae": detailed_metrics['longitudinal_ae'],
        }
        test_results_path = os.path.join(save_dir, f"test_results_{timestamp}.pth")
        torch.save(test_results, test_results_path)