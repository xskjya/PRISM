import gc
import glob
import hashlib
import json
import os
import pickle
import random
import re
import time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from modul.Scaler import DataFrameScaler
from modul.tool import get_cache_key


def _build_samples_for_one_vehicle(args):
    ego_df, T_h, T_f, motion_cols, style_long_cols, style_lat_cols, interaction_cols, scaler, balance_per_vehicle, max_samples_per_vehicle = args

    samples = []
    vehicle_samples = []

    if len(ego_df) < T_h + T_f:
        return None

    vehicle_id = ego_df["id"].iloc[0] if "id" in ego_df.columns else ego_df["trackId"].iloc[0]

    total_frames = len(ego_df)
    available_starts = total_frames - T_h - T_f + 1

    if available_starts <= 0:
        return None

    if available_starts <= max_samples_per_vehicle:
        sample_step = 1
        max_samples = available_starts
    else:
        sample_step = available_starts // max_samples_per_vehicle
        max_samples = max_samples_per_vehicle

    sample_count = 0

    for start_idx in range(0, available_starts, sample_step):
        if sample_count >= max_samples:
            break

        hist_end = start_idx + T_h

        hist_df = ego_df.iloc[start_idx:hist_end]

        future_df = ego_df.iloc[hist_end:hist_end + T_f]

        last_row = hist_df.iloc[-1]
        if scaler is not None:
            try:
                temp_df = last_row.to_frame().T
                last_real = scaler.inverse_transform(temp_df)[
                    ["ego_x", "ego_y", "ego_vx", "ego_vy"]
                ].values[0].astype(np.float32)
            except:
                last_real = last_row[["ego_x", "ego_y", "ego_vx", "ego_vy"]].values.astype(np.float32)
        else:
            last_real = last_row[["ego_x", "ego_y", "ego_vx", "ego_vy"]].values.astype(np.float32)

        if "lane_id" in future_df.columns:
            lane_ids = future_df["lane_id"].values
        else:
            intent = 0
            lane_ids = np.array([0])

        if np.all(lane_ids == lane_ids[0]):
            intent = 0
        else:
            try:
                first_lane = float(lane_ids[0]) if lane_ids[0] is not None else 0
                last_lane = float(lane_ids[-1]) if lane_ids[-1] is not None else 0

                if last_lane < first_lane:
                    intent = 1
                elif last_lane > first_lane:
                    intent = 2
                else:
                    intent = 0
            except (ValueError, TypeError):
                intent = 0

        try:
            if scaler is not None:
                temp_df = future_df.copy()
                future = scaler.inverse_transform(temp_df)[["ego_x", "ego_y" ,"ego_vx", "ego_vy","ego_ax", "ego_ay"]].values.astype(np.float32)
            else:
                future = future_df[["ego_x", "ego_y" ,"ego_vx", "ego_vy","ego_ax", "ego_ay"]].values.astype(np.float32)
        except:
            future = future_df[["ego_x", "ego_y" ,"ego_vx", "ego_vy","ego_ax", "ego_ay"]].values.astype(np.float32)

        vehicle_samples.append({
            "hist_motion": hist_df[motion_cols].values.astype(np.float32) if motion_cols else np.array([]),
            "style_long": hist_df[style_long_cols].values.astype(np.float32) if style_long_cols else np.array([]),
            "style_lat": hist_df[style_lat_cols].values.astype(np.float32) if style_lat_cols else np.array([]),
            "interaction": hist_df[interaction_cols].values.astype(np.float32) if interaction_cols else np.array([]),
            "intent": intent,
            "future": future,
            "vehicle_id": vehicle_id,
            "hist_last": last_real
        })

        sample_count += 1

    if len(vehicle_samples) == 0:
        return None

    intents = [s["intent"] for s in vehicle_samples]
    has_follow = 0 in intents
    has_left = 1 in intents
    has_right = 2 in intents

    if has_follow and not (has_left or has_right):
        vehicle_type = "pure_follow"
    elif not has_follow and (has_left or has_right):
        vehicle_type = "pure_lane_change"
    else:
        vehicle_type = "mixed"

    for sample in vehicle_samples:
        sample["vehicle_type"] = vehicle_type

    if balance_per_vehicle and len(vehicle_samples) > 0:
        follow_samples = [s for s in vehicle_samples if s["intent"] == 0]
        left_samples = [s for s in vehicle_samples if s["intent"] == 1]
        right_samples = [s for s in vehicle_samples if s["intent"] == 2]

        n_follow = len(follow_samples)
        n_left = len(left_samples)
        n_right = len(right_samples)

        if vehicle_type == "pure_follow":
            return vehicle_samples

        if vehicle_type == "pure_lane_change":
            return vehicle_samples

        if n_left + n_right > 0:
            balanced_samples = left_samples + right_samples
            remaining = max_samples_per_vehicle - len(balanced_samples)
            if remaining > 0 and n_follow > 0:
                selected_follow = random.sample(follow_samples, min(remaining, n_follow))
                balanced_samples.extend(selected_follow)

            random.shuffle(balanced_samples)
            return balanced_samples

    return vehicle_samples

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
        balance_per_vehicle=False,
        max_samples_per_vehicle=50
):

    if num_workers is None:
        num_workers = max(os.cpu_count() - 1, 1)

    num_workers = 2

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
            balance_per_vehicle,
            max_samples_per_vehicle
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
        balance_per_vehicle = True,
        max_samples_per_vehicle = 10
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
        scaler=scaler,
        balance_per_vehicle=balance_per_vehicle,
        max_samples_per_vehicle=max_samples_per_vehicle
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



def create_fixed_datasets(
        all_files,
        T_h,
        T_f,
        style_lat_cols,
        style_long_cols,
        motion_cols,
        interaction_cols,
        normal_cols,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        cache_dir="./scene_cache",
        use_cache=True,
        force_recreate=False,
        balance_per_vehicle=True,
        max_samples_per_vehicle = 10,
):
    config = {
        'T_h': T_h,
        'T_f': T_f,
        "balance_per_vehicle":balance_per_vehicle,
        "max_samples_per_vehicle": max_samples_per_vehicle,
        'train_ratio': train_ratio,
        'val_ratio': val_ratio,
        'test_ratio': test_ratio,
        'style_lat_cols': sorted(style_lat_cols) if style_lat_cols else None,
        'style_long_cols': sorted(style_long_cols) if style_long_cols else None,
        'motion_cols': sorted(motion_cols) if motion_cols else None,
        'interaction_cols': sorted(interaction_cols) if interaction_cols else None,
        'normal_cols': sorted(normal_cols) if normal_cols else None,
        'num_files': len(all_files),
        'file_names': [os.path.basename(f) for f in all_files[:5]],
    }

    cache_key = hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()

    cache_file = os.path.join(cache_dir, f"fixed_datasets_{cache_key}.pkl")
    meta_file = os.path.join(cache_dir, f"fixed_datasets_{cache_key}_meta.json")

    print(f"\n{'=' * 60}")
    print(f"Creating fixed datasets...")
    print(f"Cache key: {cache_key}")
    print(f"Cache file: {cache_file}")
    print(f"{'=' * 60}")

    if use_cache and not force_recreate and os.path.exists(cache_file):
        try:
            file_size = os.path.getsize(cache_file) / (1024 * 1024)
            print(f"✅ Cache file found! (Size: {file_size:.2f} MB)")
            print(f"⏳ Loading...")

            start_time = time.time()
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            load_time = time.time() - start_time

            train_samples = data['train_samples']
            val_samples = data['val_samples']
            test_samples = data['test_samples']
            scaler_dict = data['scaler_dict']

            print(f"✨ Successfully loaded from cache (Time: {load_time:.2f}s)")
            print(f"  Total training samples: {len(train_samples)}")
            print(f"  Total validation samples: {len(val_samples)}")
            print(f"  Total test samples: {len(test_samples)}")

            return train_samples, val_samples, test_samples, scaler_dict

        except Exception as e:
            print(f"⚠️ Cache loading failed: {e}")
            print("🔄 Rebuilding dataset...")

    print("🔄 Cache not found, building dataset...")
    build_start_time = time.time()

    all_train_samples = []
    all_val_samples = []
    all_test_samples = []
    scaler_dict = {}

    for file_path in all_files:
        file_name = os.path.basename(file_path)
        file_num = extract_file_number(file_path)
        scene_id = f"scene_{file_num:02d}"

        print(f"\nProcessing file {file_num:02d}: {file_name}")

        df = pd.read_csv(file_path)

        scaler = DataFrameScaler()
        scaler.fit(df, normal_cols)
        df = scaler.transform(df)
        scaler_dict[file_num] = scaler

        samples = load_scene_samples(
            df=df,
            T_h=T_h,
            T_f=T_f,
            style_lat_cols=style_lat_cols,
            style_long_cols=style_long_cols,
            motion_cols=motion_cols,
            interaction_cols=interaction_cols,
            scaler=scaler,
            use_cache=use_cache,
            cache_dir=cache_dir,
            force_recreate=False,
            verbose=False,
            scene_name=file_name,
            balance_per_vehicle =balance_per_vehicle,
            max_samples_per_vehicle=max_samples_per_vehicle
        )

        print(f"  Total samples: {len(samples)}")

        for sample in samples:
            sample['scene_id'] = scene_id
            sample['file_num'] = file_num

        train_s, val_s, test_s = split_samples_by_vehicle(
            samples, train_ratio, val_ratio, test_ratio
        )

        print(f"  Split: train={len(train_s)}, val={len(val_s)}, test={len(test_s)}")

        all_train_samples.extend(train_s)
        all_val_samples.extend(val_s)
        all_test_samples.extend(test_s)

        del df
        gc.collect()

    build_time = time.time() - build_start_time

    print(f"\n{'=' * 60}")
    print(f"Dataset building completed (Time: {build_time:.2f}s):")
    print(f"  Total training samples: {len(all_train_samples)}")
    print(f"  Total validation samples: {len(all_val_samples)}")
    print(f"  Total test samples: {len(all_test_samples)}")
    print(f"{'=' * 60}\n")

    if use_cache:
        try:
            print(f"💾 Saving to cache: {cache_file}")
            save_start = time.time()

            data = {
                'train_samples': all_train_samples,
                'val_samples': all_val_samples,
                'test_samples': all_test_samples,
                'scaler_dict': scaler_dict
            }

            with open(cache_file, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

            save_time = time.time() - save_start
            file_size = os.path.getsize(cache_file) / (1024 * 1024)

            metadata = {
                'cache_key': cache_key,
                'created_time': time.strftime('%Y-%m-%d %H:%M:%S'),
                'build_time': build_time,
                'save_time': save_time,
                'file_size_mb': file_size,
                'config': config,
                'stats': {
                    'num_train': len(all_train_samples),
                    'num_val': len(all_val_samples),
                    'num_test': len(all_test_samples)
                }
            }
            with open(meta_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            print(f"✅ Cache saved successfully (Size: {file_size:.2f} MB, Time: {save_time:.2f}s)")

        except Exception as e:
            print(f"⚠️ Cache save failed: {e}")

    return all_train_samples, all_val_samples, all_test_samples, scaler_dict


def extract_file_number(filename):
    basename = os.path.basename(filename)
    match = re.match(r'(\d+)', basename)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"Cannot extract number from filename {basename}")

def split_samples_by_vehicle(samples, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    vehicle_ids = list(set(s["vehicle_id"] for s in samples))

    np.random.seed(seed)
    random.seed(seed)

    np.random.shuffle(vehicle_ids)

    n_vehicles = len(vehicle_ids)
    train_end = int(n_vehicles * train_ratio)
    val_end = train_end + int(n_vehicles * val_ratio)

    train_vid = set(vehicle_ids[:train_end])
    val_vid = set(vehicle_ids[train_end:val_end])
    test_vid = set(vehicle_ids[val_end:])

    train_samples = [s for s in samples if s["vehicle_id"] in train_vid]
    val_samples = [s for s in samples if s["vehicle_id"] in val_vid]
    test_samples = [s for s in samples if s["vehicle_id"] in test_vid]

    return train_samples, val_samples, test_samples

def clean_fixed_datasets_cache(cache_dir="./scene_cache", keep_last=5):
    cache_files = glob.glob(os.path.join(cache_dir, "fixed_datasets_*.pkl"))
    meta_files = glob.glob(os.path.join(cache_dir, "fixed_datasets_*_meta.json"))

    cache_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)

    if len(cache_files) > keep_last:
        for f in cache_files[keep_last:]:
            os.remove(f)
            print(f"Deleted cache: {os.path.basename(f)}")

            meta_file = f.replace('.pkl', '_meta.json')
            if os.path.exists(meta_file):
                os.remove(meta_file)

    print(f"Cleanup completed, keeping last {keep_last} cache files")