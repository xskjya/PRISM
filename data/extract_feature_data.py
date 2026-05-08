import multiprocessing
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate
from tool.read_csv import *
import os
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as patches
import matplotlib.transforms as transforms
from matplotlib.widgets import Button, Slider
import matplotlib.patheffects as pe



Z_MAP        = 0
Z_LANE       = 1
Z_TRAJ       = 2
Z_VEHICLE    = 3
Z_NEIGHBOR   = 3
Z_INTENT_PT  = 5
Z_INTENT_TXT = 6


def mark_intent_frame(
    ax,
    x,
    y,
    label="Intent Onset",
    color="red",
    vehicle_height=4.5
):
    ax.scatter(
        x, y,
        s=10,
        color=color,
        zorder=Z_INTENT_PT
    )

    text_y = y - vehicle_height - 1.0

    ax.annotate(
        label,
        xy=(x, y),
        xytext=(x, text_y),
        ha="center",
        va="top",
        fontsize=8,
        color=color,
        zorder=Z_INTENT_TXT,
        arrowprops=dict(
            arrowstyle="->",
            linewidth=1.2,
            color=color
        ),
        path_effects=[
            pe.withStroke(linewidth=3, foreground="white")
        ]
    )


def visualize_segment(
        tracks, window_df, meta_row, save_path,
        lc_type, lane_markings,
        map_image_path,
        frame=0,
        start_movement=10,
        change_lc_fram=0
):
    fig, ax = plt.subplots(1, 1, figsize=(32, 4))
    plt.subplots_adjust(left=0.02, right=0.98, bottom=0.20, top=0.95)

    MAP_X_MIN = 0
    MAP_X_MAX = 420
    MAP_Y_MIN = 0
    MAP_Y_MAX = 38

    draw_map_background(ax, map_image_path, MAP_X_MIN, MAP_X_MAX, MAP_Y_MIN, MAP_Y_MAX)

    draw_lane_lines(ax, lane_markings, MAP_X_MIN, MAP_X_MAX, color="white")

    x = window_df[X].values
    y = window_df[Y].values

    strategy_point = window_df[FRAME].tolist().index(frame)

    ax.plot(x[:strategy_point], y[:strategy_point], color="orange", linewidth=2, label="History", linestyle="-")
    ax.plot(x[strategy_point + 1:], y[strategy_point + 1:], color="red", linewidth=1.5, label="Future", linestyle=":")
    ax.scatter(x[strategy_point], y[strategy_point], s=10, color="red", zorder=2)

    ego_delta = np.arctan2(window_df[Y_VELOCITY].iloc[strategy_point],
                           window_df[X_VELOCITY].iloc[strategy_point])
    print(f"ego_delta={ego_delta}")
    w, h = window_df[WIDTH].iloc[strategy_point], window_df[HEIGHT].iloc[strategy_point]
    print("---------------------------------------")
    if window_df[X].iloc[-1] > window_df[X].iloc[0]:
        print("X coordinate increasing, vehicle moving forward")
        direction = 1
    else:
        print("X coordinate decreasing, vehicle moving backward")
        direction = -1
    draw_bbox(ax, x[strategy_point], y[strategy_point], w, h,
              yaw=ego_delta, color="red",
              direction=direction
    )

    mark_intent_frame(
        ax,
        x[start_movement],
        y[start_movement],
        label="Intent",
        color="green"
    )
    change_row = window_df[window_df[FRAME] == change_lc_fram]
    if not change_row.empty:
        mark_intent_frame(
            ax,
            change_row[X].iloc[0],
            change_row[Y].iloc[0],
            label="Change",
            color="red"
        )

    neighbor_cols = [
        PRECEDING_ID, FOLLOWING_ID,
        LEFT_PRECEDING_ID, LEFT_ALONGSIDE_ID, LEFT_FOLLOWING_ID,
        RIGHT_PRECEDING_ID, RIGHT_ALONGSIDE_ID, RIGHT_FOLLOWING_ID
    ]

    for col in neighbor_cols:
        nid = int(meta_row.get(col, pd.Series([0])).iloc[0])
        if nid == 0:
            continue

        neighbor_aim_info = tracks[(tracks[ID] == nid) & (tracks[FRAME] == frame)]
        if neighbor_aim_info.empty:
            continue

        neighbor_delta = np.arctan2(neighbor_aim_info[Y_VELOCITY].values[0],
                                    neighbor_aim_info[X_VELOCITY].values[0])
        if col == PRECEDING_ID:
            draw_bbox(ax, neighbor_aim_info[X].values[0], neighbor_aim_info[Y].values[0],
                      neighbor_aim_info[WIDTH].values[0], neighbor_aim_info[HEIGHT].values[0],
                      yaw=neighbor_delta, color="pink", direction=direction)
        else:
            draw_bbox(ax, neighbor_aim_info[X].values[0], neighbor_aim_info[Y].values[0],
                      neighbor_aim_info[WIDTH].values[0], neighbor_aim_info[HEIGHT].values[0],
                      yaw=neighbor_delta, color="yellow", direction=direction)

    ax.set_xlim(MAP_X_MIN, MAP_X_MAX)
    ax.set_ylim(MAP_Y_MAX, MAP_Y_MIN)

    ax.set_title(f"Lane Change (Type={lc_type})", fontsize=10)
    ax.legend()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def visualize_segment_interactive(
        tracks, window_df, meta_row_df, lc_type, lane_markings,
        map_image_path, autoplay_interval=200
, start_movement=10,
    change_lc_fram=0
):
    MAP_X_MIN, MAP_X_MAX = 0, 420
    MAP_Y_MIN, MAP_Y_MAX = 0, 38
    num_frames = len(window_df)

    x = window_df[X].values
    y = window_df[Y].values

    fig, ax = plt.subplots(figsize=(32, 4))
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.25)

    ax.set_xlim(MAP_X_MIN, MAP_X_MAX)
    ax.set_ylim(MAP_Y_MAX, MAP_Y_MIN)
    ax.set_title(f"Lane Change (Type={lc_type})", fontsize=12)

    draw_map_background(ax, map_image_path, MAP_X_MIN, MAP_X_MAX, MAP_Y_MIN, MAP_Y_MAX)
    draw_lane_lines(ax, lane_markings, MAP_X_MIN, MAP_X_MAX, color="white")

    history_line, = ax.plot([], [], color="orange", linewidth=2, label="History")
    future_line, = ax.plot([], [], color="red", linewidth=1.5, linestyle=":", label="Future")
    ego_dot = ax.scatter([], [], s=30, color="red", zorder=2)

    bbox_artists = []

    neighbor_cols = [
        PRECEDING_ID, FOLLOWING_ID,
        LEFT_PRECEDING_ID, LEFT_ALONGSIDE_ID, LEFT_FOLLOWING_ID,
        RIGHT_PRECEDING_ID, RIGHT_ALONGSIDE_ID, RIGHT_FOLLOWING_ID
    ]

    current_frame_idx = [0]
    autoplay = [False]

    mark_intent_frame(
        ax,
        x[start_movement],
        y[start_movement],
        label="Intent",
        color="green"
    )
    change_row = window_df[window_df[FRAME] == change_lc_fram]
    if not change_row.empty:
        mark_intent_frame(
            ax,
            change_row[X].iloc[0],
            change_row[Y].iloc[0],
            label="Change",
            color="red"
        )

    def update(frame_idx):
        for rect in bbox_artists:
            rect.remove()
        bbox_artists.clear()

        history_line.set_data(x[:frame_idx + 1], y[:frame_idx + 1])
        if frame_idx + 1 < len(x):
            future_line.set_data(x[frame_idx + 1:], y[frame_idx + 1:])
        else:
            future_line.set_data([], [])

        ego_dot.set_offsets([[x[frame_idx], y[frame_idx]]])

        ego_delta = np.arctan2(window_df[Y_VELOCITY].iloc[frame_idx], window_df[X_VELOCITY].iloc[frame_idx])
        w, h = window_df[WIDTH].iloc[frame_idx], window_df[HEIGHT].iloc[frame_idx]

        print("---------------------------------------")
        if window_df[X].iloc[-1] > window_df[X].iloc[0]:
            print("X coordinate increasing, vehicle moving forward")
            direction = 1
        else:
            print("X coordinate decreasing, vehicle moving backward")
            direction = -1
        rect = draw_bbox(ax, x[frame_idx], y[frame_idx], w, h,
                         yaw=ego_delta, color="red",
                         direction= direction
                         )
        bbox_artists.append(rect)

        frame_number = window_df[FRAME].iloc[frame_idx]
        print(f"ego={meta_row_df[ID].values[0]} frame={frame_number} ego_delta={ego_delta}")

        print_rows = []
        for col in neighbor_cols:
            ego_dot_temp = window_df.iloc[frame_idx]
            nid = int(ego_dot_temp.get(col, pd.Series([0])))
            if nid == 0:
                continue
            neighbor_aim_info = tracks[(tracks[ID] == nid) & (tracks[FRAME] == frame_number)]
            if neighbor_aim_info.empty:
                continue
            delta = np.arctan2(neighbor_aim_info[Y_VELOCITY].values[0],
                               neighbor_aim_info[X_VELOCITY].values[0])
            color = "pink" if col == PRECEDING_ID else "yellow"
            rect = draw_bbox(ax, neighbor_aim_info[X].values[0], neighbor_aim_info[Y].values[0],
                             neighbor_aim_info[WIDTH].values[0], neighbor_aim_info[HEIGHT].values[0],
                             yaw=delta, color=color, direction=direction)
            bbox_artists.append(rect)

            print_rows.append([
                col.replace('Id', ''),
                nid,
                neighbor_aim_info[FRAME].values[0],
                delta
            ])

        print(tabulate(
            print_rows,
            headers=["Neighbor Type", "ID", "Frame", "Delta"],
            tablefmt="pretty"
        ))

        fig.canvas.draw_idle()
        current_frame_idx[0] = frame_idx

    def update2(frame_idx):
        for rect in bbox_artists:
            rect.remove()
        bbox_artists.clear()

        delta_idx = frame_idx - start_movement

        if delta_idx < -25:
            phase = "Perception"
            phase_color = "blue"
        elif -25 <= delta_idx < 0:
            phase = "Decision"
            phase_color = "orange"
        elif delta_idx == 0:
            phase = "Intent"
            phase_color = "red"
        else:
            phase = "Execution"
            phase_color = "green"

        history_line.set_data(x[:frame_idx + 1], y[:frame_idx + 1])

        if frame_idx + 1 < len(x):
            future_line.set_data(x[frame_idx + 1:], y[frame_idx + 1:])
        else:
            future_line.set_data([], [])

        ego_dot.set_offsets([[x[frame_idx], y[frame_idx]]])

        ego_delta = np.arctan2(
            window_df[Y_VELOCITY].iloc[frame_idx],
            window_df[X_VELOCITY].iloc[frame_idx]
        )
        w = window_df[WIDTH].iloc[frame_idx]
        h = window_df[HEIGHT].iloc[frame_idx]

        if window_df[X].iloc[-1] > window_df[X].iloc[0]:
            direction = 1
        else:
            direction = -1

        rect = draw_bbox(
            ax,
            x[frame_idx],
            y[frame_idx],
            w, h,
            yaw=ego_delta,
            color=phase_color,
            direction=direction
        )
        bbox_artists.append(rect)

        if delta_idx == 0:
            ax.scatter(
                x[frame_idx], y[frame_idx],
                s=140,
                facecolors='none',
                edgecolors='red',
                linewidths=2.5,
                zorder=6
            )
            ax.text(
                x[frame_idx], y[frame_idx] - 2.0,
                "Intent Onset",
                color="red",
                fontsize=10,
                ha="center",
                zorder=6
            )

        frame_number = window_df[FRAME].iloc[frame_idx]
        print_rows = []

        for col in neighbor_cols:
            ego_row = window_df.iloc[frame_idx]
            nid = int(ego_row.get(col, 0))

            if nid == 0:
                continue

            neighbor = tracks[
                (tracks[ID] == nid) &
                (tracks[FRAME] == frame_number)
                ]

            if neighbor.empty:
                continue

            delta = np.arctan2(
                neighbor[Y_VELOCITY].values[0],
                neighbor[X_VELOCITY].values[0]
            )

            color = "pink" if col == PRECEDING_ID else "yellow"

            rect = draw_bbox(
                ax,
                neighbor[X].values[0],
                neighbor[Y].values[0],
                neighbor[WIDTH].values[0],
                neighbor[HEIGHT].values[0],
                yaw=delta,
                color=color,
                direction=direction
            )
            bbox_artists.append(rect)

            print_rows.append([
                col.replace("Id", ""),
                nid,
                frame_number,
                delta
            ])

        if len(print_rows) > 0:
            print(tabulate(
                print_rows,
                headers=["Neighbor Type", "ID", "Frame", "Delta"],
                tablefmt="pretty"
            ))

        ax.set_title(
            f"Lane Change | Phase: {phase} | Frame: {frame_number}",
            fontsize=12
        )

        fig.canvas.draw_idle()
        current_frame_idx[0] = frame_idx

    ax_slider = plt.axes([0.1, 0.1, 0.8, 0.05])
    slider = Slider(ax_slider, 'Frame', 0, num_frames - 1, valinit=0, valstep=1)

    def on_slider_changed(val):
        update(int(val))
        slider.valtext.set_text(f'{int(val)}')

    slider.on_changed(on_slider_changed)

    def next_frame(event=None):
        idx = min(current_frame_idx[0] + 1, num_frames - 1)
        slider.set_val(idx)

    def prev_frame(event=None):
        idx = max(current_frame_idx[0] - 1, 0)
        slider.set_val(idx)

    ax_next = plt.axes([0.8, 0.02, 0.1, 0.04])
    btn_next = Button(ax_next, 'Next Frame')
    btn_next.on_clicked(next_frame)

    ax_prev = plt.axes([0.65, 0.02, 0.1, 0.04])
    btn_prev = Button(ax_prev, 'Prev Frame')
    btn_prev.on_clicked(prev_frame)

    def toggle_autoplay(event):
        autoplay[0] = not autoplay[0]
        btn_auto.label.set_text('Pause' if autoplay[0] else 'Autoplay')

    ax_auto = plt.axes([0.5, 0.02, 0.1, 0.04])
    btn_auto = Button(ax_auto, 'Autoplay')
    btn_auto.on_clicked(toggle_autoplay)

    def on_timer():
        if autoplay[0] and current_frame_idx[0] < num_frames - 1:
            next_frame()
        elif autoplay[0] and current_frame_idx[0] >= num_frames - 1:
            autoplay[0] = False
            btn_auto.label.set_text('Autoplay')

    timer = fig.canvas.new_timer(interval=autoplay_interval)
    timer.add_callback(on_timer)
    timer.start()

    update(start_movement)
    plt.show()


def calculate_gap_correct(ego_x, target_x, ego_width, target_width):
    gap1 = abs(ego_x - target_x)

    vehicle_length = 0
    if ego_x > target_x:
        vehicle_length = target_width
    else:
        vehicle_length = ego_width

    gap = abs(gap1 - vehicle_length)

    return gap


def get_frame_by_index(traj, i):
    frame = traj[FRAME][i]
    return frame


def get_latent_frame(traj, lc_index, vy_thresh=0.1, k=5):
    start_index = int(max(0, lc_index - (H + int(T_PRED / 2))))
    vy_windows = traj[Y_VELOCITY][start_index:lc_index]
    frame_windows = traj[FRAME][start_index:lc_index]

    for i in range(len(vy_windows) - k):
        window = vy_windows[i:i + k]
        if np.all(np.abs(window) > vy_thresh):
            return frame_windows[i]
    return None


def  detect_lane_change(traj):
    lane_ids = traj[LANE_ID]
    events = []
    for i in range(1, len(lane_ids)):
        if lane_ids[i] != lane_ids[i - 1]:
            delta = lane_ids[i] - lane_ids[i - 1]
            if delta > 0:
                lc_type = 2
            else:
                lc_type = 1
            change_lc_frame = get_frame_by_index(traj, i)
            latent_lc_frame = get_latent_frame(traj, i, vy_thresh=0.1, k=5)
            events.append({
                "latent_lc_frame": latent_lc_frame,
                "change_lc_frame": change_lc_frame,
                "lc_type": lc_type
            })
    return events


def draw_lane_lines(ax, lane_markings, x_min, x_max, color="yellow"):
    for idx, y_lane in enumerate(lane_markings):
        if idx not in [2, 3]:
            ax.plot([x_min, x_max], [y_lane, y_lane], linestyle="--", color=color, linewidth=1, alpha=0.5)
        else:
            ax.plot([x_min, x_max], [y_lane, y_lane], linestyle="-", color="black", linewidth=1, alpha=0.9)


def draw_map_background(ax, map_image_path,
                        MAP_X_MIN, MAP_X_MAX,
                        MAP_Y_MIN, MAP_Y_MAX):
    img = Image.open(map_image_path)
    extent = [
        MAP_X_MIN, MAP_X_MAX,
        MAP_Y_MIN, MAP_Y_MAX
    ]
    ax.imshow(
        img,
        extent=extent,
        origin='upper'
    )
    ax.set_aspect('equal')


def draw_bbox(ax, xc, yc, w, h, yaw,  color="red", direction=1):
    yaw = yaw + np.pi

    if direction == 1:
        cx, cy = xc, yc
        x0 = xc - w
        y0 = yc - h
    else:
        x0, y0 = xc, yc
        cx = x0 + w / 2
        cy = y0 + h / 2

    rect = patches.Rectangle((x0, y0), w, h, linewidth=1.2, edgecolor=color, facecolor="none")

    t = transforms.Affine2D().rotate_around(cx, cy, yaw) + ax.transData
    rect.set_transform(t)

    ax.set_aspect('equal')
    ax.add_patch(rect)

    return rect


def compute_ttc(d, delta_v, eps=1e-6):
    if delta_v <= eps:
        return np.inf
    if d <= 0:
        return 0.0
    return d / delta_v


def clip(x, lo, hi):
    return max(lo, min(x, hi))


def compute_time_headway(gap, v, eps=1e-3):
    if gap <= 0 or v <= eps:
        return np.inf
    return gap / v


def compute_idm_desired_gap(v, delta_v,
                            s0=2.0, T=1.5,
                            a=1.5, b=2.0):
    return s0 + v * T + (v * delta_v) / (2 * np.sqrt(a * b) + 1e-6)


def detect_latent_lc_frame(df, vy_th=0.2, win=5):
    vy = df[Y_VELOCITY].values
    frames = df[FRAME].values

    for i in range(len(vy) - win):
        window = vy[i:i+win]
        if np.mean(np.abs(window)) > vy_th:
            return frames[i]
    return None


def extract_lateral_style(track_df, lane_info=None, frame_rate=10):
    feat = {}

    vy = track_df[Y_VELOCITY].values
    vy_abs = np.abs(vy)

    feat['max_vy'] = vy_abs.max()
    feat['mean_vy'] = vy_abs.mean()
    feat['lat_jerk'] = np.std(np.diff(vy, n=2)) if len(vy) > 2 else 0.0

    if len(vy) > 1:
        vy_sign = np.sign(vy)
        vy_sign_change = np.sum(vy_sign[1:] != vy_sign[:-1])
    else:
        vy_sign_change = 0
    feat['vy_sign_change'] = vy_sign_change

    feat['intent_lead_time'] = 0.0
    feat['accepted_gap_ratio'] = np.nan
    feat['lane_change_duration'] = 0.0

    if lane_info is not None:
        lane_ids = lane_info[LANE_ID].values
        if len(np.unique(lane_ids)) > 1:
            lc_start = np.where(lane_ids != lane_ids[0])[0][0]
            lc_end = np.where(lane_ids != lane_ids[-1])[0][-1]
            feat['lane_change_duration'] = (lc_end - lc_start) / frame_rate
            feat['intent_lead_time'] = lc_start / frame_rate

            feat['accepted_gap_ratio'] = 0.5

    return feat


def extract_lateral_style_features(
    df_ego,
    change_lc_frame,
    left_gap_ratio=None,
    right_gap_ratio=None,
    fps=25
):
    latent_lc_frame = detect_latent_lc_frame(df_ego)

    if latent_lc_frame is None:
        return {
            "intent_lead_time": 0.0,
            "vy_integral_pre": 0.0,
            "vy_sign_changes": 0,
            "accepted_gap_ratio": 1.0,
            "max_vy": 0.0,
            "mean_vy": 0.0,
            "lat_jerk_norm": 0.0,
            "lc_duration": 0.0,
        }

    intent_lead_time = (change_lc_frame - latent_lc_frame) / fps
    intent_lead_time = max(intent_lead_time, 0.0)

    df_pre = df_ego[df_ego["frame"] < latent_lc_frame]
    vy_pre = df_pre["vy"].values

    vy_integral_pre = np.sum(np.abs(vy_pre)) / fps

    vy_sign_changes = np.sum(
        np.sign(vy_pre[1:]) != np.sign(vy_pre[:-1])
    )

    if left_gap_ratio is not None and right_gap_ratio is not None:
        accepted_gap_ratio = min(left_gap_ratio, right_gap_ratio)
    else:
        accepted_gap_ratio = 1.0

    df_exec = df_ego[df_ego["frame"] >= latent_lc_frame]
    vy_exec = df_exec["vy"].values

    max_vy = np.max(np.abs(vy_exec))
    mean_vy = np.mean(np.abs(vy_exec))

    ay = df_exec["ay"].values
    lat_jerk = np.diff(ay) * fps
    lat_jerk_norm = np.mean(np.abs(lat_jerk)) if len(lat_jerk) > 0 else 0.0

    lc_duration = (df_exec["frame"].iloc[-1] - latent_lc_frame) / fps

    return {
        "intent_lead_time": intent_lead_time,
        "vy_integral_pre": vy_integral_pre,
        "vy_sign_changes": vy_sign_changes,
        "accepted_gap_ratio": accepted_gap_ratio,
        "max_vy": max_vy,
        "mean_vy": mean_vy,
        "lat_jerk_norm": lat_jerk_norm,
        "lc_duration": lc_duration,
    }


def extract_feature(df: pd.DataFrame, single_traj_df: pd.DataFrame,rec_meta):
    feature_data = []

    for idx, row in single_traj_df.iterrows():
        ego_x = row[X]
        ego_y = row[Y]
        ego_vx = row[X_VELOCITY]
        ego_vy = row[Y_VELOCITY]
        ego_v = np.sqrt(ego_vx ** 2 + ego_vy ** 2)

        ego_delta = np.arctan2(row[X_VELOCITY], row[Y_VELOCITY])

        neighbor_ids = {
            "preceding": int(row[PRECEDING_ID]),
            "following": int(row[FOLLOWING_ID]),
            "leftPreceding": int(row[LEFT_PRECEDING_ID]),
            "leftAlongside": int(row.get(LEFT_ALONGSIDE_ID, 0)),
            "leftFollowing": int(row[LEFT_FOLLOWING_ID]),
            "rightPreceding": int(row[RIGHT_PRECEDING_ID]),
            "rightAlongside": int(row.get(RIGHT_ALONGSIDE_ID, 0)),
            "rightFollowing": int(row[RIGHT_FOLLOWING_ID]),
        }

        def get_vehicle_state(nid):
            if nid <= 0:
                return None

            neigh = df[(df[ID] == nid) & (df[FRAME] == row[FRAME])]
            if neigh.empty:
                return None

            n = neigh.iloc[0]
            vx = n[X_VELOCITY]
            vy = n[Y_VELOCITY]
            v = np.sqrt(vx ** 2 + vy ** 2)
            gap_x = abs(ego_x - n[X])
            gap_y = abs(ego_y - n[Y])
            delta_v = ego_v - v

            return {
                'id': nid,
                'vx': vx,
                'vy': vy,
                'v': v,
                'gap_x': gap_x,
                'gap_y': gap_y,
                'delta_v': delta_v
            }

        def get_closest_vehicle_in_zone(zone_ids):
            closest_vehicle = None
            min_gap = float('inf')

            for nid in zone_ids:
                if nid <= 0:
                    continue
                vehicle = get_vehicle_state(nid)
                if vehicle and vehicle['gap_x'] < min_gap:
                    min_gap = vehicle['gap_x']
                    closest_vehicle = vehicle

            return closest_vehicle, closest_vehicle is not None

        pre_vehicle = get_vehicle_state(neighbor_ids["preceding"])
        has_preceding = pre_vehicle is not None

        if has_preceding:
            ttc_pre = compute_ttc(pre_vehicle['gap_x'], ego_vx - pre_vehicle['vx']) if (ego_vx - pre_vehicle['vx']) > 0 else np.inf
            thw_pre = pre_vehicle['gap_x'] / ego_v if ego_v > 0 else np.inf
            pre_space = pre_vehicle['gap_x']
            pre_vx = pre_vehicle['vx']
            pre_vy = pre_vehicle['vy']
        else:
            ttc_pre = np.inf
            thw_pre = np.inf
            pre_space = np.inf
            pre_vx = np.nan
            pre_vy = np.nan

        fol_vehicle = get_vehicle_state(neighbor_ids["following"])
        has_following = fol_vehicle is not None

        if has_following:
            fol_relative_v = fol_vehicle['v'] - ego_v
            ttc_fol = compute_ttc(fol_vehicle['gap_x'], fol_relative_v) if fol_relative_v > 0 else np.inf
            thw_fol = fol_vehicle['gap_x'] / fol_vehicle['v'] if fol_vehicle['v'] > 0 else np.inf
            fol_space = fol_vehicle['gap_x']
            fol_vx = fol_vehicle['vx']
            fol_vy = fol_vehicle['vy']
        else:
            ttc_fol = np.inf
            thw_fol = np.inf
            fol_space = np.inf
            fol_vx = np.nan
            fol_vy = np.nan

        left_zone_ids = [
            neighbor_ids["leftPreceding"],
            neighbor_ids["leftAlongside"],
            neighbor_ids["leftFollowing"]
        ]

        left_vehicle, has_left_vehicle = get_closest_vehicle_in_zone(left_zone_ids)

        if has_left_vehicle:
            left_gap = left_vehicle['gap_x']
            left_lat_gap = left_vehicle['gap_y']
            left_v = left_vehicle['v']
            left_delta_v = left_vehicle['delta_v']

            lpre_vx = left_vehicle['vx'] if neighbor_ids["leftPreceding"] > 0 else np.nan
            lpre_vy = left_vehicle['vy'] if neighbor_ids["leftPreceding"] > 0 else np.nan
            lfol_vx = left_vehicle['vx'] if neighbor_ids["leftFollowing"] > 0 else np.nan
            lfol_vy = left_vehicle['vy'] if neighbor_ids["leftFollowing"] > 0 else np.nan
        else:
            left_gap = np.inf
            left_lat_gap = np.inf
            left_v = np.nan
            left_delta_v = np.nan
            lpre_vx = np.nan
            lpre_vy = np.nan
            lfol_vx = np.nan
            lfol_vy = np.nan

        right_zone_ids = [
            neighbor_ids["rightPreceding"],
            neighbor_ids["rightAlongside"],
            neighbor_ids["rightFollowing"]
        ]

        right_vehicle, has_right_vehicle = get_closest_vehicle_in_zone(right_zone_ids)

        if has_right_vehicle:
            right_gap = right_vehicle['gap_x']
            right_lat_gap = right_vehicle['gap_y']
            right_v = right_vehicle['v']
            right_delta_v = right_vehicle['delta_v']

            rpre_vx = right_vehicle['vx'] if neighbor_ids["rightPreceding"] > 0 else np.nan
            rpre_vy = right_vehicle['vy'] if neighbor_ids["rightPreceding"] > 0 else np.nan
            rfol_vx = right_vehicle['vx'] if neighbor_ids["rightFollowing"] > 0 else np.nan
            rfol_vy = right_vehicle['vy'] if neighbor_ids["rightFollowing"] > 0 else np.nan
        else:
            right_gap = np.inf
            right_lat_gap = np.inf
            right_v = np.nan
            right_delta_v = np.nan
            rpre_vx = np.nan
            rpre_vy = np.nan
            rfol_vx = np.nan
            rfol_vy = np.nan

        ego_speed = np.sqrt(ego_vx ** 2 + ego_vy ** 2)

        speed_limit = rec_meta.get("speedLimit", -1)

        if speed_limit > 0:
            v_tilde = ego_speed / speed_limit
            is_unlimited = 0
        else:
            v_tilde = ego_speed / 36.11
            is_unlimited = 1

        v_tilde = clip(v_tilde, 0.0, 2.0)

        neighbor_speeds = []

        for vx_n, vy_n in [
            (pre_vx, pre_vy),
            (fol_vx, fol_vy),
            (lpre_vx, lpre_vy),
            (lfol_vx, lfol_vy),
            (rpre_vx, rpre_vy),
            (rfol_vx, rfol_vy),
        ]:
            if not np.isnan(vx_n):
                neighbor_speeds.append(np.sqrt(vx_n ** 2 + vy_n ** 2))

        if len(neighbor_speeds) > 0:
            delta_v_flow = ego_speed - np.mean(neighbor_speeds)
        else:
            delta_v_flow = 0.0

        delta_v_flow = clip(delta_v_flow / 10.0, -2.0, 2.0)

        if neighbor_ids["preceding"] > 0 and pre_space > 0:
            thw = row[THW]
            thw = clip(thw / 2.0, 0.0, 3.0)

            delta_v = ego_speed - np.sqrt(pre_vx ** 2 + pre_vy ** 2)
            s_star = compute_idm_desired_gap(ego_speed, delta_v)
            gap_ratio = pre_space / (s_star + 1e-6)
            gap_ratio = clip(gap_ratio, 0.0, 3.0)

            inv_ttc = 1.0 / (ttc_pre + 1e-6)
            inv_ttc = clip(inv_ttc, 0.0, 1.0)
        else:
            thw = 3.0
            gap_ratio = 3.0
            inv_ttc = 0.0

        acc_long = row[X_ACCELERATION]
        acc_norm = clip(acc_long / 2.0, -2.0, 2.0)

        frame_rate = 25
        DT = 1.0 / frame_rate
        if len(feature_data) > 0:
            prev_acc = feature_data[-1]["acc_norm_raw"]
            jerk = (acc_long - prev_acc) / DT
        else:
            jerk = 0.0
        jerk_norm = clip(jerk / 1.0, -3.0, 3.0)
        brake_event = 1 if acc_long < -2.5 else 0

        lateral_feat = extract_lateral_style(single_traj_df.iloc[:idx + 1], lane_info=single_traj_df, frame_rate=frame_rate)

        row_dict = {
            "id": row[TRACK_ID],
            "frame": row[FRAME],
            "lane_id": row[LANE_ID],

            "ego_ax": row[X_ACCELERATION],
            "ego_ay": row[Y_ACCELERATION],
            "ego_vx": ego_vx,
            "ego_vy": ego_vy,
            "ego_x": ego_x,
            "ego_y": ego_y,
            "ego_delta": ego_delta,

            "has_preceding": has_preceding,
            "preceding_gap": pre_space,
            "preceding_v": pre_vehicle['v'] if has_preceding else np.nan,
            "preceding_delta_v": pre_vehicle['delta_v'] if has_preceding else np.nan,
            "ttc_preceding": ttc_pre,
            "thw_preceding": thw_pre,

            "has_following": has_following,
            "following_gap": fol_space,
            "following_v": fol_vehicle['v'] if has_following else np.nan,
            "following_delta_v": fol_vehicle['delta_v'] if has_following else np.nan,
            "ttc_following": ttc_fol,
            "thw_following": thw_fol,

            "has_left_vehicle": has_left_vehicle,
            "left_gap": left_gap,
            "left_lat_gap": left_lat_gap,
            "left_v": left_v,
            "left_delta_v": left_delta_v,

            "has_right_vehicle": has_right_vehicle,
            "right_gap": right_gap,
            "right_lat_gap": right_lat_gap,
            "right_v": right_v,
            "right_delta_v": right_delta_v,

            "max_vy": lateral_feat['max_vy'],
            "mean_vy": lateral_feat['mean_vy'],
            "lat_jerk": lateral_feat['lat_jerk'],
            "vy_sign_change": lateral_feat['vy_sign_change'],
            "intent_lead_time": lateral_feat['intent_lead_time'],
            "accepted_gap_ratio": lateral_feat['accepted_gap_ratio'],
            "lane_change_duration": lateral_feat['lane_change_duration'],

            "v_tilde": v_tilde,
            "is_unlimited": is_unlimited,
            "delta_v_flow": delta_v_flow,
            "thw": thw,
            "gap_ratio": gap_ratio,
            "inv_ttc": inv_ttc,
            "acc_norm": acc_norm,
            "jerk_norm": jerk_norm,
            "brake_event": brake_event,

            "acc_norm_raw": acc_long,
        }

        feature_data.append(row_dict)

    return pd.DataFrame(feature_data)


def parse_lane_markings(rec_meta):
    lane_markings = []

    def _parse(v):
        if v is None:
            return []

        if isinstance(v, (list, tuple, np.ndarray)):
            return [float(x) for x in v]

        if isinstance(v, str):
            v = v.strip()
            if v == "":
                return []
            return [float(x) for x in v.split(";") if x.strip() != ""]

        if isinstance(v, (int, float)):
            return [float(v)]

        return []

    lane_markings += _parse(rec_meta.get("upperLaneMarkings"))
    lane_markings += _parse(rec_meta.get("lowerLaneMarkings"))

    lane_markings = sorted(lane_markings)

    return lane_markings


def traj_to_df(traj):
    n = len(traj[FRAME])
    traj[ID] = np.repeat(traj[ID], n)

    bbox = np.asarray(traj[BBOX])
    traj[X] = bbox[:, 0]
    traj[Y] = bbox[:, 1]
    traj[WIDTH] = bbox[:, 2]
    traj[HEIGHT] = bbox[:, 3]
    del traj[BBOX]

    return pd.DataFrame(traj)


def single_scence_handler(RECORDING_ID):
    tracks_path = os.path.join(DATA_DIR, f"{RECORDING_ID:02d}_tracks.csv")
    rec_meta_path = os.path.join(DATA_DIR, f"{RECORDING_ID:02d}_recordingMeta.csv")
    static_path = os.path.join(DATA_DIR, f"{RECORDING_ID:02d}_tracksMeta.csv")
    backimg_path = os.path.join(DATA_DIR, f"{RECORDING_ID:02d}_highway.png")

    arguments = {
        "input_path": tracks_path,
        "input_static_path": static_path,
        "input_meta_path": rec_meta_path,
        "background_image": backimg_path
    }

    tracks, tracks_df = read_track_csv(arguments)
    rec_meta = read_meta_info(arguments)

    all_vehicle_feature = []

    for traj in tracks:
        traj_df = traj_to_df(traj)

        all_vehicle_feature1 = extract_feature(tracks_df, traj_df, rec_meta)
        all_vehicle_feature.append(all_vehicle_feature1)

    if len(all_vehicle_feature) > 0:
        feature_df = pd.concat(all_vehicle_feature, ignore_index=True)
        out_path = f"{SAVE_OUTPUT_DIR}/{RECORDING_ID:02d}_tracks_extract_feature.csv"
        feature_df.to_csv(out_path, index=False)
        print(f"Features saved: {out_path}")
    else:
        print(f"Scene {RECORDING_ID}: No valid trajectory windows detected")


def process_recording(recording_id):
    print(f"Starting recording {recording_id:02d}...", flush=True)
    try:
        single_scence_handler(recording_id)
        print(f"Recording {recording_id:02d} completed", flush=True)
        return True
    except Exception as e:
        print(f"Recording {recording_id:02d} failed: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return False

def get_optimal_workers_adaptive():
    cpu_count = multiprocessing.cpu_count()

    base_workers = {
        'low': 2,
        'medium': 4,
        'high': 8,
        'extreme': 16
    }

    if cpu_count <= 2:
        workers = base_workers['low']
    elif cpu_count <= 4:
        workers = base_workers['medium']
    elif cpu_count <= 8:
        workers = base_workers['high']
    else:
        workers = base_workers['extreme']

    try:
        with open('/proc/1/cgroup', 'r') as f:
            if 'docker' in f.read() or 'kubepods' in f.read():
                workers = min(workers, 4)
    except:
        pass

    try:
        import psutil
        load_avg = psutil.getloadavg()[0]
        if load_avg > cpu_count * 0.8:
            workers = max(workers // 2, 2)
    except:
        pass

    return workers


if __name__ == "__main__":
    DATA_DIR = r"./data"
    OUTPUT_DIR = "./lane_change_visualization_traj"
    SAVE_OUTPUT_DIR = "allFeature_traj_fixed_features"

    FRAME_RATE = 25
    H = FRAME_RATE * 3
    T_PRED = FRAME_RATE * 5

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(SAVE_OUTPUT_DIR, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Configuration:")
    print(f"  History: {H} frames ({H / FRAME_RATE:.1f}s)")
    print(f"  Future: {T_PRED} frames ({T_PRED / FRAME_RATE:.1f}s)")
    print(f"  Total window: {H + T_PRED + 1} frames")
    print(f"{'=' * 60}\n")

    RECORDING_IDS = range(41, 61)
    MAX_WORKERS = get_optimal_workers_adaptive()

    print(f"Processing {len(RECORDING_IDS)} recordings with {MAX_WORKERS} workers\n")

    from concurrent.futures import ThreadPoolExecutor, as_completed
    import time

    start_time = time.time()
    completed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_recording, rid): rid for rid in RECORDING_IDS}

        for future in as_completed(futures):
            rid = futures[future]
            completed += 1
            try:
                future.result()
                print(f"[{completed}/{len(RECORDING_IDS)}] Recording {rid:02d} completed")
            except Exception as e:
                print(f"[{completed}/{len(RECORDING_IDS)}] Recording {rid:02d} failed: {e}")

    total_time = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"Total time: {total_time:.1f}s ({total_time / 60:.1f} minutes)")
    print(f"Average: {total_time / len(RECORDING_IDS):.1f}s per recording")
    print(f"{'=' * 60}")