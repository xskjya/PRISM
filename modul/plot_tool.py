from matplotlib import pyplot as plt


def plot_forgetting_curve(validation_history, save_dir, init_idx):
    plt.figure(figsize=(12, 6))

    epochs = [h['epoch'] for h in validation_history]
    ades = [h['val_ade'] for h in validation_history]
    fdes = [h.get('val_fde', 0) for h in validation_history]

    plt.subplot(1, 2, 1)
    plt.plot(epochs, ades, 'b-o', linewidth=2, markersize=8, label='ADE')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('ADE (m)', fontsize=12)
    plt.title('Validation ADE Curve', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, fdes, 'r-o', linewidth=2, markersize=8, label='FDE')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('FDE (m)', fontsize=12)
    plt.title('Validation FDE Curve', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'{save_dir}/validation_curve_init_{init_idx}.png', dpi=150)
    plt.close()