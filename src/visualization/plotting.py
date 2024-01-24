import math
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def imshow_cube(x):
    cmap = ListedColormap(["red", "green", "yellow", "blue", "beige", "orange"])
    face_size = len(x) // 6
    dim = int(face_size**0.5)

    plt.figure(figsize=(5, 3))

    for k in range(6):
        face = x[k * face_size : (k + 1) * face_size]
        plt.subplot(2, 3, k + 1)
        plt.imshow(
            face.reshape(dim, dim),
            vmin=0,
            vmax=1,
            cmap=cmap,
        )
        plt.axis("off")
    plt.tight_layout()


def plot_history(history, episode_length=20, density=500, save_to_folder=None):
    gap = math.ceil(len(history["loss"]) / density)
    x = [
        n * gap * episode_length / 1000
        for n in list(range(len(history["loss"][::gap])))
    ]
    plt.figure(figsize=(12, 10))
    plt.subplot(2, 2, 1)
    plt.title("Loss")
    plt.plot(x, history["loss"][::gap])
    plt.xlabel("steps (thousands)")

    plt.subplot(2, 2, 2)
    plt.title("Average success")
    plt.plot(x, history["avg_success"][::gap])
    plt.ylim(0, 1)
    plt.xlabel("steps (thousands)")

    plt.subplot(2, 2, 3)
    plt.title("Average cumulative reward")
    plt.plot(x, history["reward"][::gap])
    plt.xlabel("steps (thousands)")

    plt.subplot(2, 2, 4)
    for n, success in history["success_per_n"].items():
        plt.plot(x, success[::gap], label=f"n={n}")
    plt.title("success rate for cube scrambled with n moves")
    plt.legend()
    plt.ylim(0, 1)
    plt.xlabel("steps (thousands)")

    if save_to_folder:
        plt.savefig(f"{save_to_folder}/history.png", dpi=300)


def plot_success_length(success_length_per_n, save_to_folder=None):
    window = 1000
    # to_plot = [h[-window:] for h in history['success_length_per_n'].values()]
    to_plot = []
    for h in success_length_per_n.values():
        arr = [n for n in h[-window:] if n != 0]
        if len(arr) == 0:
            arr = [0]
        to_plot.append(arr)
    plt.violinplot(to_plot, showmeans=True)
    plt.plot(list(range(len(success_length_per_n) + 1)), "--")
    plt.ylabel("moves actually took")
    plt.xlabel("number of shuffles")

    if save_to_folder:
        plt.savefig(f"{save_to_folder}/success_length.png", dpi=300)


def plot_success_rate(success_rate_per_n, save_to_folder=None):
    window = 1000
    # to_plot = [h[-window:] for h in history['success_length_per_n'].values()]
    to_plot = [[n for n in h[-window:]] for h in success_rate_per_n.values()]
    plt.violinplot(to_plot)
    plt.ylabel("Success Rate")
    plt.xlabel("number of shuffles")
    plt.ylim(0, 1)
    if save_to_folder:
        plt.savefig(f"{save_to_folder}/success_rate.png", dpi=300)
