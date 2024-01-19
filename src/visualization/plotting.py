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
