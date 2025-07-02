import numpy as np
import pymde
import matplotlib.pyplot as plt
import pickle

pymde.seed(42069)


def _do_pymde(fname):
    vis = []
    phase = []

    with open(fname, "rb") as f:
        info = pickle.load(f)
    cims = info["corrected_ims"]
    mask = info["mask"]
    phase_values = info["fourier"]["phase"]["central"]

    print(cims.shape)

    measured_phase = []
    for im in cims:
        im -= np.min(im)
        im /= np.max(im)
        fft_raw = np.fft.fft2(im)
        ft = np.fft.fftshift(fft_raw)
        amp = np.abs(ft)
        amp /= np.abs(fft_raw[0, 0])
        ft /= amp
        start = 20
        distance = 2

        # now to extract the values properly
        xc = 3
        yc = im.shape[0] // 2 + 1
        locs = {
            "center": (xc, yc),
        }

        p = np.angle(ft)[locs["center"][1], locs["center"][0]]
        v = np.abs(ft)[locs["center"][1], locs["center"][0]]
        vis.append(im.flatten())
        phase.append(np.angle(ft).flatten())
        measured_phase.append(p)
    vis = np.array(vis)
    phase = np.array(phase)
    embedding = pymde.preserve_neighbors(vis, embedding_dim=2, verbose=True).embed()
    pymde.plot(embedding, color_by=phase_values, color_map="magma", marker_size=4)

    indices = np.array([idx for idx in range(len(embedding)) if embedding[idx][0] > 1])
    indices3 = np.array(
        [idx for idx in range(len(embedding)) if -1 < embedding[idx][0] < 1]
    )
    indices4 = np.array(
        [idx for idx in range(len(embedding)) if -2 < embedding[idx][0] < 0]
    )
    indices2 = np.array(
        [idx for idx in range(len(embedding)) if embedding[idx][0] < -2]
    )

    fig, ((ax, bx), (cx, dx)) = plt.subplots(2, 2)
    ax.imshow(np.mean(cims[indices], 0), origin="lower")
    bx.imshow(np.mean(cims[indices2], 0), origin="lower")
    cx.imshow(np.mean(cims[indices3], 0), origin="lower")
    dx.imshow(np.mean(cims[indices4], 0), origin="lower")
    plt.show()


if __name__ == "__main__":
    fname = "/Users/jwisbell/Documents/lbti/ngc4151/fizeau_final/ngc4151_8p7_phasesel/intermediate/frame_selection/NGC4151_fs_info_cycle1.pk"
    _do_pymde(fname)
