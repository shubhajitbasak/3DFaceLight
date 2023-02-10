import torch
import numpy as np
from torch.nn import functional as F


def generate_gaussian(t, x, y, z, sigma=10):
    """
    Generates a 3D Gaussian point at location x,y,z in tensor t.

    x should be in range (-1, 1) to match the output of fastai's PointScaler.

    sigma is the standard deviation of the generated 2D Gaussian.
    """
    _gaussians = {}

    h, w, d = t.shape  # height, width, depth

    # Heatmap pixel per output pixel
    mu_x = int(0.5 * (x + 1.) * w)
    mu_y = int(0.5 * (y + 1.) * h)
    mu_z = int(0.5 * (z + 1.) * d)

    tmp_size = sigma * 3

    # Top-left
    x1, y1, z1 = int(mu_x - tmp_size), int(mu_y - tmp_size), int(mu_z - tmp_size)

    # Bottom right
    x2, y2, z2 = int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1), int(mu_z + tmp_size + 1)
    if x1 >= w or y1 >= h or z1 >= d or x2 < 0 or y2 < 0 or z2 < 0:
        return t

    size = 2 * tmp_size + 1
    tx = np.arange(0, size, 1, np.float32)
    ty = tx[:, np.newaxis]
    tz = ty[:, np.newaxis]
    x0 = y0 = z0 = size // 2

    # The gaussian is not normalized, we want the center value to equal 1
    g = _gaussians[sigma] if sigma in _gaussians \
        else torch.Tensor(np.exp(- ((tx - x0) ** 2 + (ty - y0) ** 2 + (tz - z0) ** 2) / (2 * sigma ** 2)))
    _gaussians[sigma] = g

    # Determine the bounds of the source gaussian
    g_x_min, g_x_max = max(0, -x1), min(x2, w) - x1
    g_y_min, g_y_max = max(0, -y1), min(y2, h) - y1
    g_z_min, g_z_max = max(0, -z1), min(z2, d) - z1

    # Image range
    img_x_min, img_x_max = max(0, x1), min(x2, w)
    img_y_min, img_y_max = max(0, y1), min(y2, h)
    img_z_min, img_z_max = max(0, z1), min(z2, d)

    t[img_y_min:img_y_max, img_x_min:img_x_max, img_z_min:img_z_max] = \
        g[g_y_min:g_y_max, g_x_min:g_x_max, g_z_min:g_z_max]

    return t


def coord2heatmap(w, h, ow, oh, oz, x, y, z, random_round=False, random_round_with_gaussian=False):
    """
    Inserts a coordinate (x,y,z) from a picture with
    original size (w x h) into a heatmap, by randomly assigning
    it to one of its nearest neighbor coordinates, with a probability
    proportional to the coordinate error.

    Arguments:
    x: x coordinate
    y: y coordinate
    w: original width of picture with x coordinate
    h: original height of picture with y coordinate
    """
    # Get scale
    sx = ow / w
    sy = oh / h
    sz = sy  # sbasak experiment

    # Unrounded target points
    px = x * sx
    py = y * sy
    pz = z * sz

    # Truncated coordinates
    nx, ny, nz = int(px), int(py), int(pz)

    # Coordinate error
    ex, ey, ez = px - nx, py - ny, pz - nz

    # Heatmap
    heatmap = torch.zeros(ow, oh, oz)

    if random_round_with_gaussian:
        xyzr = torch.rand(3)
        xx = (ex >= xyzr[0]).long()
        yy = (ey >= xyzr[1]).long()
        zz = (ez >= xyzr[2]).long()
        row = min(ny + yy, heatmap.shape[0] - 1)
        col = min(nx + xx, heatmap.shape[1] - 1)
        depth = min(nz + zz, heatmap.shape[2] - 1)

        # Normalize into - 1, 2
        col = (col / float(ow)) * (2) + (-1)
        row = (row / float(oh)) * (2) + (-1)
        depth = (depth / float(oz)) * (2) + (-1)
        heatmap = generate_gaussian(heatmap, col, row, depth, sigma=1.5)


    elif random_round:
        xyr = torch.rand(2)
        xx = (ex >= xyr[0]).long()
        yy = (ey >= xyr[1]).long()
        heatmap[min(ny + yy, heatmap.shape[0] - 1),
        min(nx + xx, heatmap.shape[1] - 1)] = 1
    else:
        nx = min(nx, ow - 1)
        ny = min(ny, oh - 1)
        heatmap[ny][nx] = (1 - ex) * (1 - ey)
        if (ny + 1 < oh - 1):
            heatmap[ny + 1][nx] = (1 - ex) * ey

        if (nx + 1 < ow - 1):
            heatmap[ny][nx + 1] = ex * (1 - ey)

        if (nx + 1 < ow - 1 and ny + 1 < oh - 1):
            heatmap[ny + 1][nx + 1] = ex * ey

    return heatmap


def lmks2heatmap3d(lmks, heatcube_dim=16, random_round=False, random_round_with_gaussian=False):
    w, h, ow, oh, oz = 256, 256, heatcube_dim, heatcube_dim, heatcube_dim
    heatmap = torch.rand((lmks.shape[0], lmks.shape[1], ow, oh, oz))
    for i in range(lmks.shape[0]):  # num_lmks
        for j in range(lmks.shape[1]):
            heatmap[i][j] = coord2heatmap(w, h, ow, oh, oz, lmks[i][j][0], lmks[i][j][1], lmks[i][j][2],
                                          random_round=random_round,
                                          random_round_with_gaussian=random_round_with_gaussian)

    return heatmap


def cross_loss_entropy_heatmap(p, g, pos_weight=torch.Tensor([1])):
    """\ Bx 106x 256x256
    """
    BinaryCrossEntropyLoss = torch.nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pos_weight)

    B, C, W, H = p.shape

    loss = BinaryCrossEntropyLoss(p, g)

    return loss


def heatmap2topkheatmap(heatmap, topk=7):
    """
    \ Find topk value in each heatmap and calculate softmax for them.
    \ Another non topk points will be zero.
    \Based on that https://discuss.pytorch.org/t/how-to-keep-only-top-k-percent-values/83706
    """
    N, C, H, W = heatmap.shape

    # Get topk points in each heatmap
    # And using softmax for those score
    heatmap = heatmap.view(N, C, 1, -1)

    score, index = heatmap.topk(topk, dim=-1)
    score = F.softmax(score, dim=-1)
    heatmap = F.softmax(heatmap, dim=-1)

    # Assign non-topk zero values
    # Assign topk with calculated softmax value
    res = torch.zeros(heatmap.shape)
    res = res.scatter(-1, index, score)

    # Reshape to the original size
    heatmap = res.view(N, C, H, W)
    # heatmap = heatmap.view(N, C, H, W)

    return heatmap


def mean_topk_activation(heatmap, topk=7):
    """
    \ Find topk value in each heatmap and calculate softmax for them.
    \ Another non topk points will be zero.
    \Based on that https://discuss.pytorch.org/t/how-to-keep-only-top-k-percent-values/83706
    """
    N, C, H, W = heatmap.shape

    # Get topk points in each heatmap
    # And using softmax for those score
    heatmap = heatmap.view(N, C, 1, -1)

    score, index = heatmap.topk(topk, dim=-1)
    score = torch.sigmoid(score)

    return score


def heatmap2softmaxheatmap(heatmap):
    N, C, H, W = heatmap.shape

    # Get topk points in each heatmap
    # And using softmax for those score
    heatmap = heatmap.view(N, C, 1, -1)
    heatmap = F.softmax(heatmap, dim=-1)

    # Reshape to the original size
    heatmap = heatmap.view(N, C, H, W)

    return heatmap


def heatmap2sigmoidheatmap(heatmap):
    heatmap = torch.sigmoid(heatmap)

    return heatmap


def soft_argmax_3d(heatmap3d):
    batch_size = heatmap3d.shape[0]
    depth, height, width = heatmap3d.shape[2:]
    heatmap3d = heatmap3d.reshape((batch_size, -1, depth * height * width))
    heatmap3d = F.softmax(heatmap3d, 2)
    heatmap3d = heatmap3d.reshape((batch_size, -1, depth, height, width))

    accu_x = heatmap3d.sum(dim=(2, 3))
    accu_y = heatmap3d.sum(dim=(2, 4))
    accu_z = heatmap3d.sum(dim=(3, 4))
    device = heatmap3d.device

    accu_x = accu_x * torch.arange(width).float().to(device)[None, None, :]
    accu_y = accu_y * torch.arange(height).float().to(device)[None, None, :]
    accu_z = accu_z * torch.arange(depth).float().to(device)[None, None, :]

    accu_x = accu_x.sum(dim=2, keepdim=True)
    accu_y = accu_y.sum(dim=2, keepdim=True)
    accu_z = accu_z.sum(dim=2, keepdim=True)

    coord_out = torch.cat((accu_x, accu_y, accu_z), dim=2)
    return coord_out


if __name__ == '__main__':
    lmks = np.random.rand(16, 520, 3)
    heatmap = lmks2heatmap3d(lmks, 16, True, True)
    print(heatmap.shape)
