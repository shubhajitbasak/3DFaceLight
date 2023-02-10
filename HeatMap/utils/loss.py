import torch


def loss_heatmap(gt, pre):
    """
    https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhang_Fast_Human_Pose_Estimation_CVPR_2019_paper.pdf
    \gt BxCx64x64
    \pre BxCx64x64
    """
    # nn.MSELoss()
    B, C, H, W = gt.shape
    gt = gt.view(B, C, -1)
    pre = pre.view(B, C, -1)
    loss = torch.sum((pre - gt) * (pre - gt), axis=-1)  # Sum square error in each heatmap
    loss = torch.mean(loss, axis=-1)  # MSE in 1 sample / batch over all heatmaps
    loss = torch.mean(loss, axis=-1)  # Avarage MSE in 1 batch (.i.e many sample)
    return loss


def adaptive_wing_loss(y_pred, y_true, w=14, epsilon=1.0, theta=0.5, alpha=2.1):
    """
    \ref https://arxiv.org/pdf/1904.07399.pdf
    """
    # Calculate A and C
    p1 = (1 / (1 + (theta / epsilon) ** (alpha - y_true)))
    p2 = (alpha - y_true) * ((theta / epsilon) ** (alpha - y_true - 1)) * (1 / epsilon)
    A = w * p1 * p2
    C = theta * A - w * torch.log(1 + (theta / epsilon) ** (alpha - y_true))

    # Asolute value
    absolute_x = torch.abs(y_true - y_pred)

    # Adaptive wingloss
    losses = torch.where(theta > absolute_x, w * torch.log(1.0 + (absolute_x / epsilon) ** (alpha - y_true)),
                         A * absolute_x - C)
    losses = torch.sum(losses, axis=[2, 3])
    losses = torch.mean(losses)

    return losses  # Mean wingloss for each sample in batch