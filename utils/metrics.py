import torch
import numpy as np

def dice_coefficient(pred, target, num_classes, epsilon=1e-6):
    """
    Dice coefficient for multi-class segmentation.

    Args:
        pred (Tensor): (B, D, H, W) - 예측된 클래스 index (argmax 후)
        target (Tensor): (B, D, H, W) - ground truth label
        num_classes (int): 클래스 수
        epsilon (float): 0으로 나누는 상황 방지용 작은 수

    Returns:
        dice_scores (list of float): 클래스별 Dice score
        mean_dice (float): 전체 평균 Dice score
    """
    dice_scores = []

    for cls in range(num_classes):
        pred_inds = (pred == cls).float()
        target_inds = (target == cls).float()

        intersection = (pred_inds * target_inds).sum()
        union = pred_inds.sum() + target_inds.sum()

        dice = (2. * intersection + epsilon) / (union + epsilon)
        dice_scores.append(dice.item())

    return dice_scores, sum(dice_scores) / len(dice_scores)


def iou_score(pred, target, num_classes, epsilon=1e-6):
    """
    IoU (Jaccard Index) 계산 함수

    Args:
        pred, target, num_classes: dice_coefficient와 동일

    Returns:
        iou_scores (list of float), mean_iou (float)
    """
    iou_scores = []

    for cls in range(num_classes):
        pred_inds = (pred == cls).float()
        target_inds = (target == cls).float()

        intersection = (pred_inds * target_inds).sum()
        union = pred_inds.sum() + target_inds.sum() - intersection

        iou = (intersection + epsilon) / (union + epsilon)
        iou_scores.append(iou.item())

    return iou_scores, sum(iou_scores) / len(iou_scores)
