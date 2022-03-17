import os
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage.filters import convolve


def Object(pred, gt):
    x = np.mean(pred[gt == 1])
    sigma_x = np.std(pred[gt == 1])
    score = 2.0 * x / (x ** 2 + 1 + sigma_x + np.finfo(np.float64).eps)

    return score

def S_Object(pred, gt):
    pred_fg = pred.copy()
    pred_fg[gt != 1] = 0.0
    O_fg = Object(pred_fg, gt)
    
    pred_bg = (1 - pred.copy())
    pred_bg[gt == 1] = 0.0
    O_bg = Object(pred_bg, 1-gt)

    u = np.mean(gt)
    Q = u * O_fg + (1 - u) * O_bg

    return Q

def centroid(gt):
    if np.sum(gt) == 0:
        return gt.shape[0] // 2, gt.shape[1] // 2
    
    else:
        x, y = np.where(gt == 1)
        return int(np.mean(x).round()), int(np.mean(y).round())

def divide(gt, x, y):
    LT = gt[:x, :y]
    RT = gt[x:, :y]
    LB = gt[:x, y:]
    RB = gt[x:, y:]

    w1 = LT.size / gt.size
    w2 = RT.size / gt.size
    w3 = LB.size / gt.size
    w4 = RB.size / gt.size

    return LT, RT, LB, RB, w1, w2, w3, w4

def ssim(pred, gt):
    x = np.mean(pred)
    y = np.mean(gt)
    N = pred.size

    sigma_x2 = np.sum((pred - x) ** 2 / (N - 1 + np.finfo(np.float64).eps))
    sigma_y2 = np.sum((gt - y) ** 2 / (N - 1 + np.finfo(np.float64).eps))

    sigma_xy = np.sum((pred - x) * (gt - y) / (N - 1 + np.finfo(np.float64).eps))

    alpha = 4 * x * y * sigma_xy
    beta = (x ** 2 + y ** 2) * (sigma_x2 + sigma_y2)

    if alpha != 0:
        Q = alpha / (beta + np.finfo(np.float64).eps)
    elif alpha == 0 and beta == 0:
        Q = 1
    else:
        Q = 0
    
    return Q

def S_Region(pred, gt):
    x, y = centroid(gt)
    gt1, gt2, gt3, gt4, w1, w2, w3, w4 = divide(gt, x, y)
    pred1, pred2, pred3, pred4, _, _, _, _ = divide(pred, x, y)

    Q1 = ssim(pred1, gt1)
    Q2 = ssim(pred2, gt2)
    Q3 = ssim(pred3, gt3)
    Q4 = ssim(pred4, gt4)

    Q = Q1 * w1 + Q2 * w2 + Q3 * w3 + Q4 * w4

    return Q

def StructureMeasure(pred, gt):
    y = np.mean(gt)

    if y == 0:
        x = np.mean(pred)
        Q = 1 - x
    elif y == 1:
        x = np.mean(pred)
        Q = x
    else:
        alpha = 0.5
        Q = alpha * S_Object(pred, gt) + (1 - alpha) * S_Region(pred, gt)
        if Q < 0:
            Q = 0
    
    return Q

def fspecial_gauss(size, sigma):
       """Function to mimic the 'fspecial' gaussian MATLAB function
       """
       x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
       g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
       return g/g.sum()

def original_WFb(pred, gt):
    E = np.abs(pred - gt)
    dst, idst = distance_transform_edt(1 - gt, return_indices=True)

    K = fspecial_gauss(7, 5)
    Et = E.copy()
    Et[gt != 1] = Et[idst[:, gt != 1][0], idst[:, gt != 1][1]]
    EA = convolve(Et, K, mode='nearest')
    MIN_E_EA = E.copy()
    MIN_E_EA[(gt == 1) & (EA < E)] = EA[(gt == 1) & (EA < E)]

    B = np.ones_like(gt)
    B[gt != 1] = 2.0 - 1 * np.exp(np.log(1 - 0.5) / 5 * dst[gt != 1])
    Ew = MIN_E_EA * B

    TPw = np.sum(gt) - np.sum(Ew[gt == 1])
    FPw = np.sum(Ew[gt != 1])

    R = 1 - np.mean(Ew[gt == 1])
    P = TPw / (TPw + FPw + np.finfo(np.float64).eps)
    Q = 2 * R * P / (R + P + np.finfo(np.float64).eps)

    return Q

def Fmeasure_calu(pred, gt, threshold):
    if threshold > 1:
        threshold = 1

    Label3 = np.zeros_like(gt)
    Label3[pred >= threshold] = 1

    NumRec = np.sum(Label3 == 1)
    NumNoRec = np.sum(Label3 == 0)

    LabelAnd = (Label3 == 1) & (gt == 1)
    NumAnd = np.sum(LabelAnd == 1)
    num_obj = np.sum(gt)
    num_pred = np.sum(Label3)

    FN = num_obj - NumAnd
    FP = NumRec - NumAnd
    TN = NumNoRec - FN

    if NumAnd == 0:
        PreFtem = 0
        RecallFtem = 0
        FmeasureF = 0
        Dice = 0
        SpecifTem = 0
        IoU = 0

    else:
        IoU = NumAnd / (FN + NumRec)
        PreFtem = NumAnd / NumRec
        RecallFtem = NumAnd / num_obj
        SpecifTem = TN / (TN + FP)
        Dice = 2 * NumAnd / (num_obj + num_pred)
        FmeasureF = ((2.0 * PreFtem * RecallFtem) / (PreFtem + RecallFtem))
    
    return PreFtem, RecallFtem, SpecifTem, Dice, FmeasureF, IoU

   
def Fmeasure_calu2(pred, gt, threshold):
    if threshold > 1:
        threshold = 1

    Label3 = np.zeros_like(gt)
    Label3[pred >= threshold] = 1
    
    LabelAnd = (Label3 == 1) & (gt == 1)
    NumRec = np.sum(Label3 == 1)
    NumAnd = np.sum(LabelAnd == 1)
    num_obj = np.sum(gt)
    num_pred = np.sum(Label3)

    FN = num_obj - NumAnd

    if NumAnd == 0:
        PreFtem = 0
        RecallFtem = 0
        FmeasureF = 0
        Dice = 0
        SpecifTem = 0
        IoU = 0

    else:
        IoU = NumAnd / (FN + NumRec)
        Dice = 2 * NumAnd / (num_obj + num_pred)
    
    return Dice, IoU

def AlignmentTerm(pred, gt):
    mu_pred = np.mean(pred)
    mu_gt = np.mean(gt)

    align_pred = pred - mu_pred
    align_gt = gt - mu_gt

    align_mat = 2 * (align_gt * align_pred) / (align_gt ** 2 + align_pred ** 2 + np.finfo(np.float64).eps)
    
    return align_mat

def EnhancedAlighmentTerm(align_mat):
    enhanced = ((align_mat + 1) ** 2) / 4
    return enhanced

def EnhancedMeasure(pred, gt):
    if np.sum(gt) == 0:
        enhanced_mat = 1 - pred
    elif np.sum(1 - gt) == 0:
        enhanced_mat = pred.copy()
    else:
        align_mat = AlignmentTerm(pred, gt)
        enhanced_mat = EnhancedAlighmentTerm(align_mat)
    
    score = np.sum(enhanced_mat) / (gt.size - 1 + np.finfo(np.float64).eps)
    return score
  


def evaluate(dataset, exp_path, gt_path, pred_path, exp_name, light=True):
  
  Thresholds = np.linspace(1, 0, 255)
  preds = os.listdir(pred_path)
  threshold_Emeasure = np.zeros((len(preds), len(Thresholds)))
  threshold_IoU = np.zeros((len(preds), len(Thresholds)))
  threshold_Dice = np.zeros((len(preds), len(Thresholds)))
  Smeasure = np.zeros(len(preds))
  wFmeasure = np.zeros(len(preds))
  MAE = np.zeros(len(preds))
  
  names = os.listdir(pred_path)
  
  for i, name in enumerate(names):

    pred = Image.open(os.path.join(pred_path, name))
    gt = Image.open(os.path.join(gt_path,name))

    if light:
      pred = pred.resize((352,352))
      gt = gt.resize((352,352))

    pred = np.array(pred)
    gt = np.array(gt)
    
    if len(pred.shape) != 2:
      pred = pred[:, :, 0]
    if len(gt.shape) != 2:
      gt = gt[:, :, 0]
    
    assert pred.shape == gt.shape
    
    gt_mask = gt.astype(np.float64) / 255
    gt_mask = (gt_mask > 0.5).astype(np.float64)
    
    pred_mask = pred.astype(np.float64) / 255
    
    Smeasure[i] = StructureMeasure(pred_mask, gt_mask)
    wFmeasure[i] = original_WFb(pred_mask, gt_mask)
    MAE[i] = np.mean(np.abs(gt_mask - pred_mask))
    
    threshold_E = np.zeros(len(Thresholds))
    threshold_Iou = np.zeros(len(Thresholds))
    threshold_Dic = np.zeros(len(Thresholds))
    
    for j, threshold in enumerate(Thresholds):
      threshold_Dic[j], threshold_Iou[j] = Fmeasure_calu2(pred_mask, gt_mask, threshold)
      Bi_pred = np.zeros_like(pred_mask)
      Bi_pred[pred_mask >= threshold] = 1
      threshold_E[j] = EnhancedMeasure(Bi_pred, gt_mask)
      
      threshold_Emeasure[i, :] = threshold_E
      threshold_Dice[i, :] = threshold_Dic
      threshold_IoU[i, :] = threshold_Iou
    
  mae = np.mean(MAE)
  Sm = np.mean(Smeasure)
  wFm = np.mean(wFmeasure)

  column_E = np.mean(threshold_Emeasure, axis=0)
  meanEm = np.mean(column_E)
  maxEm = np.max(column_E)

  column_Dic = np.mean(threshold_Dice, axis=0)
  meanDic = np.mean(column_Dic)
  maxDic = np.max(column_Dic)

  column_IoU = np.mean(threshold_IoU, axis=0)
  meanIoU = np.mean(column_IoU)
  maxIoU = np.max(column_IoU)

  header = ['exp', 'dataset', 'mDic', 'mIoU', 'wFm', 'Sm', 'mEm', 'maxEm','MAE']
  values = [exp_name,
            dataset,
            np.round(meanDic,4),
            np.round(meanIoU,4),
            np.round(wFm,4),
            np.round(Sm,4),
            np.round(meanEm,4),
            np.round(maxEm,4),
            np.round(mae,4)]
  
  csv = os.path.join(exp_path, 'results.csv')
  if os.path.exists(csv):
    dataf = pd.DataFrame([values])
    mode = 'a'
  else:
    dataf = pd.DataFrame([header,values])
    mode = 'w'
  dataf.to_csv(csv, mode=mode, index=False, header=False)

  for h,v in zip(header,values):
    print(f'{h}:{v} ', end='')
  print()

