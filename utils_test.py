
from __future__ import division
from __future__ import print_function

import os
import shutil
import numpy as np
import matplotlib
matplotlib.use('AGG')    # Show Plot Disabled
import matplotlib.pyplot as plt
import random
import string
import pdb
from scipy.spatial import distance

'''Usage
def get_overlap_f1(self, overlap, bg_class):
    return utils.get_overlap_f1_colin(self.result, self.label,
                                        n_classes=self.class_num,
                                        overlap=overlap,
                                        bg_class=bg_class)

f_scores = []
for overlap in [0.1, 0.25, 0.5, 0.75]:
    f_scores.append(utils.get_overlap_f1_colin(preditions, gts,
                                    n_classes=gesture_class_num, 
                                    bg_class=bg_class, 
                                    overlap=overlap))
'''

def main():
    gt = [[0,5,1,4,3],[0,5,1,4,3],[0,5,1,4,3],[0,5,1,4,3],[0,5,1,4,3]]
    pred = [[0,5,1,2,3],[0,5,1,2,3],[0,5,1,2,3],[0,5,1,2,3],[0,5,1,2,3]]

    p1 = [[0,0,0,1,1,1,1,1],[0,0,0,1,1,1,1,1],[0,0,0,1,1,1,1,1]]
    p2 = [[0,0,0,1,1,1,1,1],[0,0,0,1,1,1,1,1],[0,0,0,1,1,1,1,1]]

    p3 = [[[1,1,1,4],[1,1,1,4],[1,2,3,4],[1,2,3,4]],[[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4]],[[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4]]]
    p4 = [[[1,2,3,4],[1,1,1,4],[1,2,3,4],[1,2,3,4]],[[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4]],[[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4]]]

    p5 = [0,1,0,0,1,1,1,1,1,1,0]
    p6 = [0,2,0,0,1,1,1,1,1,1,0]
    res = distance.jaccard(p5,p6)

    print(1-res)

def segment_labels(Yi):
    idxs = [0] + (np.nonzero(np.diff(Yi))[0]+1).tolist() + [len(Yi)]
    Yi_split = np.array([Yi[idxs[i]] for i in range(len(idxs)-1)])
    return Yi_split

def segment_intervals(Yi):
    idxs = [0] + (np.nonzero(np.diff(Yi))[0]+1).tolist() + [len(Yi)]
    intervals = [(idxs[i],idxs[i+1]) for i in range(len(idxs)-1)]
    return intervals


def overlap_fn(p,y, n_classes=0, bg_class=None, overlap=.1):

        true_intervals = np.array(segment_intervals(y))
        true_labels = segment_labels(y)
        pred_intervals = np.array(segment_intervals(p))
        pred_labels = segment_labels(p)

        # Remove background labels
        if bg_class is not None:
            true_intervals = true_intervals[true_labels!=bg_class]
            true_labels = true_labels[true_labels!=bg_class]
            pred_intervals = pred_intervals[pred_labels!=bg_class]
            pred_labels = pred_labels[pred_labels!=bg_class]

        n_true = true_labels.shape[0]
        n_pred = pred_labels.shape[0]

        # We keep track of the per-class TPs, and FPs.
        # In the end we just sum over them though.
        TP = np.zeros(n_classes, np.float)
        FP = np.zeros(n_classes, np.float)
        true_used = np.zeros(n_true, np.float)

        for j in range(n_pred):
            # Compute IoU against all others
            intersection = np.minimum(pred_intervals[j,1], true_intervals[:,1]) - np.maximum(pred_intervals[j,0], true_intervals[:,0])
            union = np.maximum(pred_intervals[j,1], true_intervals[:,1]) - np.minimum(pred_intervals[j,0], true_intervals[:,0])
            IoU = (intersection / union)*(pred_labels[j]==true_labels)

            # Get the best scoring segment
            idx = IoU.argmax()

            # If the IoU is high enough and the true segment isn't already used
            # Then it is a true positive. Otherwise is it a false positive.
            print(intersection)
            print(union)
            print(IoU)

            if IoU[idx] >= overlap and not true_used[idx]:
                TP[pred_labels[j]] += 1
                true_used[idx] = 1
            else:
                FP[pred_labels[j]] += 1


        TP = TP.sum()
        FP = FP.sum()
        # False negatives are any unused true segment (i.e. "miss")
        FN = n_true - true_used.sum()
        
        precision = TP / (TP+FP)
        recall = TP / (TP+FN)
        F1 = 2 * (precision*recall) / (precision+recall)  #RuntimeWarning: invalid value encountered in double_scalars

        # If the prec+recall=0, it is a NaN. Set these to 0.
        F1 = np.nan_to_num(F1)

        return F1*100

def get_overlap_f1_colin(P, Y, n_classes=0, bg_class=None, overlap=.1, **kwargs):
    def overlap_(p,y, n_classes, bg_class, overlap):

        true_intervals = np.array(segment_intervals(y))
        true_labels = segment_labels(y)
        pred_intervals = np.array(segment_intervals(p))
        pred_labels = segment_labels(p)

        # Remove background labels
        if bg_class is not None:
            true_intervals = true_intervals[true_labels!=bg_class]
            true_labels = true_labels[true_labels!=bg_class]
            pred_intervals = pred_intervals[pred_labels!=bg_class]
            pred_labels = pred_labels[pred_labels!=bg_class]

        n_true = true_labels.shape[0]
        n_pred = pred_labels.shape[0]

        # We keep track of the per-class TPs, and FPs.
        # In the end we just sum over them though.
        TP = np.zeros(n_classes, np.float)
        FP = np.zeros(n_classes, np.float)
        true_used = np.zeros(n_true, np.float)

        for j in range(n_pred):
            # Compute IoU against all others
            intersection = np.minimum(pred_intervals[j,1], true_intervals[:,1]) - np.maximum(pred_intervals[j,0], true_intervals[:,0])
            union = np.maximum(pred_intervals[j,1], true_intervals[:,1]) - np.minimum(pred_intervals[j,0], true_intervals[:,0])
            IoU = (intersection / union)*(pred_labels[j]==true_labels)

            # Get the best scoring segment
            idx = IoU.argmax()

            # If the IoU is high enough and the true segment isn't already used
            # Then it is a true positive. Otherwise is it a false positive.
            if IoU[idx] >= overlap and not true_used[idx]:
                TP[pred_labels[j]] += 1
                true_used[idx] = 1
            else:
                FP[pred_labels[j]] += 1


        TP = TP.sum()
        FP = FP.sum()
        # False negatives are any unused true segment (i.e. "miss")
        FN = n_true - true_used.sum()
        
        precision = TP / (TP+FP)
        recall = TP / (TP+FN)
        F1 = 2 * (precision*recall) / (precision+recall)  #RuntimeWarning: invalid value encountered in double_scalars

        # If the prec+recall=0, it is a NaN. Set these to 0.
        F1 = np.nan_to_num(F1)

        return F1*100

    if type(P) == list:
        return np.mean([overlap_(P[i],Y[i], n_classes, bg_class, overlap) for i in range(len(P))])
    else:
        return overlap_(P, Y, n_classes, bg_class, overlap)

main();