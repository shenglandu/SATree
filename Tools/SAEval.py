# Modified from PointGroup evaluation script: https://github.com/dvlab-research/PointGroup/blob/master/util/eval.py

import os
import numpy as np
import logging
from ply import read_ply


# Specify the global variables
BACKGROUND_LABEL = -100
OVERLAPS = np.append(np.arange(0.5, 0.95, 0.05), 0.25)
MIN_REGION_SIZE = 100


def evaluate_matches(matches):
    overlaps = OVERLAPS

    # Record matches
    ap = np.zeros((len(overlaps), ), np.float32)
    for oi, overlap_th in enumerate(overlaps):
        pred_visited = {}
        for m in matches.keys():
            for pi, pdict in matches[m]['pred'].items():
                if 'filename' in pdict:
                    pred_visited[pdict['filename']] = False
        # initialize
        y_true = np.empty(0)
        y_score = np.empty(0)
        hard_false_negatives = 0
        has_gt = False
        has_pred = False

        # retrieve matches
        for m in matches.keys():
            pred_instances = matches[m]['pred']
            gt_instances = matches[m]['gt']
            # filter groups in ground truth
            gt_instances = [gt for gt in gt_instances.values() if
                            gt['gt_id'] > BACKGROUND_LABEL and gt['vert_count'] >= MIN_REGION_SIZE]
            if gt_instances:
                has_gt = True
            if pred_instances:
                has_pred = True
            cur_true = np.ones(len(gt_instances))
            cur_score = np.ones(len(gt_instances)) * (-float("inf"))
            cur_match = np.zeros(len(gt_instances)).astype(bool)
            # collect matches
            for gti, gt in enumerate(gt_instances):
                found_match = False
                for pred in gt['matched_pred']:
                    # greedy assignments
                    if pred_visited[pred['filename']]:
                        continue
                    overlap = float(pred['intersection'])/(gt['vert_count']+pred['vert_count']-pred['intersection'])
                    if overlap > overlap_th:
                        # confidence = pred['confidence']
                        confidence = overlap # use overlap as confidence
                        # if already have a prediction for this gt, the prediction with the lower score is FP
                        if cur_match[gti]:
                            max_score = max(cur_score[gti], confidence)
                            min_score = min(cur_score[gti], confidence)
                            cur_score[gti] = max_score
                            # append false positive
                            cur_true = np.append(cur_true, 0)
                            cur_score = np.append(cur_score, min_score)
                            cur_match = np.append(cur_match, True)
                        # otherwise set score
                        else:
                            found_match = True
                            cur_match[gti] = True
                            cur_score[gti] = confidence
                            pred_visited[pred['filename']] = True
                if not found_match:
                    hard_false_negatives += 1
            # remove non-matched ground truth instances
            cur_true = cur_true[cur_match == True]
            cur_score = cur_score[cur_match == True]

            # collect non-matched predictions as false positive
            for pi, pred in pred_instances.items():
                found_gt = False
                for gt in pred['matched_gt']:
                    overlap = float(gt['intersection']) / (
                            gt['vert_count'] + pred['vert_count'] - gt['intersection'])
                    if overlap > overlap_th:
                        found_gt = True
                        break
                if not found_gt:
                    num_ignore = pred['void_intersection']
                    for gt in pred['matched_gt']:
                        # small ground truth instances
                        if gt['vert_count'] < MIN_REGION_SIZE:
                            num_ignore += gt['intersection']
                    proportion_ignore = float(num_ignore) / pred['vert_count']
                    # if not ignored append false positive
                    if proportion_ignore <= overlap_th:
                        cur_true = np.append(cur_true, 0)
                        confidence = pred["confidence"]
                        cur_score = np.append(cur_score, confidence)

            # append to overall results
            y_true = np.append(y_true, cur_true)
            y_score = np.append(y_score, cur_score)

        # compute average precision
        if has_gt and has_pred:
            # compute precision recall curve first

            # sorting and cumsum
            score_arg_sort = np.argsort(y_score)
            y_score_sorted = y_score[score_arg_sort]
            y_true_sorted = y_true[score_arg_sort]
            y_true_sorted_cumsum = np.cumsum(y_true_sorted)

            # unique thresholds
            (thresholds, unique_indices) = np.unique(y_score_sorted, return_index=True)
            num_prec_recall = len(unique_indices) + 1

            # prepare precision recall
            num_examples = len(y_score_sorted)
            if (len(y_true_sorted_cumsum) == 0):
                num_true_examples = 0
            else:
                num_true_examples = y_true_sorted_cumsum[-1]
            precision = np.zeros(num_prec_recall)
            recall = np.zeros(num_prec_recall)

            # deal with the first point
            y_true_sorted_cumsum = np.append(y_true_sorted_cumsum, 0)
            # deal with remaining
            for idx_res, idx_scores in enumerate(unique_indices):
                cumsum = y_true_sorted_cumsum[idx_scores - 1]
                tp = num_true_examples - cumsum
                fp = num_examples - idx_scores - tp
                fn = cumsum + hard_false_negatives
                p = float(tp) / (tp + fp)
                r = float(tp) / (tp + fn)
                precision[idx_res] = p
                recall[idx_res] = r

            # first point in curve is artificial
            precision[-1] = 1.
            recall[-1] = 0.

            # compute average of precision-recall curve
            recall_for_conv = np.copy(recall)
            recall_for_conv = np.append(recall_for_conv[0], recall_for_conv)
            recall_for_conv = np.append(recall_for_conv, 0.)

            stepWidths = np.convolve(recall_for_conv, [-0.5, 0, 0.5], 'valid')
            # integrate is now simply a dot product
            ap_current = np.dot(precision, stepWidths)

        elif has_gt:
            ap_current = 0.0
        else:
            ap_current = float('nan')
        ap[oi] = ap_current

    return ap


def compute_averages(aps):
    o50 = np.where(np.isclose(OVERLAPS,0.5))
    o25 = np.where(np.isclose(OVERLAPS,0.25))
    oAllBut25 = np.where(np.logical_not(np.isclose(OVERLAPS,0.25)))
    avg_dict = {}
    avg_dict['all_ap'] = np.nanmean(aps[oAllBut25])
    avg_dict['all_ap_50%'] = np.nanmean(aps[o50])
    avg_dict['all_ap_25%'] = np.nanmean(aps[o25])
    return avg_dict


def assign_instances_for_scan(scene_name, seg_file):
    # Load the seg data
    seg_data = read_ply(seg_file)
    pred_ids = seg_data['pred_ins'].astype(np.int32)
    gt_ids = seg_data['gt_ins'].astype(np.int32)

    # Initialize gt2pred
    gt2pred = {}
    instance_ids = np.unique(gt_ids)
    for id in instance_ids:
        if id == BACKGROUND_LABEL:
            continue
        gt_instance = {}
        gt_instance['gt_id'] = id
        gt_instance['vert_count'] = np.count_nonzero(id == gt_ids)
        gt_instance['matched_pred'] = []
        gt2pred[id] = gt_instance

    # Initialize pred2gt
    pred2gt = {}
    num_pred_instances = 0
    bool_void = gt_ids == BACKGROUND_LABEL

    # Load prediction id
    if len(pred_ids) != len(gt_ids):
        print('wrong number of lines in mask: ' + '# pred (%d) vs #gt (%d)' % (len(pred_ids), len(gt_ids)))

    # Retrieve over masks
    instance_preds = np.unique(pred_ids)
    for i in instance_preds:
        if i == BACKGROUND_LABEL:
            continue
        pred_mask = pred_ids == i
        num = np.count_nonzero(pred_mask)
        if num < MIN_REGION_SIZE:
            continue  # skip if too small

        # Initialize prediction instance
        pred_instance = {}
        pred_instance['filename'] = '{}_{:03d}'.format(scene_name, num_pred_instances)
        pred_instance['pred_id'] = num_pred_instances
        pred_instance['vert_count'] = num
        pred_instance['confidence'] = 0
        pred_instance['void_intersection'] = np.count_nonzero(np.logical_and(bool_void, pred_mask))

        # matched gt instances
        matched_gt = []
        for gt_id, gt_inst in gt2pred.items():
            intersection = np.count_nonzero(np.logical_and(gt_ids == gt_id, pred_mask))
            if intersection > 0:
                gt_copy = gt_inst.copy()
                pred_copy = pred_instance.copy()
                gt_copy['intersection'] = intersection
                pred_copy['intersection'] = intersection
                matched_gt.append(gt_copy)
                gt2pred[gt_id]['matched_pred'].append(pred_copy)
        pred_instance['matched_gt'] = matched_gt
        pred2gt[num_pred_instances] = pred_instance
        num_pred_instances += 1

    return gt2pred, pred2gt


def print_results(forest_name, avgs):
    # Create logger
    log_path = 'Eval'
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    log_file = os.path.join(log_path, forest_name + '_eval.log')
    logger = create_logger(log_file)

    sep = ""
    col1 = ":"
    lineLen = 64
    logger.info("")
    logger.info("#" * lineLen)
    line  = ""
    line += "{:<15}".format("what"      ) + sep + col1
    line += "{:>15}".format("AP"        ) + sep
    line += "{:>15}".format("AP_50%"    ) + sep
    line += "{:>15}".format("AP_25%"    ) + sep
    logger.info(line)

    all_ap_avg = avgs["all_ap"]
    all_ap_50o = avgs["all_ap_50%"]
    all_ap_25o = avgs["all_ap_25%"]

    logger.info("-" * lineLen)
    line = "{:<15}".format("average") + sep + col1
    line += "{:>15.3f}".format(all_ap_avg) + sep
    line += "{:>15.3f}".format(all_ap_50o) + sep
    line += "{:>15.3f}".format(all_ap_25o) + sep
    logger.info(line)
    logger.info("")


def create_logger(log_file):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler()
    log_format = '[%(asctime)s  %(levelname)s  %(filename)s  line %(lineno)d  %(process)d]  %(message)s'
    handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(handler)

    logging.basicConfig(level=logging.DEBUG, format=log_format, filename=log_file)    # filename: build a FileHandler
    return logger


def evaluate(forest_name, seg_files):
    # Initialize
    matches = {}

    for i in range(len(seg_files)):
        # obtain the scene name
        scene_name = seg_files[i].split('/')[-1]
        scene_name = scene_name.split('.')[0][:-4]
        if forest_name not in scene_name:
            continue
        print('evaluating scene: ' + scene_name + '------')
        # assign gt to predictions
        gt2pred, pred2gt = assign_instances_for_scan(scene_name, seg_files[i])
        matches[scene_name] = {}
        matches[scene_name]['gt'] = gt2pred
        matches[scene_name]['pred'] = pred2gt

    ap_scores = evaluate_matches(matches)
    avgs = compute_averages(ap_scores)

    # Write results to a logger
    print_results(forest_name, avgs)


def main():
    # Specify paths
    seg_path = '/mnt/materials/PROJECT#3_Tree_Segmentation/Code/0_Preprocessing/Tree_xyz/forinstance'
    forest_name = 'RMIT'

    # Get prediction files and gt files
    seg_files = [f for f in os.listdir(seg_path) if f.endswith('_seg.ply')]
    if len(seg_files) == 0:
        print('No result seg files found.')
        return
    for i in range(len(seg_files)):
        seg_files[i] = os.path.join(seg_path, seg_files[i])

    # Evaluate
    evaluate(forest_name, seg_files)


if __name__ == '__main__':
    main()


