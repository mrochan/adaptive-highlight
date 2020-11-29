import os

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve
from torch.nn import functional as F


def postprocess_prediction(strided_prediction, nframes, fps=30):
    # expanding dimensions of the current strided_prediction to satisfy F.interpolate required input dimension
    if strided_prediction.dim() == 1:
        strided_prediction = strided_prediction.view(1, 1, strided_prediction.shape[0])
    elif strided_prediction.dim() == 2:
        strided_prediction = strided_prediction.unsqueeze(0)

    # convert strided output to match the number of frames
    nframes_prediction = F.interpolate(strided_prediction, size=nframes, mode='nearest').squeeze().cpu().numpy()

    # structure the nframes vector to 5s segments similar to video2gif
    segments = [(start, int(start + fps * 5)) for start in range(0, nframes, int(fps * 5))]

    # compute the average of each segment and assign the averaged score to all the frames in that segment
    for segment in segments:
        avg_score = np.mean(nframes_prediction[segment[0]: segment[1]])
        nframes_prediction[segment[0]: segment[1]] = avg_score

    # return the vector with length equal to the nframes
    return nframes_prediction


# Reference: https://github.com/gyglim/video2gif_dataset/blob/master/v2g_evaluation/__init__.py
def average_precision(y_true, y_predict, interpolate=True, point_11=False):
    '''
    Average precision in different formats: (non-) interpolated and/or 11-point approximated
    point_11=True and interpolate=True corresponds to the 11-point interpolated AP used in
    the PASCAL VOC challenge up to the 2008 edition and has been verfied against the vlfeat implementation
    The exact average precision (interpolate=False, point_11=False) corresponds to the one of vl_feat

    :param y_true: list/ numpy vector of true labels in {0,1} for each element
    :param y_predict: predicted score for each element
    :param interpolate: Use interpolation?
    :param point_11: Use 11-point approximation to average precision?
    :return: average precision
    '''

    # Check inputs
    assert len(y_true) == len(y_predict), "Prediction and ground truth need to be of the same length"
    if len(set(y_true)) == 1:
        if y_true[0] == 0:
            raise ValueError('True labels cannot all be zero')
        else:
            return 1
    else:
        assert sorted(set(y_true)) == [0, 1], "Ground truth can only contain elements {0,1}"

    # Compute precision and recall
    precision, recall, _ = precision_recall_curve(y_true, y_predict)
    recall = recall.astype(np.float32)

    if interpolate:  # Compute the interpolated precision
        for i in range(1, len(precision)):
            precision[i] = max(precision[i - 1], precision[i])

    if point_11:  # Compute the 11-point approximated AP
        precision_11 = [precision[np.where(recall >= t)[0][-1]] for t in np.arange(0, 1.01, 0.1)]
        return np.mean(precision_11)
    else:  # Compute the AP using precision at every additionally recalled sample
        indices = np.where(np.diff(recall))
        return np.mean(precision[indices])


def get_user_ground_truth(user_dir):
    user_gt_path = os.path.join(user_dir, 'train', 'gt')

    # check if the user exists
    if not os.path.exists(user_gt_path):
        raise IOError('User path {} does not exist'.format(user_dir))
    else:
        # read all the ground-truth train videos, ideally should be only one user ground-truth train video with one or more user selected clips
        user_gt_videos = [os.path.join(user_gt_path, uv) for uv in os.listdir(user_gt_path)]
        assert len(user_gt_videos) > 0, "No ground-truth video folders found for the user"

        uvideo = user_gt_videos[0]
        # collecting all the ground-truth csvs of all the user selected clips from the same video
        gt_video_csvs = sorted([os.path.join(uvideo, uclip) for uclip in os.listdir(uvideo) if uclip.endswith('nframes.csv')])
        # extracting the binary valued ground-truth value from each of the above read csv file
        gt_video_labels = [pd.read_csv(uclip, delimiter=',')['gt_value'].values for uclip in gt_video_csvs]

        return gt_video_labels


