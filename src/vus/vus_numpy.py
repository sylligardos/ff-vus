"""
@who: Emmanouil Sylligardos
@where: Ecole Normale Superieur (ENS), Paris, France
@when: PhD Candidate, 2nd year (2025)
@what: FF-VUS
"""

from utils.utils import time_it

import numpy as np
import math
from skimage.util.shape import view_as_windows as viewW
import matplotlib.pyplot as plt
import seaborn as sns


class VUSNumpy():
    def __init__(
            self, 
            slope_size=100, 
            step=1, 
            zita=(1/math.sqrt(2)),
            global_mask=False, 
            slopes='precomputed',
            existence='optimized',
            conf_matrix='dynamic',
            interpolation='stepwise',
            metric='vus_pr'
        ):
        """
        Initialize the VUSNumpy metric class.

        Args:
            slope_size (int): Number of slope values for anomaly encoding.
            step (int): Increment between slope values; must evenly divide slope_size.
            zita (float): Slope parameter, must be in [0, 1].
            global_mask (bool): Whether to apply a global mask to the data.
            slopes (str): Slope computation mode, either 'precomputed' or 'function'.
            existence (str): Existence calculation mode: 'None', 'trivial', 'optimized', or 'matrix'.
            conf_matrix (str): Confusion matrix calculation mode: 'trivial', 'dynamic', or 'dynamic_plus'.
            interpolation (str): Interpolation method for PR curve: 'linear' or 'stepwise'.
            metric (str): Metric to compute: 'auc_pr', 'auc_roc', 'vus_pr', 'vus_roc', or 'all'.
        """
        if slope_size < 0 or step < 0:
            raise ValueError(f"Error with the slope_size {slope_size} or step {step}. They should be positive values.")
        if step == 0 and slope_size > 0:
            raise ValueError(f"Error with step {step} and slope_size {slope_size}. Step can't be 0 with non-zero slope_size.")
        if (step > 0 and slope_size > 0) and (slope_size % step) != 0:
            raise ValueError(f"Error with the step: {step}. It should be a positive value and a perfect divider of the slope_size.")
        if zita < 0 and zita > 1:
            raise ValueError(f"Error with the zita: {zita}. It should be a value between 0 and 1.")
        
        self.slope_size, self.step = slope_size, step
        self.zita, self.global_mask = zita, global_mask
        
        self.slope_values = np.arange(self.slope_size + 1, step=self.step) if step > 0 else np.array([0])
        self.n_slopes = self.slope_values.shape[0]

        # Define valid argument sets
        slopes_args = ['precomputed', 'function']
        existence_args = ['None', 'trivial', 'optimized', 'matrix']
        conf_matrix_args = ['trivial', 'dynamic', 'dynamic_plus']
        interpolation_args = ['linear', 'stepwise']
        metric_args = ['auc_pr', 'auc_roc', 'vus_pr', 'vus_roc', 'all']

        # Validate and set modes
        self.add_slopes_mode = self.validate_args(slopes, slopes_args)
        self.existence_mode = self.validate_args(existence, existence_args)
        self.conf_matrix_mode = self.validate_args(conf_matrix, conf_matrix_args)
        self.interpolation_mode = self.validate_args(interpolation, interpolation_args)
        self.metric_mode = self.validate_args(metric, metric_args)

        # Precompute slopes
        self.prepare_procomputed_slopes()


    def validate_args(self, value, valid_options):
        if value not in valid_options:
            raise ValueError(f"Unknown argument {value}. Must be one of {valid_options}.")
        return value
    
    def prepare_procomputed_slopes(self):
        if self.add_slopes_mode:
            if self.slope_size > 0:
                self.pos_slopes = self.precompute_slopes()
                self.neg_slopes = self.pos_slopes[:, ::-1]
            else:
                self.pos_slopes, self.neg_slopes = np.array([]), np.array([])

    @time_it
    def compute(self, label, score):
        """
        Main computation function for the VUS metric.

        Args:
            label (np.ndarray): Binary ground truth vector.
            score (np.ndarray): Prediction scores.

        Returns:
            vus_pr (float): Computed VUS-PR metric.
            time_analysis (dict): Timing breakdown for each computation step.
        """
        ((start_no_edges, end_no_edges), (start_with_edges, end_with_edges)), time_anom_coord = self.get_anomalies_coordinates(label)
        safe_mask, time_safe_mask = self.create_safe_mask(label, start_with_edges, end_with_edges, extra_safe=self.global_mask)

        thresholds, time_thresholds = self.get_unique_thresholds(score)

        # Apply global mask
        if self.global_mask:
            sm, extra_time_sm = self.get_score_mask(score[~safe_mask], thresholds)
            extra_fp = sm.sum(axis=1)
            label, score = label[safe_mask], score[safe_mask]

            ((start_no_edges, end_no_edges), (start_with_edges, end_with_edges)), extra_time_anom_coord = self.get_anomalies_coordinates(label)
            safe_mask, extra_time_safe_mask = self.create_safe_mask(label, start_with_edges, end_with_edges)
            
        pos, time_pos = self.distance_from_anomaly(label, start_with_edges, end_with_edges, clip=True)
        sm, time_sm = self.get_score_mask(score, thresholds)

        labels, time_slopes = self.add_slopes(label, start_no_edges, end_no_edges, pos)
        existence, time_existence = self.compute_existence(labels, sm, score, thresholds, start_with_edges, end_with_edges, safe_mask)
        (fp, fn, tp, positives, negatives, fpr), time_confusion = self.compute_confusion_matrix(labels, sm)
        
        # Add rest of FPs
        if self.global_mask:
            fp += extra_fp
            time_anom_coord += extra_time_anom_coord
            time_safe_mask += extra_time_safe_mask
            time_sm += extra_time_sm

        (precision, recall), time_pr_rec = self.precision_recall_curve(tp, fp, positives, existence)
        vus_pr, time_integral = self.auc(recall, precision)

        time_analysis = {
            "Anomaly Coordinates time": time_anom_coord,
            "Safe mask time": time_safe_mask,
            "Thresholds time": time_thresholds,
            "Score mask time": time_sm,
            "Position time": time_pos,
            "Slopes time": time_slopes,
            "Existence time": time_existence,
            "Confusion matrix time": time_confusion,
            "Precision recall curve time": time_pr_rec,
            "Integral time": time_integral,
        }

        return vus_pr, time_analysis
    
    @time_it
    def get_score_mask(self, score, thresholds):
        return score >= thresholds[:, None]
    
    @time_it
    def get_unique_thresholds(self, score):
        return np.sort(np.unique(score))[::-1]
    
    @time_it
    def create_safe_mask(self, label, start_points, end_points, extra_safe=False):
        """
        A safe mask is a mask of the label that every anomaly is one point bigger (left and right)
        than the bigger slope. This allows us to mask the label safely, without changing anything in the implementation.
        """
        length = label.shape[0]
        mask = np.zeros(length, dtype=np.int8)

        safe_extension = self.slope_size + 1 + int(extra_safe)
        
        start_safe_points = np.maximum(start_points - safe_extension, 0)
        end_safe_points = np.minimum(end_points + safe_extension + 1, length - 1)

        np.add.at(mask, start_safe_points, 1)
        np.add.at(mask, end_safe_points, -1)

        mask = np.cumsum(mask)
        mask = mask > 0
        # TODO: Why does it work?
        mask[-1] = (end_safe_points[-1] == (length - 1)) if extra_safe else label[-1]
        
        return mask
    
    def _strided_indexing_roll(self, a, r):
        # Concatenate with sliced to cover all rolls
        a_ext = np.concatenate((a, a[:, : - 1]), axis=1)

        # Get sliding windows; use advanced-indexing to select appropriate ones
        n = a.shape[1]
        return viewW(a_ext, (1, n))[np.arange(len(r)), (n - r) % n, 0]
    
    def _slopes_mask(self, n, m, as_type='int'):
        k = m // n
        row_indices = np.arange(1, n + 1)[:, None]
        col_indices = np.arange(m)

        mask = col_indices < (row_indices * k)
        return ~mask
    
    def precompute_slopes(self):
        '''
        Create all the required slopes at once.
        step should be an exact divider of l.
        The very first slopes, which is the slope of length 0, is not provided
        Every row is a slope, row 0 is the smallest and the last row the biggest
        These are the positive slopes
        '''
        steps = ((1 - self.zita) / self.slope_values[1:])
        stops = 1 + steps[::-1] * self.slope_values[:-1]
        
        slopes = np.linspace(self.zita, stops, self.slope_size + 1).T[::-1]
        slopes[self._slopes_mask(*slopes.shape)] = 0
        
        rolls = self.slope_size - self.slope_values[1:]
        shifted_slopes = self._strided_indexing_roll(slopes, rolls)

        return shifted_slopes
    
    @time_it
    def distance_from_anomaly(self, label, start_points, end_points, clip=False):
        '''
        For every point in the label, returns a time series that shows the distance
        of that point to its closest anomaly. Compute the distance of each point 
        to every anomaly and keep the minimum.
        '''
        length = len(label)

        pos = np.full(length, np.inf)
        if start_points.shape[0] == 0 and end_points.shape[0] == 0:
            return pos
        
        anomaly_boundaries = np.concat([start_points, end_points])
        indices = np.arange(length)[:, None]

        distances = np.abs(indices - anomaly_boundaries)
        pos = np.min(distances, axis=1)
        
        if clip:
            pos[label.astype(bool)] = 0
            np.clip(pos, a_min=0, a_max=self.slope_size, out=pos)
        
        return pos
    
    @time_it
    def get_anomalies_coordinates(self, label: np.array):
        """
        Return the starting and ending points of all anomalies in label,
        both with and without edge inclusion.

        Args:
            label (np.array): vector of 1s and 0s

        Returns:
            ((start_no_edges, end_no_edges), (start_with_edges, end_with_edges))
        """
        diff = np.diff(label)
        start_no_edges = np.where(diff == 1)[0] + 1
        end_no_edges = np.where(diff == -1)[0]
        
        start_with_edges = np.append([0], start_no_edges) if label[0] else start_no_edges
        end_with_edges = np.append(end_no_edges, len(label) - 1) if label[-1] else end_no_edges

        if start_with_edges.shape != end_with_edges.shape:
            raise ValueError(f'The number of start and end points of anomalies does not match, {start_with_edges} != {end_with_edges}')
            
        return (start_no_edges, end_no_edges), (start_with_edges, end_with_edges)

    def add_slopes_function(self, label, pos):
        """
        Compute the slopes of a label using a predefined function.

        This function is about 5 - 10 times slower than the precomputed version on CPUs.
        The pos part requires about 10% of the total execution time.
        
        
        Another potential improvement for the motivated
        f_pos[1:, valid_mask][(slope_values - pos[:, valid_mask]) < 0] = 0
        
        Args:
            label: Binary vector indicating anomaly regions (shape: [T]).
            pos : Distance to the nearest anomaly boundary (shape: [T]).
            
        Returns:
            TODO
                """

        valid_mask = (self.slope_size - pos) >= 0
        slope_values = self.slope_values[1:, None]
        pos = pos[None, :]

        f_pos = np.zeros((self.n_slopes, len(label)))
        f_pos[0] = label
        f_pos[1:, valid_mask] = 1 - (((1 - self.zita) * pos[:, valid_mask]) / slope_values)
        f_pos[1:][(slope_values - pos) < 0] = 0                                                 # Improvement goes here
        f_pos[1:, label.astype(bool)] = 1

        return f_pos

    def add_slopes_precomputed(self, label, start_points, end_points):
        """
        Start points and end points of anomalies are without edges
        """
        result = np.repeat([label], self.n_slopes, axis=0)
        if self.n_slopes == 1: 
            return result

        for curr_point in start_points:
            slope_start = max(curr_point - self.slope_size, 0)
            slope_end = curr_point + 1  # to include it in the transformation, read desc
            adjusted_l = slope_end - slope_start
            result[1:, slope_start: slope_end] = np.maximum(result[1:, slope_start: slope_end], self.pos_slopes[:, -adjusted_l:])
        for curr_point in end_points:
            slope_start = curr_point
            slope_end = min(curr_point + self.slope_size, len(label) - 1) + 1       # to include it in the transformation, read desc
            adjusted_l = slope_end - slope_start
            result[1:, slope_start: slope_end] = np.maximum(result[1:, slope_start: slope_end], self.neg_slopes[:, :adjusted_l])

        return result
    
    @time_it
    def add_slopes(self, label, start_points, end_points, pos):
        if self.add_slopes_mode == 'precomputed':
            return self.add_slopes_precomputed(label, start_points, end_points)
        else:
            return self.add_slopes_function(label, pos)
    
    @time_it
    def compute_existence(self, labels, sm, score, thresholds, start_points, end_points, safe_mask):
        if self.existence_mode == 'optimized':
            return self.existence_optimized(labels, score, thresholds, start_points, end_points)
        if self.existence_mode == 'matrix':
            return self.existence_matrix(labels, sm, safe_mask)
        if self.existence_mode == 'None':
            # return self.no_existence(thresholds)
            return None
        if self.existence_mode == 'trivial':
            return self.existence_trivial(labels, sm, thresholds, start_points, end_points)
    
    def no_existence(self, thresholds):
        return np.ones((self.n_slopes, len(thresholds)))

    def existence_trivial(self, labels, score_mask, thresholds, start_points, end_points):
        n_slopes, T = labels.shape
        n_thresholds = thresholds.shape[0]
        existence = np.zeros((n_slopes, n_thresholds))
        n_anomalies = np.full(n_slopes, start_points.shape[0], dtype=int)

        all_start_points = np.maximum((np.tile(start_points, (n_slopes, 1)) - self.slope_values[:, None]), 0)
        all_end_points = np.minimum((np.tile(end_points, (n_slopes, 1)) + self.slope_values[:, None]), T - 1)

        for k, (start, end) in enumerate(zip(start_points[1:], end_points[:-1])):
            distance = start - end
            overlap = (2 * self.slope_size) - distance        
            if overlap > 0:
                index = math.ceil((self.slope_size - (overlap // 2)) / self.step)
                n_anomalies[index:] -= 1
                all_start_points[index:, (k + 1)] = -1
                all_end_points[index:, k] = -1

        for i in range(n_slopes):
            for j in range(n_thresholds):
                for k in range(n_anomalies[i]):
                    start = all_start_points[i, all_start_points[i] >= 0][k]
                    end = all_end_points[i, all_end_points[i] >= 0][k]  
                    if np.any(score_mask[j, start: end + 1]):
                        existence[i, j] += 1

        return existence / n_anomalies[:, None]

    def existence_optimized(self, labels, score, thresholds, start_points, end_points):
        """
        Optimized existence computation (~400x faster than trivial implementation).
        Suitable for CPU applications.
        """
        n_slopes, T = labels.shape
        n_thresholds = thresholds.shape[0]
        existence = np.zeros((n_slopes, n_thresholds))
        thresholds_dict = {value: index for index, value in enumerate(thresholds)}
        n_anomalies = np.full(n_slopes, start_points.shape[0], dtype=int)
        
        all_start_points = np.maximum((np.tile(start_points, (n_slopes, 1)) - self.slope_values[:, None]), 0)
        all_end_points = np.minimum((np.tile(end_points, (n_slopes, 1)) + self.slope_values[:, None]), T - 1)

        anomalies_overlaps = np.zeros(start_points.shape, dtype=int)
        for k, (start, end) in enumerate(zip(start_points[1:], end_points[:-1])):
            distance = start - end
            overlap = (2 * self.slope_size) - distance        
            if overlap > 0:
                index = math.ceil((self.slope_size - (overlap // 2)) / self.step)
                n_anomalies[index:] -= 1
                anomalies_overlaps[k] = index
        
        prev_existence = np.zeros((n_slopes, n_thresholds), dtype=int)
        for k in range(n_anomalies[0]):
            overlap_index = anomalies_overlaps[k - 1]       # This works because the last one never has an overlap ;)
            
            og_start = start_points[k]
            og_end = end_points[k]
            
            curr_existence = np.zeros((n_slopes, n_thresholds), dtype=int)
            tmp_existence = np.zeros((n_slopes, n_thresholds), dtype=int)

            i = n_slopes - 1
            while i >= 0:
                start = all_start_points[i, k]
                end = all_end_points[i, k]
                
                argmax_t = [score[start:og_start + 1][::-1].argmax(), score[og_start:og_end + 1].argmax(), score[og_end:end + 1].argmax()]
                max_t = [score[og_start - argmax_t[0]], score[og_start + argmax_t[1]], score[og_end + argmax_t[2]]]
            
                if max_t[1] >= max_t[0] and max_t[1] >= max_t[2]:   # Case middle
                    threshold_index = thresholds_dict[max_t[1]]
                    curr_existence[:i+1, threshold_index:] += 1
                    break
                elif max_t[0] > max_t[2]:    # Case left
                    threshold_index = thresholds_dict[max_t[0]]
                    curr_existence[math.ceil(argmax_t[0] / self.step):i+1, threshold_index:] += 1
                    i = math.ceil(argmax_t[0] / self.step) - 1
                elif max_t[2] >= max_t[0]:   # Case right
                    threshold_index = thresholds_dict[max_t[2]]
                    curr_existence[math.ceil(argmax_t[2] / self.step):i+1, threshold_index:] += 1
                    i = math.ceil(argmax_t[2] / self.step) - 1

            if overlap_index > 0:
                tmp_existence[overlap_index:] = np.logical_or(curr_existence[overlap_index:], prev_existence[overlap_index:])
                curr_existence[overlap_index:] = np.logical_and(curr_existence[overlap_index:], np.logical_not(prev_existence[overlap_index:]))
            existence += curr_existence
            prev_existence = np.logical_or(tmp_existence, curr_existence)

        return existence / n_anomalies[:, None]

    def existence_matrix(self, labels, score_mask, safe_mask):
        """
        This implementation enables parallel computation for the existence.
        However, it is only ~5 times faster than the trivial implementation 
        and ~80 times slower than the optimized one on CPU.

        Additionally, due to impelementation specific reasons, it is not equal to the 
        previous implementation by an average error of 0.005. It is not wrong per say
        but acts differently in the edge case where the slopes of two anomalies meet
        on an even step.
        """
        # Compute mask for relevant points
        if not self.global_mask:
            labels = labels[:, safe_mask]
            score_mask = score_mask[:, safe_mask]

        # Normalize labels to 0 or 1
        norm_labels = (labels > 0).astype(np.int8)

        # Compute step function (stairs)
        diff = np.diff(norm_labels, prepend=np.zeros((norm_labels.shape[0], 1)), axis=1)
        diff = np.clip(diff, 0, 1)
        stairs = np.cumsum(diff, axis=1)
        labels_stairs = norm_labels * stairs

        # Multiply every score with every slope
        score_hat = labels_stairs[:, None, :] * score_mask[None, :, :]

        # Compute cumulative max along the time dimension
        cm = np.maximum.accumulate(score_hat, axis=2)

        # Compute differences and normalize
        cm_diff = np.diff(cm, axis=2)
        cm_diff_norm = np.maximum(cm_diff - 1, 0)

        # Compute total anomalies and missed anomalies
        total_anomalies = stairs[:, -1][:, None]
        final_anomalies_missed = total_anomalies - cm[:, :, -1]
        n_anomalies_not_found = np.sum(cm_diff_norm, axis=2) + final_anomalies_missed
        n_anomalies_found = total_anomalies - n_anomalies_not_found

        return (n_anomalies_found / total_anomalies)
    
    @time_it
    def compute_confusion_matrix(self, labels, sm):
        """
        Scikit-learn order: tn, fp, fn, tp
        """

        if self.conf_matrix_mode == "trivial":
            conf_matrix = self.conf_matrix_trivial(labels, sm)
        elif self.conf_matrix_mode == "dynamic":
            conf_matrix = self.conf_matrix_dyn(labels, sm)
        else:
            conf_matrix = self.conf_matrix_dyn_plus(labels, sm)
        fn, tp, positives = conf_matrix
        
        # conf_matrix = self.conf_matrix_dyn(labels, sm)
        # conf_matrix_1 = self.conf_matrix_dyn_plus(labels, sm)
        # fn, tp, positives = conf_matrix
        # fn_1, tp_1, positives_1 = conf_matrix_1
        # print(tp)
        # # print(np.mean(tp - tp_1), np.max(tp - tp_1), np.all(tp == tp_1))
        # # print(np.mean(fn - fn_1), np.max(fn - fn_1), np.all(fn == fn_1))
        # # print(np.mean(positives - positives_1), np.max(positives - positives_1), np.all(positives == positives_1))
        # exit()

        fp = sm.sum(axis=1) - tp
        negatives = labels[0].shape[0] - positives
        fpr = fp / negatives

        return fp, fn, tp, positives, negatives, fpr

    def conf_matrix_trivial(self, labels, sm):
        label = labels[0]
        sm_inv = ~sm

        true_positives = np.matmul(labels, sm.T)
        false_negatives = np.matmul(label, sm_inv.T)[:, np.newaxis].T

        positives = ((true_positives + false_negatives) + label.sum()) / 2

        return false_negatives, true_positives, positives
    
    def conf_matrix_dyn(self, labels, sm):
        if not self.global_mask:
            mask = np.where(labels[-1] > 0)[0]
            labels = labels[:, mask]
            sm = sm[:, mask]

        true_positives = np.matmul(labels, sm.T)
        false_negatives = np.matmul(labels[0], ~sm.T)[None, :]
        positives = ((true_positives + false_negatives) + labels[0].sum()) / 2

        return false_negatives, true_positives, positives
    
    def conf_matrix_dyn_plus(self, labels, sm):
        if not self.global_mask:
            slope_mask = np.where(np.logical_and(labels[-1] > 0, labels[-1] < 1))[0]
        else:
            slope_mask = np.where(labels[-1] < 1)[0]
        label_as_mask = np.where(labels[0])[0]
        
        initial_tps = sm[:, label_as_mask].sum(axis=1)
        slope_tps = np.matmul(labels[:, slope_mask], sm[:, slope_mask].T)
        true_positives = initial_tps + slope_tps
        false_negatives = (~sm)[:, label_as_mask].sum(axis=1)
        positives = ((true_positives + false_negatives) + label_as_mask.shape[0]) / 2

        return false_negatives, true_positives, positives
    
    @time_it
    def precision_recall_curve(self, tp, fp, positives, existence):
        ones, zeros = np.ones(self.n_slopes)[:, np.newaxis], np.zeros(self.n_slopes)[:, np.newaxis]
        precision = np.hstack((ones, (tp / (tp + fp))))
        recall = np.hstack((zeros, (tp / positives)))
        recall[recall > 1] = 1
        if existence is not None:
            recall[:, 1:] = np.multiply(recall[:, 1:], existence)

        return precision, recall
    
    @time_it
    def auc(self, x, y):
        if self.interpolation_mode == 'linear':
            return self.linear_interpolation(x, y).mean()
        else:
            return self.stepwise_interpolation(x, y).mean()

    def linear_interpolation(self, x, y):
        return np.trapezoid(y, x, axis=1)

    def stepwise_interpolation(self, x, y):
        width_pr = x[:, 1:] - x[:, :-1]
        height_pr = y[:, 1:]
        return np.sum(np.multiply(width_pr, height_pr), axis=1)
    
    def analyze_label(self, label):
        """
        Analyze a binary time series anomaly label vector
        and return basic metrics.

        Args:
            label (np.ndarray): a 1D numpy array of 0s and 1s
        Return:
            length (int): total length of the label
            n_anomalies (int): number of anomalies in the label
            anomalies_avg_length (int): the average length of the anomalies
        """
        length = len(label)

        start_points, end_points = self.get_anomalies_coordinates(label)
        end_points += 1     # end points are exclusive, so this is necessary
        
        n_anomalies = start_points.shape[0]
        anomaly_lengths = np.array([e - s for s, e in zip(start_points, end_points)])
        anomalies_avg_length = np.mean(anomaly_lengths)
        
        return length, n_anomalies, anomalies_avg_length