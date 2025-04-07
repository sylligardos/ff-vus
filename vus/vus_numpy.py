"""
@who: Emmanouil Sylligardos
@where: Ecole Normale Superieur (ENS), Paris, France
@when: PhD Candidate, 2nd year (2025)
@what: FF-VUS
"""

from src.utils.utils import time_it, compute_slopes_and_compare, visualize_differences_1d, compare_vectors

import numpy as np
import math
from skimage.util.shape import view_as_windows as viewW
import matplotlib.pyplot as plt
import seaborn as sns
import time
from scipy.special import digamma


class VUSNumpy():
    def __init__(
            self, 
            slope_size=100, 
            step=1, 
            zita=(1/math.sqrt(2)), 
            slopes='precomputed',
            existence='optimized',
            conf_matrix='dynamic', 
            interpolation='stepwise',
            metric='vus_pr',
        ):
        """
        Initialize the VUS metric.
        TODO: Instead of saving the variables you can save the appropriate functions in the variables instead of

        Args:
            TODO: Write the arguments description when done
        """
        if slope_size < 0:
            raise ValueError(f"Error with the slope_size: {slope_size}. It should be a positive value.")
        if step < 0 or (slope_size % step) != 0:
            raise ValueError(f"Error with the step: {step}. It should be a positive value and a perfect divider of the slope_size.")
        if zita < 0 and zita > 1:
            raise ValueError(f"Error with the zita: {zita}. It should be a value between 0 and 1.")
        
        self.slope_size = slope_size
        self.step = step
        self.zita = zita
        
        self.slope_values = np.arange(self.slope_size + 1, step=self.step)
        self.n_slopes = self.slope_values.shape[0]

        slopes_args = ['precomputed', 'function']
        if slopes in slopes_args:
            self.add_slopes_mode = slopes
            if slopes == 'precomputed':
                if slope_size > 0:
                    self.pos_slopes = self._precompute_slopes()
                    self.neg_slopes = self.pos_slopes[:, ::-1]
                else:
                    self.pos_slopes = np.array([])
                    self.neg_slopes = np.array([])
        else:
            raise ValueError(f"Unknown argument for slopes: {slopes}. 'slopes' should be one of {existence_args}")
        
        existence_args = [None, 'trivial', 'optimized', 'matrix']
        if existence in existence_args:
            self.existence_mode = existence
        else:
            raise ValueError(f"Unknown argument for existence: {existence}. 'existence' should be one of {existence_args}")
        
        conf_matrix_args = ['trivial', 'dynamic', 'dynamic_plus']
        if conf_matrix in conf_matrix_args:
            self.conf_matrix_mode = conf_matrix
        else:
            raise ValueError(f"Unknown argument for conf_matrix: {conf_matrix}. 'conf_matrix' should be one of {conf_matrix_args}")
        
        interpolation_args = ['linear', 'stepwise']
        if interpolation in interpolation_args:
            self.interpolation_mode = interpolation
        else:
            raise ValueError(f"Unknown argument for interpolation: {interpolation}. 'interpolation' should be one of {interpolation_args}")
        
        metric_args = ['vus_pr', 'vus_roc', 'all']
        if metric in metric_args:
            self.metric = metric
        else:
            raise ValueError(f"Unknown argument for metric: {metric}. 'metric' should be one of {metric_args}")

    @time_it
    def compute(self, label, score, return_vus_roc=False):
        """
        The main computing function of the metric
        """
        # TODO: Compute anomaly indexes and position here once, and then feed it to the functions bellow
        # TODO: Maybe the last 3 steps can be combined into one function that efficiently handles it

        thresholds = self.get_unique_thresholds(score)

        sm = self.get_score_mask(score, thresholds)

        labels = self.add_slopes(label)

        existence = self.compute_existence(labels, sm, score, thresholds)
        
        fp, fn, tp, positives, negatives, fpr, tpr = self.compute_confusion_matrix(labels, sm)

        precision, recall = self.precision_recall_curve(tp, fp, positives, existence)
        
        if self.metric == 'vus_pr':
            return self.auc(recall, precision).mean()
        elif self.metric == 'vus_roc':
            return self.auc(fpr, tpr).mean()
        else:
            return self.auc(recall, precision).mean(), self.auc(fpr, tpr).mean()

    def get_score_mask(self, score, thresholds):
        return score >= thresholds[:, None]
    
    def get_unique_thresholds(self, score):
        return np.sort(np.unique(score))[::-1]
    
    def _create_safe_mask(self, label):
        """
        A safe mask is a mask of the label that every anomaly is one point bigger (left and right)
        than the bigger slope. This allows us to mask the label safely, without changing anything in the implementation.
        """

        pos = self._distance_from_anomaly(label)
        mask = pos <= (self.slope_size + 1)
        return np.logical_or(mask, label)
    
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
    
    def _precompute_slopes(self):
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
    
    def _distance_from_anomaly(self, label, clip=False):
        '''
        For every point in the label, returns a time series that shows the distance
        of that point to its closest anomaly. Compute the distance of each point 
        to every anomaly and keep the minimum.
        '''
        length = len(label)
        start_points, end_points = self.get_anomalies_coordinates(label)
        indices = np.arange(length)[:, None]

        anomaly_boundaries = np.concat([start_points, end_points])
        if len(anomaly_boundaries) == 0:
            return np.ones(length) * np.inf
        
        distances = np.abs(indices - anomaly_boundaries)
        pos = np.min(distances, axis=1)
        if clip:
            pos[label.astype(bool)] = 0
            pos = np.clip(pos, 0, self.slope_size)
        
        return pos
    
    def get_anomalies_coordinates(self, label, include_edges=True):
        '''
        Return the starting and ending points of all anomalies in label
        If edges is True, then return the first and last point,
        only if anomalies exist in the very beggining or very end.
        '''
        diff = np.diff(label)
        start_points = np.where(diff == 1)[0] + 1
        end_points = np.where(diff == -1)[0]

        if include_edges:
            if label[-1]:
                end_points = np.append(end_points, len(label) - 1)
            if label[0]:
                start_points = np.append([0], start_points)
            if start_points.shape != end_points.shape:
                raise ValueError(f'The number of start and end points of anomalies does not match, {start_points} != {end_points}')
            
        return start_points, end_points
    
    def harmonic_series(self, n):
        """
        Approximate the nth harmonic number H_n using the Euler-Mascheroni approximation.
        
        Args:
            n (int): The number of terms in the harmonic series.

        Returns:
            float: Approximated value of the harmonic series.
        """
        return digamma(n + 1) + np.euler_gamma

    def add_slope_approximate(self, label):
        """
        This implementation of approximating the mean slope fast is not correct yet.
        Even, the speed up is 2-3 times faster for CPUs so maybe it's not worth it.
        
        There is no (up to now) easy/fast way to approximate the mean of the slopes that is 
        significantly faster than just actually computing them.
        """
        pos = self._distance_from_anomaly(label)
        mask = np.where(np.logical_and((pos < self.n_slopes), (pos > 0))) 
        h_l = self.harmonic_series(self.slope_size)
        # mean_slope = np.mean(self.neg_slopes, axis=0)

        slope = np.zeros(label.shape)
        slope[mask] = (self.slope_size - pos[mask] + 1) * self.zita + (1 - self.zita) * (h_l - self.harmonic_series(pos[mask] - 1))
        slope /= self.n_slopes
        # slope[(self.slope_size - pos) < 0] = 0
        slope[np.where(label.astype(bool))] = 1

        return slope

    def _slope_function(self, label, pos):
        """
        Computes slope transformations for multiple l values efficiently.
        
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
    
    def add_slopes_function(self, label):
        """
        Compute the slopes of a label using a predefined function.

        This function is about 5 - 10 times slower than the precomputed version on CPUs.
        The pos part requires about 10% of the total execution time.
        """

        pos = self._distance_from_anomaly(label)
        slopes = self._slope_function(label, pos)
        
        return slopes

    def add_slopes_precomputed(self, label):
        start_points, end_points = self.get_anomalies_coordinates(label, include_edges=False)

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
    
    def add_slopes(self, label):
        if self.add_slopes_mode == 'precomputed':
            return self.add_slopes_precomputed(label)
        else:
            return self.add_slopes_function(label)
    
    def compute_existence(self, labels, sm, score, thresholds):
        if self.existence_mode == 'optimized':
            return self.existence_optimized(labels, score, thresholds)
        if self.existence_mode == 'matrix':
            return self.existence_matrix(labels, sm)
        if self.existence_mode is None:
            # return self.no_existence(thresholds)
            return None
        if self.existence_mode == 'trivial':
            return self.existence_trivial(labels, sm, thresholds)
    
    def no_existence(self, thresholds):
        return np.ones((self.n_slopes, len(thresholds)))

    def existence_trivial(self, labels, score_mask, thresholds):
        n_slopes, T = labels.shape
        n_thresholds = thresholds.shape[0]
        existence = np.zeros((n_slopes, n_thresholds))
        
        start_points, end_points = self.get_anomalies_coordinates(labels[0])
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

    def existence_optimized(self, labels, score, thresholds):
        """
        Optimized existence computation (~400x faster than trivial implementation).
        Suitable for CPU applications.
        """
        n_slopes, T = labels.shape
        n_thresholds = thresholds.shape[0]
        existence = np.zeros((n_slopes, n_thresholds))
        thresholds_dict = {value: index for index, value in enumerate(thresholds)}

        start_points, end_points = self.get_anomalies_coordinates(labels[0])
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

    def existence_matrix(self, labels, score_mask):
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
        mask = self._create_safe_mask(labels[0])
        labels = labels[:, mask]
        score_mask = score_mask[:, mask]

        # Normalize labels to binary (0 or 1)
        norm_labels = (labels > 0).astype(np.int8)
        score_mask = score_mask

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

        fp = sm.sum(axis=1) - tp
        negatives = labels[0].shape[0] - positives
        fpr = fp / negatives
        tpr = tp / positives

        return fp, fn, tp, positives, negatives, fpr, tpr

    def conf_matrix_trivial(self, labels, sm):
        label = labels[0]
        sm_inv = ~sm

        true_positives = np.matmul(labels, sm.T)
        false_negatives = np.matmul(label, sm_inv.T)[:, np.newaxis].T

        positives = ((true_positives + false_negatives) + label.sum()) / 2

        return false_negatives, true_positives, positives
    
    def conf_matrix_dyn(self, labels, sm):
        mask = np.where(labels[-1] > 0)[0]
        masked_labels = labels[:, mask]
        masked_sm = sm[:, mask]
        masked_sm_inv = ~masked_sm

        true_positives = np.matmul(masked_labels, masked_sm.T)
        false_negatives = np.matmul(masked_labels[0], masked_sm_inv.T)[:, np.newaxis].T
        

        positives = ((true_positives + false_negatives) + masked_labels[0].sum()) / 2

        return false_negatives, true_positives, positives
    
    def conf_matrix_dyn_plus(self, labels, sm):
        label = labels[0]
        sm_inv = ~sm
        
        slope_mask = np.where(np.logical_and(labels[-1] > 0, labels[-1] < 1))[0]
        label_as_mask = np.where(label)[0]
        
        initial_tps = sm[:, label_as_mask].sum(axis=1)
        slope_tps = np.matmul(labels[:, slope_mask], sm[:, slope_mask].T)
        true_positives = initial_tps + slope_tps

        false_negatives = sm_inv[:, label_as_mask].sum(axis=1)

        positives = ((true_positives + false_negatives) + label_as_mask.shape[0]) / 2

        return false_negatives, true_positives, positives
    
    def precision_recall_curve(self, tp, fp, positives, existence):
        ones, zeros = np.ones(self.n_slopes)[:, np.newaxis], np.zeros(self.n_slopes)[:, np.newaxis]
        precision = np.hstack((ones, (tp / (tp + fp))))
        recall = np.hstack((zeros, (tp / positives)))
        recall[recall > 1] = 1
        if existence is not None:
            recall[:, 1:] = np.multiply(recall[:, 1:], existence)

        return precision, recall
    
    def auc(self, x, y):
        if self.interpolation_mode == 'linear':
            return self.linear_interpolation(x, y)
        else:
            return self.stepwise_interpolation(x, y)    

    def linear_interpolation(self, x, y):
        return np.trapezoid(y, x, axis=1)

    def stepwise_interpolation(self, x, y):
        width_pr = x[:, 1:] - x[:, :-1]
        height_pr = y[:, 1:]
        return np.sum(np.multiply(width_pr, height_pr), axis=1)
    
