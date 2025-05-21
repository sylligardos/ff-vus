"""
@who: Emmanouil Sylligardos
@where: Ecole Normale Superieur (ENS), Paris, France
@when: PhD Candidate, 2nd year (2025)
@what: FF-VUS
"""


import torch
from utils.utils import time_it
import matplotlib.pyplot as plt
import seaborn as sns
import time
from tqdm import tqdm

class VUSTorch():
    def __init__(
            self, 
            slope_size=100, 
            step=1, 
            zita=(1/torch.sqrt(torch.tensor(2))), 
            conf_matrix='dynamic',
            device=None,
            max_memory_tokens=None,
        ):
        """
        Initialize the torch version of the VUS metric.

        Args:
            TODO: Write the arguments description when done
        """
        if slope_size < 0 or step < 0:
            raise ValueError(f"Error with the slope_size {slope_size} or step {step}. They should be positive values.")
        if step == 0 and slope_size > 0:
            raise ValueError(f"Error with step {step} and slope_size {slope_size}. Step can't be 0 with non-zero slope_size.")
        if (step > 0 and slope_size > 0) and (slope_size % step) != 0:
            raise ValueError(f"Error with the step: {step}. It should be a positive value and a perfect divider of the slope_size.")
        if zita < 0 and zita > 1:
            raise ValueError(f"Error with the zita: {zita}. It should be a value between 0 and 1.")
        
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        if self.device == 'cpu':
            print('You are using the GPU version of VUS on a CPU. If this is intended you can try the lighter numpy version!')

        self.slope_size = slope_size
        self.step = step
        self.zita = zita
        
        self.slope_values = torch.arange(start=0, end=self.slope_size + 1, step=self.step, device=self.device) if step > 0 else torch.tensor([0])
        self.n_slopes = self.slope_values.shape[0]

        conf_matrix_args = ['dynamic', 'dynamic_plus']
        if conf_matrix in conf_matrix_args:
            self.conf_matrix_mode = conf_matrix
        else:
            raise ValueError(f"Unknown argument for conf_matrix: {conf_matrix}. 'conf_matrix' should be one of {conf_matrix_args}")

        if max_memory_tokens is None:
            if self.device == 'cuda':
                available_memory, total_memory = torch.cuda.mem_get_info()
                self.max_memory_tokens = available_memory / 50
            else:
                self.max_memory_tokens = 100e+9
        else:
            self.max_memory_tokens = max_memory_tokens


    def compute(self, label, score):
        """
        The main computing function of the metric
        
        TODO: Try smaller types when possible, e.g. thresholds. Is it worth it?
        TODO: I think I can do pos in the for loop, so try to remove pos from find_safe_splits
        TODO: Then remove the break into chunks functionality from distance from anomaly
        """
        tic = time.time()

        # Initialization
        ((_, _), (start_with_edges, end_with_edges)), time_anomalies_coord = time_it(self.get_anomalies_coordinates_both)(label)

        sloped_label_mem_size = self.n_slopes * len(label) * 4
        if sloped_label_mem_size > self.max_memory_tokens:
            n_splits = int(sloped_label_mem_size // self.max_memory_tokens)
            safe_mask = self.create_safe_mask(label, start_with_edges, end_with_edges)
            split_points = self.find_safe_splits(label, safe_mask, n_splits)

            # Total values holders
            fp, tp, positives = 0, 0, 0
            time_sm = 0
            time_slopes = 0
            time_existence = 0
            time_confusion = 0
            anomalies_found = 0
            total_anomalies = 0

            prev_split = 0
            for curr_split in tqdm(split_points, desc='In distance from anomaly', disable=False if n_splits > 10000 else True):
                label_c = label[prev_split:curr_split]
                score_c = score[prev_split:curr_split]
                pos_c = pos[prev_split:curr_split]
                prev_split = curr_split

                # Preprocessing
                pos, time_pos = time_it(self.distance_from_anomaly)(label_c, start_with_edges, end_with_edges)
                (thresholds, _), time_thresholds = time_it(self.get_unique_thresholds)(score_c)
                sm, chunk_time_sm = time_it(self.get_score_mask)(score_c, thresholds)

                labels_c, chunk_time_slope = time_it(self.add_slopes)(label_c, pos_c)
                (anomalies_found_c, total_anomalies_c), chunk_time_existence = time_it(self.compute_existence)(labels_c, sm, pos_c, normalize=False)
                (fp_c, fn_c, tp_c, pos_c, neg_c, fpr_c), chunk_time_conf = time_it(self.compute_confusion_matrix)(labels_c, sm)

                tp += tp_c
                fp += fp_c
                positives += pos_c
                anomalies_found += anomalies_found_c
                total_anomalies += total_anomalies_c

                time_sm += chunk_time_sm
                time_slopes += chunk_time_slope
                time_existence += chunk_time_existence
                time_confusion += chunk_time_conf

            # Combine all chunks
            existence = anomalies_found / total_anomalies
        else:
            pos, time_pos = time_it(self.distance_from_anomaly)(label, start_with_edges, end_with_edges)
            (thresholds, _), time_thresholds = time_it(self.get_unique_thresholds)(score)
            sm, time_sm = time_it(self.get_score_mask)(score, thresholds)
        
            labels, time_slopes = time_it(self.add_slopes)(label, pos)    
            existence, time_existence = time_it(self.compute_existence)(labels, sm, pos)
            (fp, fn, tp, positives, negatives, fpr), time_confusion = time_it(self.compute_confusion_matrix)(labels, sm)
        
        (precision, recall), time_pr_rec = time_it(self.precision_recall_curve)(tp, fp, positives, existence)
        vus_pr, time_integral = time_it(self.auc)(recall, precision)
        toc = time.time()

        time_analysis = {
            "Total time": toc - tic,
            "Anomalies coordinates time": time_anomalies_coord,
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

    def find_safe_splits(self, label, safe_mask, n_splits):
        """
        Finds approximately evenly spaced safe split points where pos is a local maximum
        and far enough from anomalies.
        """
        length = torch.tensor(label.shape[0], device=self.device)
        valid_splits_mask = ~safe_mask
        valid_splits = torch.nonzero(valid_splits_mask).squeeze(1)

        if valid_splits.numel() == 0 or n_splits <= 1:
            return torch.tensor([], device=self.device)

        # print(length, n_splits)
        ideal_splits = torch.linspace(0, length - 1, steps=n_splits + 1, device=self.device)[1:-1]        

        dists = torch.abs(valid_splits[:, None] - ideal_splits[None, :])
        min_indices = dists.argmin(dim=0)
        selected_splits = valid_splits[min_indices]
        selected_splits = torch.unique(selected_splits.sort().values, sorted=True)

        # Final safety check
        assert torch.all(label[selected_splits] != 1), "Some selected splits fall inside anomalies!"
        assert torch.all(safe_mask[selected_splits] == 0), "Some splits are too close to anomalies!"

        return torch.cat((selected_splits, length[None]), dim=0)
    
    def get_score_mask(self, score, thresholds):
        return score >= thresholds[:, None]
    
    def get_unique_thresholds(self, score):
        return torch.sort(torch.unique(score), descending=True)
    
    def get_anomalies_coordinates_both(self, label: torch.Tensor):
        """
        Return the starting and ending points of all anomalies in label,
        both with and without edge inclusion.

        Args:
            label (torch.tensor): vector of 1s and 0s
        Returns:
            ((start_no_edges, end_no_edges), (start_with_edges, end_with_edges))
        """
        device = label.device
        diff = torch.diff(label)
        start_no_edges = torch.where(diff == 1)[0] + 1
        end_no_edges = torch.where(diff == -1)[0]
        
        if start_no_edges.shape != end_no_edges.shape:
            raise ValueError(f"The number of start and end points of anomalies does not match: {start_with_edges} != {end_with_edges}")

        start_with_edges = torch.cat((torch.tensor([0], device=device), start_no_edges)) if label[0] else start_no_edges
        end_with_edges = torch.cat((end_no_edges, torch.tensor([len(label) - 1], device=device))) if label[-1] else end_no_edges

        return (start_no_edges, end_no_edges), (start_with_edges, end_with_edges)
    
    def create_safe_mask(self, label: torch.Tensor, start_points: torch.Tensor, end_points: torch.Tensor) -> torch.Tensor:
        """
        A safe mask is a mask of the label that every anomaly is one point bigger (left and right)
        than the bigger slope. This allows us to mask the label safely, without changing anything in the implementation.
        """
        length = torch.tensor(label.shape[0])
        mask = torch.zeros(length, dtype=torch.int8, device=self.device)
        
        start_safe_points = torch.maximum(start_points - (self.slope_size + 1), mask[0])
        end_safe_points = torch.minimum(end_points + (self.slope_size + 1) + 1, length - 1)

        mask[start_safe_points] += 1
        mask[end_safe_points] -= 1

        mask = mask.cumsum(dim=0)
        mask = mask > 0
        mask[-1] = label[-1]
        
        return mask
    
    def distance_from_anomaly(self, label: torch.Tensor, start_points: torch.Tensor, end_points: torch.Tensor, clip: bool = False) -> torch.Tensor:
        """
        Computes distance to closest anomaly boundary for each point. Uses PyTorch for GPU compatibility.
        """
        device = label.device
        length = label.size(0)

        pos = torch.full((length,), float('inf'), device=device)
        if start_points.numel() == 0 and end_points.numel() == 0:
            return pos

        anomaly_boundaries = torch.cat([start_points, end_points])
        indices = torch.arange(length, device=device)[:, None]

        # pos_mem_size = len(anomaly_boundaries) * length * pos.element_size()
        # if pos_mem_size > self.max_memory_tokens:
        #     n_chunks = max(1, pos_mem_size // self.max_memory_tokens)
        #     chunk_size = int(max(1, len(anomaly_boundaries) // n_chunks))

        #     for chunk in tqdm(anomaly_boundaries.split(chunk_size), desc='In distance from anomaly', disable=False if n_chunks > 10000 else True):
        #         curr_distances = torch.abs(indices - chunk[None, :])
        #         curr_min_distances = curr_distances.min(dim=1).values
        #         pos = torch.minimum(pos, curr_min_distances)
        # else:
        distances = torch.abs(indices - anomaly_boundaries)
        pos = torch.min(distances, dim=1).values
        
        if clip:
            pos = pos.clone()
            pos[label.bool()] = 0
            pos = torch.clamp(pos, max=self.slope_size)

        return pos
    
    def add_slopes(self, label: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """
        Computes slope transformations for multiple slope values.

        Args:
            label: Binary tensor indicating anomaly regions (shape: [T], dtype: float).
            pos  : Distance to the nearest anomaly boundary (shape: [T]).

        Returns:
            Tensor of shape (n_slopes, T) with transformed slope values.
        """

        device = label.device
        T = label.size(0)

        valid_mask = (self.slope_size - pos) >= 0
        slope_values = self.slope_values[1:, None]
        pos = pos[None, :]
        f_pos = torch.zeros((self.n_slopes, T), device=device)
        f_pos[0] = label

        numerator = (1 - self.zita) * pos
        f_pos[1:, valid_mask] = 1 - (numerator[:, valid_mask] / slope_values)
        f_pos[1:][(slope_values - pos) < 0] = 0
        f_pos[1:, label.bool()] = 1

        return f_pos

    def compute_existence(self, labels: torch.Tensor, score_mask: torch.Tensor, safe_mask: torch.Tensor = None, normalize: bool = True, allow_recursion: bool = True) -> torch.Tensor:
        """
        PyTorch version of the existence matrix computation. This function uses an approximation fallback mechanism
        in case the memory requirements exceed the limit set by 'max_memory_tokens'.

        Args:
            labels: Tensor of shape [s, T], binary anomaly label one for each slope.
            score_mask: Tensor of shape [t, T], score mask over every threshold.

        Returns:
            Tensor of shape [s, t] representing existence score (fraction of anomalies found).
        """
        device = labels.device

        # Compute mask for relevant points
        if safe_mask is not None:
            labels = labels[:, safe_mask]
            score_mask = score_mask[:, safe_mask]
        
        if labels.shape[1] < 1:
            return 0 if normalize else (0, 0)

        # Check size and fallback if too big
        s, T = labels.shape
        t = score_mask.shape[0]
        if allow_recursion and (s * t * T > self.max_memory_tokens):
            n_anomalies_found_0, total_anomalies = self.compute_existence(labels[0:1], score_mask, normalize=False, allow_recursion=False)
            n_anomalies_found_s, total_anomalies = self.compute_existence(labels[-1:], score_mask, normalize=False, allow_recursion=False)
            
            interp_weights = torch.linspace(0, 1, steps=s, device=device).view(1, -1)
            n_anomalies_found = n_anomalies_found_0 + (n_anomalies_found_s - n_anomalies_found_0) * interp_weights.T
        else:
            # Normalize labels to binary (0 or 1)
            norm_labels = (labels > 0).int()

            # Compute step function (stairs)
            diff = torch.diff(norm_labels, dim=1, prepend=torch.zeros((norm_labels.size(0), 1), device=device))
            diff = torch.clamp(diff, min=0, max=1)
            stairs = torch.cumsum(diff, dim=1)
            labels_stairs = norm_labels * stairs

            # Multiply each score with every labeled anomaly step
            score_hat = labels_stairs[:, None, :] * score_mask[None, :, :]

            # Cumulative max along the time axis
            cm = torch.cummax(score_hat, dim=2).values  # shape: [B, S, T']

            # Compute differences along time and normalize
            cm_diff = torch.diff(cm, dim=2)
            cm_diff_norm = torch.clamp(cm_diff - 1, min=0)

            # Total anomalies and missed anomalies
            total_anomalies = stairs[:, -1][:, None]
            final_anomalies_missed = total_anomalies - cm[:, :, -1]
            n_anomalies_not_found = torch.sum(cm_diff_norm, dim=2) + final_anomalies_missed
            n_anomalies_found = total_anomalies - n_anomalies_not_found

        if normalize:
            return n_anomalies_found / total_anomalies
        else:
            return n_anomalies_found, total_anomalies

    
    def compute_confusion_matrix(self, labels: torch.Tensor, sm: torch.Tensor):
        if self.conf_matrix_mode == "dynamic":
            conf_matrix = self.conf_matrix_dyn(labels, sm)
        else:
            conf_matrix = self.conf_matrix_dyn_plus(labels, sm)
        fn, tp, positives = conf_matrix

        # conf_matrix = self.conf_matrix_dyn(labels, sm)
        # conf_matrix_1 = self.conf_matrix_dyn_plus(labels, sm)
        # fn, tp, positives = conf_matrix
        # fn_1, tp_1, positives_1 = conf_matrix_1
        # print(torch.where(tp != tp_1))
        # # print(torch.mean(tp - tp_1), torch.max(tp - tp_1), torch.all(tp == tp_1))
        # # print(torch.mean(fn - fn_1), torch.max(fn - fn_1), torch.all(fn == fn_1))
        # # print(torch.mean(positives - positives_1), torch.max(positives - positives_1), torch.all(positives == positives_1))
        # exit()

        fp = sm.sum(axis=1) - tp
        negatives = labels[0].shape[0] - positives
        fpr = fp / negatives

        return fp, fn, tp, positives, negatives, fpr
    

    def conf_matrix_dyn(self, labels: torch.Tensor, sm: torch.Tensor):
        """
        Computes confusion matrix using only the dynamic (non-always-one) region of the labels.
        
        Args:
            labels: Tensor of shape [2, T], where labels[0] is binary and labels[1] encodes slope info.
            sm: Tensor of shape [S, T], where S is the number of slopes (score masks).
        
        Returns:
            false_negatives: Tensor [S]
            true_positives: Tensor [S]
            positives: Tensor [S]
        """
        mask = torch.where(labels[-1] > 0)[0]
        
        masked_labels = labels[:, mask]
        masked_sm = sm[:, mask]
        masked_sm_inv = ~masked_sm
        
        true_positives = torch.matmul(masked_labels, masked_sm.float().T)
        false_negatives = torch.matmul(masked_labels[0], masked_sm_inv.float().T)[None, :]

        positives = ((true_positives + false_negatives) + masked_labels[0].sum()).div(2.0)

        return false_negatives, true_positives, positives

    def conf_matrix_dyn_plus(self, labels: torch.Tensor, sm: torch.Tensor):
        """
        Optimized confusion matrix computation that avoids redundant work in always-one regions.
        
        Args:
            labels: Tensor of shape [2, T]
            sm: Tensor of shape [S, T]
        
        Returns:
            false_negatives: Tensor [S]
            true_positives: Tensor [S]
            positives: Tensor [S]
        """
        label = labels[0]
        sm_inv = ~sm

        slope_mask = torch.where(torch.logical_and(labels[-1] > 0, labels[-1] < 1))[0]
        label_as_mask = torch.where(label)[0]

        initial_tps = sm[:, label_as_mask].sum(dim=1)
        slope_tps = torch.matmul(labels[:, slope_mask], sm[:, slope_mask].float().T)
        true_positives = initial_tps + slope_tps

        false_negatives = sm_inv[:, label_as_mask].sum(dim=1)

        positives = ((true_positives + false_negatives) + label_as_mask.size(0)).div(2)

        return false_negatives, true_positives, positives
    
    def precision_recall_curve(self, tp: torch.Tensor, fp: torch.Tensor, positives: torch.Tensor, existence: torch.Tensor = None):
        """
        Computes precision-recall curve points given TP, FP, positives, and optional existence weighting.
        
        Args:
            tp: Tensor of shape [n_slopes]
            fp: Tensor of shape [n_slopes]
            positives: Tensor of shape [n_slopes]
            existence: Optional tensor of shape [n_slopes, N] or None

        Returns:
            precision: Tensor of shape [n_slopes, N+1]
            recall: Tensor of shape [n_slopes, N+1]
        """
        device = tp.device

        ones = torch.ones(self.n_slopes, 1, device=device)
        zeros = torch.zeros(self.n_slopes, 1, device=device)
        precision = torch.cat([ones, tp / (tp + fp)], dim=1)
        recall = torch.cat([zeros, tp / positives], dim=1)

        recall[recall > 1] = 1
        if existence is not None:
            recall[:, 1:] = recall[:, 1:] * existence
            
        return precision, recall

    def auc(self, x, y):
        width_pr = x[:, 1:] - x[:, :-1]
        height_pr = y[:, 1:]
        auc = torch.sum(torch.multiply(width_pr, height_pr), axis=1)
        return auc.mean()