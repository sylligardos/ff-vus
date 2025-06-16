"""
@who: Emmanouil Sylligardos
@where: Ecole Normale Superieur (ENS), Paris, France
@when: PhD Candidate, 1st year (2024)
@what: MSAD-E
"""

from templates import sh_templates

import itertools
import os
from copy import deepcopy
import re
import numpy as np
from tqdm import tqdm


def natural_keys(text):
    def atoi(text):
        return int(text) if text.isdigit() else text
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

experiments = {
    "synthetic_data_generation": {
        "job_name": "synthetic",
        "environment": "ffvus",
        "script_name": "src/generate_synthetic.py",
        "template": 'cleps_cpu',
        "args": {
            "n_timeseries": [10],
            "ts_length": [1_000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000],
            "n_anomalies": [3, 5, 10, 50, 100],
            "avg_anomaly_length": [1, 10, 100, 1_000, 10_000],
        },
        "rules": ["True if n_anomalies * avg_anomaly_length <= 0.2 * ts_length else False"]
    },
    "compute_metric": {
        "job_name": "compute_metric",
        "environment": "ffvus",
        "script_name": "src/compute_metric.py",
        "template": 'cleps_cpu',
        "args": {
            "dataset": ['tsb'], # + os.listdir(os.path.join('data', 'synthetic')),
            "metric": ['ff_vus_pr'], #, 'rf', 'affiliation', 'range_auc_pr', 'auc_pr', 'vus_pr'
            "slope_size": [0, 10, 100, 1000], #[0] + [int(x) for x in 2**np.arange(11)],
            "step":  [1, 10, 100, 1000], #[int(x) for x in 2**np.arange(11)],
            "slopes": ['precomputed'], #, 'function'
            "existence": ['None', 'trivial', 'optimized'], #, 'matrix'
            "conf_matrix": ['trivial', 'dynamic', 'dynamic_plus'],
        },
        "rules": [
            # Rule 1: If metric is not in the three allowed ones, only the first value of all other args is allowed
            # "True if 'metric' in ['ff_vus_pr', 'range_auc_pr', 'vus_pr'] and (slope_size == 0 and step == 1 and 'slopes' == 'precomputed' and 'existence' == 'None' and 'conf_matrix' == 'trivial') else False",
            # Rule 2: If metric is in ['range_auc_pr', 'vus_pr'], allow all slope_size but only first for the rest
            # "True if 'metric' not in ['range_auc_pr', 'vus_pr'] and (step == 1 and 'slopes' == 'precomputed' and 'existence' == 'None' and 'conf_matrix' == 'trivial') else False",
            # Rule 3: step cannot be greater than slope_size
            "True if step <= slope_size or (slope_size == 0 and 'metric' == 'ff_vus_pr' and step == 1) else False"
        ]
    },
    "vus_ffvus_auc_synthetic": {
        "job_name": "vus_ffvus_auc_synthetic",
        "environment": "ffvus",
        "script_name": "src/compute_metric.py",
        "template": 'cleps_cpu',
        "args": {
            "dataset": ['all_synthetic'], # + os.listdir(os.path.join('data', 'synthetic')),
            "metric": ['vus_pr', 'ff_vus_pr', 'auc_pr'], #, 'rf', 'affiliation', 'range_auc_pr', 'auc_pr', 'vus_pr', 'ff_vus_pr_gpu'
            "slope_size": [0, 16, 32, 64, 128, 256],
            "step":  [1],
            "slopes": ['precomputed'], #, 'function'
            "existence": ['optimized'], #, 'matrix'
            "conf_matrix": ['dynamic_plus'],
        },
        "rules": [
            "True if (slope_size == 0 and 'metric' == 'auc_pr') or 'metric' != 'auc_pr' else False"
        ]
    },
    "vus_ffvus_ffvusgpu_tsb": {
        "environment": "ffvus",
        "script_name": "src/compute_metric.py",
        "template": 'cleps_gpu',
        "args": {
            "dataset": ['tsb'], # + os.listdir(os.path.join('data', 'synthetic')),
            "metric": ['ff_vus_pr_gpu'], #, 'rf', 'affiliation', 'range_auc_pr', 'auc_pr', 'vus_pr', 'ff_vus_pr_gpu'
            "slope_size": [-1], # [0, 16, 32, 64, 128, 256],
            "step":  [1],
            # "slopes": ['function'], #, 'function'
            "existence": ['None'], #, 'matrix'
            # "conf_matrix": ['dynamic_plus'],
        },
        "rules": []
    }
}

def create_shell_scripts():
    parent_dir = "scripts"
    experiment_name = "vus_ffvus_auc_0"

    logs_saving_dir = os.path.join("experiments", experiment_name)
    os.makedirs(logs_saving_dir, exist_ok=True)
    experiment_desc = experiments[experiment_name]
    
    # Analyse json
    job_name = experiment_name
    saving_dir = os.path.join(parent_dir, job_name)
    environment = experiment_desc["environment"]
    script_name = experiment_desc["script_name"]
    template = sh_templates[experiment_desc["template"]]
    args = experiment_desc["args"]
    arg_names = list(args.keys())
    arg_values = list(args.values())
    rules = experiment_desc['rules']
    
    jobs = set()
    os.makedirs(saving_dir, exist_ok=True)
    combinations = list(itertools.product(*arg_values))
    for combination in tqdm(combinations):
        curr_cmd = script_name
        curr_job_name = job_name
        curr_rules = deepcopy(rules)

        for name, value in zip(arg_names, combination):
            curr_cmd += f" --{name} {value}"
            curr_job_name += f"_{value}"
            for i in range(len(curr_rules)):
                curr_rules[i] = curr_rules[i].replace(name, str(value))
        
        curr_cmd += f"--experiment {experiment_name}"

        # Evaluate rules to see if we accept this combination, if not skip
        rules_evaluation = [eval(rule) for rule in curr_rules]
        if not all(rules_evaluation):
            continue
            
        # Fill in template and write the .sh file
        with open(os.path.join(saving_dir, f'{curr_job_name}.sh'), 'w') as rsh:
            rsh.write(template.format(curr_job_name, logs_saving_dir, logs_saving_dir, environment, curr_cmd))
        
        jobs.add(curr_job_name)

    # Create sh file to conduct all experiments 
    run_all_sh = ""
    jobs = list(jobs)
    jobs.sort(key=natural_keys)
    for job in jobs:
        run_all_sh += f"sbatch {os.path.join(saving_dir, f'{job}.sh')}\n"
    
    with open(os.path.join(saving_dir, f'conduct_{job_name}.sh'), 'w') as rsh:
        rsh.write(run_all_sh)
        

if __name__ == "__main__":
    create_shell_scripts()
