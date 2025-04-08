"""
@who: Emmanouil Sylligardos
@where: Ecole Normale Superieur (ENS), Paris, France
@when: PhD Candidate, 1st year (2024)
@what: MSAD-E
"""

from templates import sh_templates

import itertools
import os
from datetime import datetime
from copy import deepcopy
import re


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def create_shell_scripts():
    parent_dir = "scripts"

    date_str = datetime.now().strftime("%d_%m_%Y")
    logs_saving_dir = f"experiments/{date_str}"
    
    experiment_desc = {
        "job_name": "synthetic",
        "environment": "ffvus",
        "script_name": "src/generate_synthetic_dataset.py",
        "args": {
            "n_timeseries": [10],
            "ts_length": [1_000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000],
            "n_anomalies": [3, 5, 10, 50, 100],
            "avg_anomaly_length": [1, 10, 100, 1_000, 10_000],
        },
        "rules": ["True if n_anomalies * avg_anomaly_length <= 0.2 * ts_length else False"]
    }
    template = sh_templates['cleps_cpu']
    
    # Analyse json
    saving_dir = os.path.join(parent_dir, experiment_desc['job_name'])
    environment = experiment_desc["environment"]
    script_name = experiment_desc["script_name"]
    args = experiment_desc["args"]
    arg_names = list(args.keys())
    arg_values = list(args.values())
    rules = experiment_desc['rules']
    cmd = f"{script_name}"
    job_name = experiment_desc["job_name"]

    # Generate all possible combinations of arguments
    combinations = list(itertools.product(*arg_values))
    
    # Create the commands
    jobs = set()
    for combination in combinations:
        curr_cmd = cmd
        curr_job_name = job_name
        curr_rules = deepcopy(rules)

        for name, value in zip(arg_names, combination):
            curr_cmd += f" --{name} {value}"
            curr_job_name += f"_{value}"
            for i in range(len(curr_rules)):
                curr_rules[i] = curr_rules[i].replace(name, str(value))
        
        # Evaluate rules to see if we accept this combination, if not skip
        rules_evaluation = [eval(rule) for rule in curr_rules]
        if not all(rules_evaluation):
            continue

        # Create the saving dir for the scripts if it doesn't exist
        if not os.path.exists(saving_dir):
            os.makedirs(saving_dir)
            
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
    
    with open(os.path.join(saving_dir, f'conduct_{experiment_desc["job_name"]}.sh'), 'w') as rsh:
        rsh.write(run_all_sh)
        

if __name__ == "__main__":
    create_shell_scripts()