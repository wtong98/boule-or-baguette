"""Configs for model sweeps"""

# <codecell>
import itertools
import sys
sys.path.append('../../../../')
from task.prop import full_text_ds_path, or_text_ds_path, imply_text_ds_path, php_text_ds_path


model_sets = [
    {'name': 'Qwen/Qwen2.5-Coder-0.5B', '2k_bs': 32, '32k_bs': 2},
    {'name': 'Qwen/Qwen2.5-Coder-7B', '2k_bs': 16, '32k_bs': 1},
    {'name': 'Qwen/Qwen2.5-Coder-32B', '2k_bs': 4, '32k_bs': 1},
]

prompt_styles = [
    'ar_cot',
    'dp',
    'dp_full'
]

tasks = [
    # 'full',
    # 'imply',
    # 'or',
    'php'
]

task_to_ds_path = {
    'full': full_text_ds_path,
    'imply': imply_text_ds_path,
    'or': or_text_ds_path,
    'php': php_text_ds_path,
}

# Corresponds roughly to 50,90 percentiles
task_to_splits = {
    'full': [4, 8],
    'or': [6, 12],
    'imply': [4, 10],
    'php': [60, 120]
}


configs = []
for model_set, task, prompt in itertools.product(model_sets, tasks, prompt_styles):
    total_batch_size = 128
    model_name = model_set['name']

    if prompt == 'dp':
        # TODO: need firmer estimate of size difference in packed examples, per dataset
        total_batch_size = 32

    if task != 'php':
        batch_size = model_set['2k_bs']
    else:
        batch_size = model_set['32k_bs']

    accum_steps = total_batch_size // batch_size
    if accum_steps < 1:
        accum_steps = 1
        batch_size = total_batch_size
    

    base_config = {
        'model_name': model_name,
        'task_name': task,
        'batch_size': batch_size,
        'accum_steps': accum_steps,
        'num_samples': 15,
        'max_length': 32768 if task == 'php' else 2048,
        'log_every': 100,
        'save_every': 250,
        'ds_path': task_to_ds_path[task],
        'splits': task_to_splits[task],
        'prompt': prompt,
        'project_name': f'prop_{task}',
        'run_name_prefix': f"{model_name.split('/')[-1]}-{prompt}",
    }

    for split in base_config['splits']:
        curr = base_config.copy()
        curr['train_split'] = split
        curr['run_name'] = f"{base_config['run_name_prefix']} (split={split})"
        curr['output_dir'] = f"~/scratch/ckpt/{curr['project_name']}/{curr['run_name'].replace(' ', '_')}"
        configs.append(curr)

len(configs)