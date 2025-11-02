"""Configs for model sweeps"""

# <codecell>
import itertools
import sys
sys.path.append('../../../../')
from task.prop import full_text_ds_path, or_text_ds_path, imply_text_ds_path, php_text_ds_path

model_names = [
    'Qwen/Qwen2.5-Coder-14B',
    'Qwen/Qwen2.5-Coder-7B',
    # 'Qwen/Qwen2.5-Coder-3B',
    # 'Qwen/Qwen2.5-Coder-1.5B',
    'Qwen/Qwen2.5-Coder-0.5B',
    # 'Qwen/Qwen2.5-0.5B',
]

prompt_styles = [
    'ar_cot',
    # 'dp',
    # 'dp_full'
]

tasks = [
    'full',
    'or',
    # 'imply'
]

task_to_ds_path = {
    'full': full_text_ds_path,
    'or': or_text_ds_path,
    'imply': imply_text_ds_path
}

# Corresponds roughly to 50,90 percentiles
task_to_splits = {
    'full': [4, 8],
    'or': [6, 12],
    'imply': [4, 10]
}


configs = []
for model_name, task, prompt in itertools.product(model_names, tasks, prompt_styles):
    batch_size = 32
    if '14B' in model_name:
        batch_size = 4
    elif '7B' in model_name:
        batch_size = 16
    
    accum_steps = 128 // batch_size

    if prompt == 'dp':
        accum_steps = 32 // batch_size

    base_config = {
        'model_name': model_name,
        'batch_size': batch_size,
        'accum_steps': accum_steps,
        'num_samples': 50,
        'max_length': 2048,
        'ds_path': task_to_ds_path[task],
        'splits': task_to_splits[task],
        'prompt': prompt,
        'project_name': f'big_prop_{task}',
        'run_name_prefix': f"big_batch-{model_name.split('/')[-1]}-{prompt}"
    }

    for split in base_config['splits']:
        curr = base_config.copy()
        curr['train_split'] = split
        curr['run_name'] = f"{base_config['run_name_prefix']} (split={split})"
        curr['output_dir'] = f"~/scratch/ckpt/{curr['project_name']}/{curr['run_name'].replace(' ', '_')}"
        configs.append(curr)

len(configs)