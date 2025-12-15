"""Configs for model sweeps"""

# <codecell>
import itertools
import sys
sys.path.append('../../../../')
from task.prop import or_text_ds_path

itr_ids = list(range(3))

base_len = 1024

model_sets = [
    # {'name': 'Qwen/Qwen2.5-Coder-0.5B', '2k_bs': 32, '32k_bs': 1},
    # {'name': 'Qwen/Qwen2.5-Coder-1.5B', '2k_bs': 32, '32k_bs': 1},
    # {'name': 'Qwen/Qwen2.5-Coder-7B', '2k_bs': 32, '32k_bs': 1},
    {'name': 'Qwen/Qwen2.5-Coder-32B', '2k_bs': 16, '32k_bs': 1},
]

prompt_styles = [
    'ar_cot',
    'dp',
    # 'dp_full'
]

tasks = [
    {'name': 'or', 'train_len': 8192, 'test_len': 32_768},
]

task_to_ds_path = {
    'or': or_text_ds_path,
}

# Corresponds roughly to 50, 90th percentiles
task_to_splits = {
    'or': [18],
}

task_to_test_splits = {
    'or': [18, 30, 45],
}


configs = []

for i in itr_ids:
    for model_set, task_args, prompt in itertools.product(model_sets, tasks, prompt_styles):
        total_batch_size = 128
        model_name = model_set['name']
        task = task_args['name']

        if task != 'php':
            batch_size = model_set['2k_bs']
        else:
            batch_size = model_set['32k_bs']

        len_scale_fac = base_len / task_args['train_len']
        batch_size = int(batch_size * len_scale_fac)

        accum_steps = total_batch_size // batch_size
        if accum_steps < 1:
            accum_steps = 1
            batch_size = total_batch_size
        

        base_config = {
            'model_name': model_name,
            'task_name': task,
            'batch_size': batch_size,
            'accum_steps': accum_steps,
            'max_steps': 5_000,
            'num_samples': 16,
            'max_length': task_args['train_len'],
            'max_test_length': task_args['test_len'],
            'log_every': 250 if prompt == 'dp' else 999_999_999,
            'save_every': 250,
            'ds_path': task_to_ds_path[task],
            'splits': task_to_splits[task],
            'test_splits': task_to_test_splits[task],
            'prompt': prompt,
            'project_name': f'prop_{task}',
            'run_name_prefix': f"{model_name.split('/')[-1]}-{prompt}",
            'packing': False
        }

        for split in base_config['splits']:
            curr = base_config.copy()
            curr['train_split'] = split
            curr['run_name'] = f"{base_config['run_name_prefix']} (split={split}, itr={i})"
            curr['output_dir'] = f"~/scratch/ckpt/{curr['project_name']}/{curr['run_name'].replace(' ', '_')}"
            configs.append(curr)

len(configs)