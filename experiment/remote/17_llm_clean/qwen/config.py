"""Configs for model sweeps"""

# <codecell>
import itertools
import sys

sys.path.append('../')
from run import perform_run

data_dir = '/n/netscratch/pehlevan_lab/Lab/wlt/data/prop_gen/data'

perform = perform_run
if sys.argv[-1] == 'eval':
    from eval import perform_eval
    perform = perform_eval
else:
    if sys.argv[-1] != 'run':
        print(f"unrecognized mode: {sys.argv[-1]}, defaulting to run")


# itr_ids = list(range(5))
itr_ids = list(range(5, 10))
base_len = 1024

model_sets = [
    {'name': 'Qwen/Qwen2.5-Coder-0.5B', '2k_bs': 64, '32k_bs': 1},
    {'name': 'Qwen/Qwen2.5-Coder-1.5B', '2k_bs': 64, '32k_bs': 1},
    {'name': 'Qwen/Qwen2.5-Coder-3B', '2k_bs': 64, '32k_bs': 1},
    {'name': 'Qwen/Qwen2.5-Coder-7B', '2k_bs': 32, '32k_bs': 1},
    {'name': 'Qwen/Qwen2.5-Coder-14B', '2k_bs': 32, '32k_bs': 1},
    {'name': 'Qwen/Qwen2.5-Coder-32B', '2k_bs': 32, '32k_bs': 1},
]

prompt_styles = [
    'ar_cot',
    'dp',
]

tasks = [
    {'name': 'full', 'train_len': 512, 'test_len': 2048},
    {'name': 'imply', 'train_len': 1024, 'test_len': 16_384},
]

task_to_ds_path = {
    'full': data_dir + '/hf_full_text_pipe',
    'imply': data_dir + '/hf_implies_text_pipe',
}

# Corresponds roughly to 50, 90th percentiles
task_to_splits = {
    'full': [4],
    'imply': [6],
}

task_to_test_splits = {
    'full': [4],
    'imply': [6],
}


configs = []

for i in itr_ids:
    for model_set, task_args, prompt in itertools.product(model_sets, tasks, prompt_styles):
        total_batch_size = 32
        model_name = model_set['name']
        task = task_args['name']

        batch_size = model_set['2k_bs']

        len_scale_fac = base_len / task_args['train_len']
        batch_size = min(int(batch_size * len_scale_fac), total_batch_size)

        accum_steps = total_batch_size // batch_size
        if accum_steps < 1:
            accum_steps = 1
            batch_size = total_batch_size
        

        base_config = {
            'model_name': model_name,
            'task_name': task,
            'batch_size': batch_size,
            'accum_steps': accum_steps,
            'max_steps': 2_500,
            'num_samples': 16,
            'max_length': task_args['train_len'],
            'max_test_length': task_args['test_len'],
            'log_every': 250 if prompt == 'dp' else 999_999_999,
            'save_every': 250,
            'ds_path': task_to_ds_path[task],
            'splits': task_to_splits[task],
            'test_splits': task_to_test_splits[task],
            'prompt': prompt,
            'project_name': f'qwen_{task}',
            'run_name_prefix': f"{model_name.split('/')[-1]}-{prompt}",
            'packing': False
        }

        for split in base_config['splits']:
            curr = base_config.copy()
            curr['train_split'] = split
            curr['run_name'] = f"{base_config['run_name_prefix']} (split={split}, itr={i})"
            curr['output_dir'] = f"~/scratch/ckpt/final/{curr['project_name']}/{curr['run_name'].replace(' ', '_')}"
            configs.append(curr)

len(configs)
# <codecell>

perform(configs)
