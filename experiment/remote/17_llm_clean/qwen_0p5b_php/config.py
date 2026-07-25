"""Regenerate the Qwen2.5-Coder-7B PITA checkpoints."""

# <codecell>
import itertools
import sys

sys.path.append('../')
from run import perform_run


data_path = '/n/netscratch/pehlevan_lab/Lab/wlt/data/pita'

perform = perform_run
if sys.argv[-1] == 'eval':
    from eval import perform_eval
    perform = perform_eval
elif sys.argv[-1] != 'run':
    print(f"unrecognized mode: {sys.argv[-1]}, defaulting to run")


itr_ids = list(range(5))
base_len = 1024
total_batch_size = 32

model = {
    'name': 'Qwen/Qwen2.5-Coder-0.5B',
    '2k_bs': 32,
}

prompt_styles = [
    'ar_cot',
    'dp',
]

# Thresholds, context lengths, and step counts match the original clean Qwen
# experiments. The PHP project name is retained for checkpoint compatibility.
tasks = [
    {
        'name': 'php',
        'dataset_split': 'php',
        'train_split': 60,
        'train_len': 32_768,
        'test_len': 32_768,
        'max_steps': 500,
        'project_name': 'qwen_php_enum',
    },
]


configs = []

for itr_id, task, prompt in itertools.product(itr_ids, tasks, prompt_styles):
    len_scale_fac = base_len / task['train_len']
    batch_size = min(int(model['2k_bs'] * len_scale_fac), total_batch_size)
    batch_size = max(batch_size, 1)
    accum_steps = max(total_batch_size // batch_size, 1)

    run_name = (
        f"{model['name'].split('/')[-1]}-{prompt} "
        f"(split={task['train_split']}, itr={itr_id})"
    )

    configs.append({
        'model_name': model['name'],
        'task_name': task['name'],
        'dataset_split': task['dataset_split'],
        'dataset_num_proc': 16,
        'dataset_cache_dir': data_path + '/.datasets_cache',
        'batch_size': batch_size,
        'accum_steps': accum_steps,
        'max_steps': task['max_steps'],
        'num_samples': 16,
        'max_length': task['train_len'],
        'max_test_length': task['test_len'],
        'log_every': 250 if prompt == 'dp' else 999_999_999,
        'save_every': 250,
        'ds_path': data_path,
        'splits': [task['train_split']],
        'test_splits': [task['train_split']],
        'train_split': task['train_split'],
        'prompt': prompt,
        'project_name': task['project_name'],
        'run_name': run_name,
        'output_dir': (
            f"~/scratch/ckpt/final/{task['project_name']}/"
            f"{run_name.replace(' ', '_')}"
        ),
        'packing': False,
    })

print(len(configs))
# <codecell>
perform(configs)
