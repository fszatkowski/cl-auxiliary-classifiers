from pathlib import Path

import torch
from tqdm import tqdm

global device
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def process_data(data_dict):
    outputs_tag = data_dict["outputs_tag"]
    targets = data_dict["targets"]
    probs_tag = outputs_tag.softmax(dim=2)
    num_ics = probs_tag.shape[1]
    preds_tag = probs_tag.argmax(dim=2)
    max_conf_tag = probs_tag.max(dim=2).values
    max_conf_ic_idx = max_conf_tag.argmax(dim=1)
    max_conf_preds = (
        torch.nn.functional.one_hot(max_conf_ic_idx, num_ics) * preds_tag
    ).sum(dim=1)
    return max_conf_preds, max_conf_ic_idx, targets


def get_seed_static_ee_results(path):
    test_logits_dir = list(path.rglob("logits_test"))[0]
    per_task_test_logit_dirs = list(test_logits_dir.glob("*"))
    num_tasks = len(per_task_test_logit_dirs)

    num_ics = 7
    ic_selection = torch.zeros((num_ics, num_tasks + 1))
    ic_hits = torch.zeros((num_ics, num_tasks + 1))

    for task_dir in per_task_test_logit_dirs:
        task_id = int(task_dir.name.split("_")[1])
        logit_paths = list(task_dir.glob("*.pt"))
        if len(logit_paths) == 0:
            print(f"Incomplete data in {task_dir}")
            return None
        data = [
            torch.load(logit_path, map_location=device) for logit_path in logit_paths
        ]
        outputs = [process_data(data_dict) for data_dict in data]
        max_conf_preds = torch.cat([output[0] for output in outputs], dim=0)
        max_conf_ic_idx = torch.cat([output[1] for output in outputs], dim=0)
        targets = torch.cat([output[2] for output in outputs], dim=0)

        ic_selection_one_hot = torch.nn.functional.one_hot(max_conf_ic_idx, num_ics)
        ic_selection_vector = ic_selection_one_hot.sum(dim=0)
        ic_selection[:, task_id] = ic_selection_vector

        hits = (max_conf_preds == targets).unsqueeze(1).float()
        per_ic_hits = (hits * ic_selection_one_hot).sum(dim=0)
        ic_hits[:, task_id] = per_ic_hits

    # Make the last rows totals
    ic_selection[:, -1] = ic_selection[:, :-1].sum(dim=1)
    ic_hits[:, -1] = ic_hits[:, :-1].sum(dim=1)
    accs = 100 * ic_hits / ic_selection
    ic_selection = 100 * ic_selection / ic_selection.sum(dim=0)
    return ic_selection, accs


def method_fn(x):
    if x.startswith("finetuning_ex0"):
        return "FT"
    elif x.startswith("finetuning_ex2000"):
        return "FT+Ex"
    elif x.startswith("lwf"):
        return "LwF"
    elif x.startswith("bic"):
        return "BiC"
    elif x.startswith("ancl"):
        return "ANCL"
    elif x.startswith("ssil"):
        return "SSIL"
    elif x.startswith("lode"):
        return "LODE"
    elif x.startswith("er"):
        return "ER"
    elif x.startswith("ewc"):
        return "EWC"
    elif x.startswith("gdumb"):
        return "GDumb"
    else:
        raise ValueError()


def parse_setting_path(path):
    method = method_fn(path.name)
    seed_dirs = sorted(list(path.glob("*")))

    cumulative_ic_selection = None
    cumulative_accs = None
    cnt = 0
    for seed_dir in seed_dirs:
        results = get_seed_static_ee_results(seed_dir)
        if results is None:
            continue
        else:
            ic_selection, accs = results

        if cnt == 0:
            cumulative_ic_selection = ic_selection
            cumulative_accs = accs
        else:
            cumulative_ic_selection += ic_selection
            cumulative_accs += accs
        cnt += 1
    cumulative_ic_selection /= cnt
    cumulative_accs /= cnt

    return {
        "method": method,
        "ic_selection": cumulative_ic_selection,
        "accs": cumulative_accs,
    }


if __name__ == "__main__":
    root = Path(__file__).parent.parent.parent
    result_dir = root / "results"
    output_dir = root / "analysis_outputs" / "decision_masks"
    output_dir.mkdir(exist_ok=True, parents=True)
    output_path = output_dir / "decision_masks.pt"

    outputs = []
    for setting in ["CIFAR100x5", "CIFAR100x10"]:
        setting_paths = sorted(list(result_dir.joinpath(setting).glob("*")))
        setting_paths = [p for p in setting_paths if p.name.endswith("sdn")]

        for path in tqdm(setting_paths, desc=f"Parsing results for {setting}..."):
            output = parse_setting_path(path)
            output["setting"] = setting
            outputs.append(output)
    torch.save(outputs, output_path)
