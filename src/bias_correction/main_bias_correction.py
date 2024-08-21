from argparse import ArgumentParser
from pathlib import Path
from typing import List, Optional

import torch

from bias_correction.data_utils import load_data, load_exit_costs
from bias_correction.eval import evaluate
from bias_correction.methods.hyperopt_temperature import HyperoptTemperatureCorrection
from bias_correction.methods.hyperopt_temperature_seq import (
    HyperoptSequentialTemperatureCorrection,
)
from bias_correction.methods.hyperopt_temperature_shared import (
    HyperoptSharedTemperatureCorrection,
)
from bias_correction.methods.icicle import ICICLE
from bias_correction.methods.identity import Identity
from bias_correction.methods.lcsgd import LCSGD
from bias_correction.methods.lctsgd import LCTSGD
from bias_correction.methods.lctsgdalt import LCTSGDAlt
from bias_correction.methods.mean_lctsgd import MeanLCTSGD
from bias_correction.methods.taskwise_confidence_bias import TCB
from bias_correction.methods.taskwise_temperature import TaskWiseTemperatureCorrection
from bias_correction.methods.tlc import TLC
from bias_correction.methods.tlcbsgd import TLCBSGD
from bias_correction.methods.tlcic import TLCIC
from bias_correction.methods.tlcicsgd import TLCICSGD
from bias_correction.methods.tlcsgd import TLCSGD
from bias_correction.visualize import (
    plot_biases,
    plot_cost_vs_acc_comparison,
    plot_ic_stats,
)


def main(
    input_dir: Path,
    output_dir: Optional[Path],
    methods: List[str],
    device: str,
    n_thresholds: int,
    icicle_u: List[float],
    icicle_eps: float,
    icicle_batch_size: int,
    sgd_n_steps: int,
    sgd_lr: float,
    tlc_algorithm: str,
    tlc_max_iters: int,
    tlc_hp_space: str,
    tlc_hp_mu: Optional[float],
    tlc_hp_sigma: Optional[float],
    tlc_hp_min: Optional[float],
    tlc_hp_max: Optional[float],
    tcb_biases: List[float],
    ttc_bases: List[float],
    ttc_deltas: List[float],
    plot_best_methods: bool,
    overwrite: bool = True,
):
    if output_dir is None:
        output_dir = input_dir / "bc"

    if (output_dir / "cost_vs_acc.png").exists() and not overwrite:
        return

    train_data, test_data = load_data(input_dir)
    exit_costs = load_exit_costs(input_dir)

    n_tasks = len(test_data)
    n_cls = train_data.logits.shape[1]
    classes_per_task = train_data.logits.shape[2] // n_tasks

    bias_correctors = {"baseline": Identity(n_tasks, n_cls, classes_per_task, device)}
    for method in methods:
        if method == "icicle":
            for u in icicle_u:
                icicle_bias_correction = ICICLE(
                    n_tasks=n_tasks,
                    n_cls=n_cls,
                    classes_per_task=classes_per_task,
                    u=u,
                    eps=icicle_eps,
                    batch_size=icicle_batch_size,
                    device=device,
                )
                icicle_bias_correction.fit_bias_correction(train_data, test_data)
                bias_correctors[f"icicle_u{u}"] = icicle_bias_correction
        elif method == "tcb":
            for tcb_bias in tcb_biases:
                tcb_bias_correction = TCB(
                    n_tasks=n_tasks,
                    n_cls=n_cls,
                    classes_per_task=classes_per_task,
                    device=device,
                    bias=tcb_bias,
                )
                tcb_bias_correction.fit_bias_correction(train_data, test_data)
                bias_correctors[f"tcb_{tcb_bias}"] = tcb_bias_correction
        elif method == "tlc":
            tlc_bias_correction = TLC(
                n_tasks=n_tasks,
                n_cls=n_cls,
                classes_per_task=classes_per_task,
                device=device,
                algorithm=tlc_algorithm,
                max_iters=tlc_max_iters,
                hp_space=tlc_hp_space,
                hp_mu=tlc_hp_mu,
                hp_sigma=tlc_hp_sigma,
                hp_min=tlc_hp_min,
                hp_max=tlc_hp_max,
            )
            tlc_bias_correction.fit_bias_correction(train_data, test_data)
            bias_correctors["tlc"] = tlc_bias_correction
        elif method == "lcsgd":
            tlc_bias_correction = LCSGD(
                n_tasks=n_tasks,
                n_cls=n_cls,
                classes_per_task=classes_per_task,
                n_steps=sgd_n_steps,
                lr=sgd_lr,
                device=device,
            )
            tlc_bias_correction.fit_bias_correction(train_data, test_data)
            bias_correctors["lcsgd"] = tlc_bias_correction
        elif method == "lctsgd":
            tlc_bias_correction = LCTSGD(
                n_tasks=n_tasks,
                n_cls=n_cls,
                classes_per_task=classes_per_task,
                n_steps=sgd_n_steps,
                lr=sgd_lr,
                device=device,
            )
            tlc_bias_correction.fit_bias_correction(train_data, test_data)
            bias_correctors["lctsgd"] = tlc_bias_correction
        elif method == "lctsgdalt":
            tlc_bias_correction = LCTSGDAlt(
                n_tasks=n_tasks,
                n_cls=n_cls,
                classes_per_task=classes_per_task,
                n_steps=sgd_n_steps,
                lr=sgd_lr,
                device=device,
            )
            tlc_bias_correction.fit_bias_correction(train_data, test_data)
            bias_correctors["lctsgdalt"] = tlc_bias_correction
        elif method == "tlcsgd":
            tlc_bias_correction = TLCSGD(
                n_tasks=n_tasks,
                n_cls=n_cls,
                classes_per_task=classes_per_task,
                n_steps=sgd_n_steps,
                lr=sgd_lr,
                device=device,
            )
            tlc_bias_correction.fit_bias_correction(train_data, test_data)
            bias_correctors["tlcsgd"] = tlc_bias_correction
        elif method == "tlcbsgd":
            tlc_bias_correction = TLCBSGD(
                n_tasks=n_tasks,
                n_cls=n_cls,
                classes_per_task=classes_per_task,
                n_steps=sgd_n_steps,
                lr=sgd_lr,
                device=device,
            )
            tlc_bias_correction.fit_bias_correction(train_data, test_data)
            bias_correctors["tlcbsgd"] = tlc_bias_correction
        elif method == "tlcicsgd":
            tlc_bias_correction = TLCICSGD(
                n_tasks=n_tasks,
                n_cls=n_cls,
                classes_per_task=classes_per_task,
                n_steps=sgd_n_steps,
                lr=sgd_lr,
                device=device,
            )
            tlc_bias_correction.fit_bias_correction(train_data, test_data)
            bias_correctors["tlcicsgd"] = tlc_bias_correction
        elif method == "tlcic":
            tlc_bias_correction = TLCIC(
                n_tasks=n_tasks,
                n_cls=n_cls,
                classes_per_task=classes_per_task,
                device=device,
                algorithm=tlc_algorithm,
                max_iters=tlc_max_iters,
                hp_space=tlc_hp_space,
                hp_mu=tlc_hp_mu,
                hp_sigma=tlc_hp_sigma,
                hp_min=tlc_hp_min,
                hp_max=tlc_hp_max,
            )
            tlc_bias_correction.fit_bias_correction(train_data, test_data)
            bias_correctors["tlcic"] = tlc_bias_correction
        elif method == "ttc":
            for ttc_base in ttc_bases:
                for ttc_delta in ttc_deltas:
                    ttc_bias_correction = TaskWiseTemperatureCorrection(
                        n_tasks=n_tasks,
                        n_cls=n_cls,
                        classes_per_task=classes_per_task,
                        device=device,
                        base=ttc_base,
                        per_task_delta=ttc_delta,
                    )
                    ttc_bias_correction.fit_bias_correction(train_data, test_data)
                    bias_correctors[f"ttc_b{ttc_base}_d{ttc_delta}"] = (
                        ttc_bias_correction
                    )
        elif method == "lctsgdttc":
            assert (
                "lctsgd" in bias_correctors
            ), "Please run 'lctsgd' first before running 'lctsgd+tc'"
            for ttc_base in ttc_bases:
                for ttc_delta in ttc_deltas:
                    ttc_bias_correction = TaskWiseTemperatureCorrection(
                        n_tasks=n_tasks,
                        n_cls=n_cls,
                        classes_per_task=classes_per_task,
                        device=device,
                        base=ttc_base,
                        per_task_delta=ttc_delta,
                        logit_adapter=bias_correctors["lctsgd"],
                    )
                    ttc_bias_correction.fit_bias_correction(train_data, test_data)
                    bias_correctors[f"lctsgdttc_b{ttc_base}_d{ttc_delta}"] = (
                        ttc_bias_correction
                    )
        elif method == "lctsgdhtc":
            assert (
                "lctsgd" in bias_correctors
            ), "Please run 'lctsgd' first before running 'lctsgdhtc'"
            hyperopt_bias_correction = HyperoptTemperatureCorrection(
                n_tasks=n_tasks,
                n_cls=n_cls,
                classes_per_task=classes_per_task,
                device=device,
                algorithm=tlc_algorithm,
                max_iters=1000,
                hp_space="normal",
                hp_mu=1.0,
                hp_sigma=0.1,
                logit_adapter=bias_correctors["lctsgd"],
                exit_costs=exit_costs,
                thresholds=n_thresholds,
            )
            hyperopt_bias_correction.fit_bias_correction(train_data, test_data)
            bias_correctors["lctsgdhtc"] = hyperopt_bias_correction
        elif method == "lctsgdhstc":
            assert (
                "lctsgd" in bias_correctors
            ), "Please run 'lctsgd' first before running 'lctsgdhstc'"
            hyperopt_bias_correction = HyperoptSequentialTemperatureCorrection(
                n_tasks=n_tasks,
                n_cls=n_cls,
                classes_per_task=classes_per_task,
                device=device,
                algorithm=tlc_algorithm,
                max_iters=1000,
                hp_space="normal",
                hp_mu=1.0,
                hp_sigma=0.1,
                logit_adapter=bias_correctors["lctsgd"],
                exit_costs=exit_costs,
                thresholds=n_thresholds,
            )
            hyperopt_bias_correction.fit_bias_correction(train_data, test_data)
            bias_correctors["lctsgdhstc"] = hyperopt_bias_correction
        elif method == "lctsgdhshtc":
            assert (
                "lctsgd" in bias_correctors
            ), "Please run 'lctsgd' first before running 'lctsgdhshtc'"
            hyperopt_bias_correction = HyperoptSharedTemperatureCorrection(
                n_tasks=n_tasks,
                n_cls=n_cls,
                classes_per_task=classes_per_task,
                device=device,
                algorithm=tlc_algorithm,
                max_iters=1000,
                hp_space="normal",
                hp_mu=1.0,
                hp_sigma=0.1,
                logit_adapter=bias_correctors["lctsgd"],
                exit_costs=exit_costs,
                thresholds=n_thresholds,
            )
            hyperopt_bias_correction.fit_bias_correction(train_data, test_data)
            bias_correctors["lctsgdhshtc"] = hyperopt_bias_correction
        elif method == "mlsctsgd":
            lct_bias_corretion = MeanLCTSGD(
                n_tasks=n_tasks,
                n_cls=n_cls,
                classes_per_task=classes_per_task,
                n_steps=sgd_n_steps,
                lr=sgd_lr,
                device=device,
            )
            lct_bias_corretion.fit_bias_correction(train_data, test_data)
            bias_correctors["mlsctsgd"] = lct_bias_corretion
        else:
            raise NotImplementedError()

    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths = {}
    method_results = {}
    eval_dir = output_dir / "eval_results"
    eval_dir.mkdir(parents=True, exist_ok=True)
    for method, bias_corrector in bias_correctors.items():
        results = evaluate(bias_corrector, test_data, exit_costs, n_thresholds)
        output_path = eval_dir / f"{method}.pt"
        torch.save(results, output_path)
        output_paths[method] = output_path
        method_results[method] = results

    if plot_best_methods:
        best_paths = {}
        best_results = {}
        for method_name, method in output_paths.items():
            key = method_name.split("_")[0]
            result = method_results[method_name]
            if key not in best_results:
                best_results[key] = result
                best_paths[key] = method
            else:
                if result["auc"] > best_results[key]["auc"]:
                    best_results[key] = result
                    best_paths[key] = method
        output_paths = best_paths

    plot_cost_vs_acc_comparison(output_paths, output_dir / "cost_vs_acc.png")
    plot_biases(bias_correctors, output_dir / "biases")
    plot_ic_stats(output_paths, output_dir / "ic_stats")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=False, default=None)
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        required=True,
        choices=["tlc", "tlcic", "tlcsgd", "ttc", "tcb", "icicle"],
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n_thresholds", type=int, default=101)
    parser.add_argument(
        "--icicle_u",
        type=float,
        nargs="+",
        default=[0.01, 0.02, 0.05, 0.10, 0.15, 0.20],
    )
    parser.add_argument("--icicle_eps", type=float, default=0.01)
    parser.add_argument("--icicle_batch_size", type=int, default=1024)
    parser.add_argument("--sgd_n_steps", type=int, default=100)
    parser.add_argument("--sgd_lr", type=float, default=0.01)
    parser.add_argument("--tlc_max_iters", type=int, default=100)
    parser.add_argument("--tlc_algorithm", type=str, default="tpe")
    parser.add_argument("--tlc_hp_space", type=str, default="normal")
    parser.add_argument("--tlc_hp_min", type=float, default=-2.0)
    parser.add_argument("--tlc_hp_max", type=float, default=2.0)
    parser.add_argument("--tlc_hp_mu", type=float, default=0.0)
    parser.add_argument("--tlc_hp_sigma", type=float, default=1.0)
    parser.add_argument("--tcb_biases", type=float, nargs="+", default=[0.05])
    parser.add_argument("--ttc_bases", type=float, default=[1.0, 2.0, 3.0])
    parser.add_argument(
        "--ttc_deltas", type=float, default=[0.01, 0.02, 0.05, 0.1, 0.2, 0.3]
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(
        input_dir=Path(
            "results/CIFAR100x10/finetuning_ex2000_cifar100_resnet32_sdn/seed0/cifar100_icarl_finetuning/"
        ),
        output_dir=Path(
            "bias_correction/CIFAR100x10/finetuning_ex2000_cifar100_resnet32_sdn/seed0/cifar100_icarl_finetuning/"
        ),
        # methods=['tlc', 'tlcsgd', 'tlcic', 'tlcicsgd','ttc', 'tcb', 'icicle'],
        # methods=["tlc", "lctsgd", 'lctsgdhstc', "lctsgdhtc"],
        methods=["tlc", "lctsgd", "lctsgdalt"],
        device="cuda" if torch.cuda.is_available() else "cpu",
        n_thresholds=101,
        icicle_u=[0.05, 0.10, 0.15, 0.2],
        icicle_eps=0.01,
        icicle_batch_size=512,
        sgd_lr=0.001,
        sgd_n_steps=10000,
        tlc_max_iters=1000,
        tlc_algorithm="tpe",
        tlc_hp_space="normal",
        tlc_hp_min=-2.0,
        tlc_hp_max=2.0,
        tlc_hp_mu=0.0,
        tlc_hp_sigma=1.0,
        tcb_biases=[0.02, 0.05, 0.10, 0.15, 0.2],
        ttc_bases=[0.5, 1.0, 2.0],
        ttc_deltas=[0.1, 0.15, 0.2, 0.25, 0.3],
        plot_best_methods=True,
    )

    exit()
    args = parse_args()
    main(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        methods=args.methods,
        device=args.device,
        n_thresholds=args.n_thresholds,
        icicle_u=args.icicle_u,
        icicle_eps=args.icicle_eps,
        icicle_batch_size=args.icicle_batch_size,
        sgd_n_steps=args.sgd_n_steps,
        sgd_lr=args.sgd_lr,
        tlc_max_iters=args.tlc_max_iters,
        tlc_algorithm=args.tlc_algorithm,
        tlc_hp_space=args.tlc_hp_space,
        tlc_hp_min=args.tlc_hp_min,
        tlc_hp_max=args.tlc_hp_max,
        tlc_hp_mu=args.tlc_hp_mu,
        tlc_hp_sigma=args.tlc_hp_sigma,
        tcb_biases=args.tcb_biases,
        ttc_deltas=args.ttc_deltas,
    )
