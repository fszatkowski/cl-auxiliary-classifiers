from tests import run_main_and_assert

FAST_LOCAL_TEST_ARGS = (
    "--exp-name local_test --datasets mnist"
    " --network LeNet --num-tasks 3 --seed 1 --batch-size 32"
    " --nepochs 3"
    " --num-workers 0"
    " --gridsearch-tasks -1"
    " --approach podnet"
)


def test_podnet_exemplars():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --num-exemplars-per-class 20"
    args_line += " --pod-fmap-layers conv1 conv2"
    run_main_and_assert(args_line)


def test_podnet_exemplars_early_exit():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --num-exemplars-per-class 20"
    args_line += " --ic-config test_mnist"
    run_main_and_assert(args_line)
