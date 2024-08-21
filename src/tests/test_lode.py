from tests import run_main_and_assert

FAST_LOCAL_TEST_ARGS = (
    "--exp-name local_test --datasets mnist"
    " --network LeNet --num-tasks 3 --seed 1 --batch-size 32"
    " --nepochs 2 --lr-factor 10 --momentum 0.9 --lr-min 1e-7"
    " --num-workers 0"
)


def test_lode():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --approach lode"
    args_line += " --num-exemplars 200"
    run_main_and_assert(args_line)


def test_lode_with_early_exits():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --approach lode"
    args_line += " --num-exemplars 200"
    args_line += " --ic-config test_mnist"
    run_main_and_assert(args_line)
