from tests import run_main_and_assert

FAST_LOCAL_TEST_ARGS = (
    "--exp-name local_test --datasets mnist"
    " --network LeNet --num-tasks 3 --seed 1 --batch-size 32"
    " --nepochs 3"
    " --num-workers 0"
    " --approach ewc"
)


def test_ewc_without_exemplars():
    run_main_and_assert(FAST_LOCAL_TEST_ARGS)


def test_ewc_with_exemplars():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --num-exemplars 200"
    run_main_and_assert(args_line)


def test_ewc_with_early_exits():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --ic-config test_mnist"
    run_main_and_assert(args_line)
