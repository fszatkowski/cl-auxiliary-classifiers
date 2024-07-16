from tests import run_main_and_assert

FAST_LOCAL_TEST_ARGS = (
    "--exp-name local_test --datasets mnist"
    " --network LeNet --num-tasks 3 --seed 1 --batch-size 32"
    " --nepochs 3"
    " --num-workers 0"
    " --approach ancl"
    " --lamb 0.1"
    " --lamb-a 0.1"
)


def test_lwfa_without_exemplars():
    run_main_and_assert(FAST_LOCAL_TEST_ARGS)


def test_lwfa_with_exemplars():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --num-exemplars 200"
    run_main_and_assert(args_line)


def test_lwfa_taskwise():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --taskwise-kd"
    run_main_and_assert(args_line)


def test_lwfa_taskwise_with_exemplars():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --taskwise-kd --num-exemplars 200"
    run_main_and_assert(args_line)


def test_lwfa_with_early_exits():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --ic-config test_mnist"
    run_main_and_assert(FAST_LOCAL_TEST_ARGS)


def test_lwfa_with_early_exits_with_exemplars():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --ic-config test_mnist --num-exemplars 200"
    run_main_and_assert(args_line)


def test_lwfa_taskwise_with_early_exits():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --taskwise-kd --ic-config test_mnist"
    run_main_and_assert(FAST_LOCAL_TEST_ARGS)


def test_lwfa_taskwise_with_early_exits_with_exemplars():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --taskwise-kd --ic-config test_mnist --num-exemplars 200"
    run_main_and_assert(args_line)
