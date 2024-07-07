from tests import run_main_and_assert

FAST_LOCAL_TEST_ARGS = (
    "--exp-name local_test "
    "--datasets cifar100"
    " --network resnet32"
    " --num-tasks 2 --seed 1"
    " --batch-size 128"
    " --nepochs 10"
    " --num-workers 0"
    " --approach prd"
    " --projection-head-output-size 128"
    " --projection-head-hidden-size 256"
    " --projection-head-num-layers 2"
    " --feature-dim 64"
    " --n-classes-per-task 50"
    " --n-tasks 2"
)


def test_prd():
    args_line = FAST_LOCAL_TEST_ARGS
    run_main_and_assert(args_line)
