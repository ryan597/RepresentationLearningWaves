import pytorch_lightning as pl
from test_tube import HyperOptArgumentParser, SlurmCluster

from train import main

if __name__ == "__main__":

    parser = HyperOptArgumentParser(strategy="grid_search", add_help=False)

    # Fixed args
    parser.add_argument("--train_path", default="data",
                        help="Path to directory of training datasets")
    parser.add_argument("--valid_path", default="data/test",
                        help="Path to directory of validation datasets")
    parser.add_argument("--test_path",
                        help="Path to directory of testing datasets",
                        default=None)
    parser.add_argument("--masks",
                        help="Train for segmentation or frame prediciton",
                        default=False)
    parser.add_argument("--checkpoint",
                        help="Path to checkpoint",
                        default=False)
    parser.add_argument("--testing", default=False)
    parser.opt_list("--lr", default=0.001, type=float, options=[1e-3, 1e-4], tunable=False)

    # Tunable args
    parser.opt_list("--step", default=1, type=int, options=[1, 3], tunable=True)
    parser.opt_list("--seq_length", default=5, type=int, options=[5], tunable=False)
    parser.opt_list("--freeze", default=0, type=int, options=[0], tunable=False)
    parser.opt_list("--size", default=512, type=int, options=[512], tunable=False)

    # Backbone Args
    parser.opt_list("--backbone", default="attention", type=str,
                    options=["resnet", "baseline", "resunet", "attention"],
                    tunable=False)
    parser.opt_list("--layers", default=50, type=int, options=[50], tunable=False)

    parser = pl.Trainer.add_argparse_args(parser)

    hyperparams = parser.parse_args()

    cluster = SlurmCluster(hyperparam_optimizer=hyperparams, log_path="outputs")

    # configure cluster
    cluster.job_time = "15:00:00"
    cluster.per_experiment_nb_nodes = 4
    cluster.per_experiment_nb_gpus = 8
    cluster.memory_mb_per_node = 0  # use all available memory
    cluster.add_slurm_cmd(cmd="account", value="ndear024a", comment="")
    cluster.add_slurm_cmd(cmd="partition", value="GpuQ", comment="")
    cluster.add_slurm_cmd(cmd="ntasks-per-node", value=2, comment="")

    cluster.add_slurm_cmd(cmd="exclude", value="n368", comment="")

    cluster.load_modules(["intel/2019u5", "cuda/11.3", "cudnn", "conda/2"])
    cluster.add_command("source activate rlwave")
    cluster.add_command("export NCCL_DEBUG=WARN")
    cluster.add_command("export PYTHONFAULTHANDLER=1")
    cluster.add_command("export NCCL_SOCKET_IFNAME=eth0")
    cluster.add_command("export NCCL_IB_DISABLE=1")

    cluster.optimize_parallel_cluster_gpu(main, nb_trials=2, job_name="gridSearch")
