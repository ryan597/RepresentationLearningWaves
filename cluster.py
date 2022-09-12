import pytorch_lightning as pl
from test_tube import SlurmCluster, HyperOptArgumentParser
from train import main

if __name__ == "__main__":

    parser = HyperOptArgumentParser(strategy="grid_search", add_help=False)

    parser.add_argument("--train_path",
                        help="Path to directory of training datasets")
    parser.add_argument("--valid_path",
                        help="Path to directory of validation datasets",
                        default=None)
    parser.add_argument("--test_path",
                        help="Path to directory of testing datasets",
                        default=None)
    parser.add_argument("--size",
                        help="Image shape as (SIZE, 2*SIZE)")
    parser.add_argument("--batch_size",
                        help="Number of samples to include in each batch")
    parser.add_argument("--backbone",
                        help="Backbone of model, resnet or resunet",
                        default="resunet")
    parser.add_argument("--masks",
                        help="Train for segmentation or frame prediciton",
                        default=False)
    parser.add_argument("--dual",
                        help="Whether to use single or dual image inputs",
                        default=False)
    parser.add_argument("--checkpoint",
                        help="Path to checkpoint",
                        default=False)

    parser.opt_list("--freeze", default=5, type=int,
        options=[0, 1, 2, 3, 4, 5, 6, 7, 8], tunable=True)
    parser.opt_list("--lr", default=0.001, type=float,
        options=[1e-2, 1e-3, 1e-4], tunable=True)
    parser.opt_list("--layers", default=50, type=int,
        options=[18, 50], tunable=True)

    parser = pl.Trainer.add_argparse_args(parser)

    hyperparams = parser.parse_args()

    # Slurm cluster submits 9 jobs, each with a set of hyperparams
    cluster = SlurmCluster(
        hyperparam_optimizer=hyperparams,
        log_path="outputs",
    )

    cluster.add_slurm_cmd(cmd="time", value="48:00:00", comment="")
    cluster.add_slurm_cmd(cmd="nodes", value=4, comment="")
    cluster.add_slurm_cmd(cmd="ntasks-per-node", value=2, comment="")
    cluster.add_slurm_cmd(cmd="account", value="ndmat033a")
    cluster.add_slurm_cmd(cmd="partition", value="GpuQ")

    cluster.add_command("module purge")
    cluster.load_modules(["intel/2019u5", "cuda/11.3", "cudnn", "conda/2"])
    cluster.add_command("source activate rlwave")
    cluster.add_command("export NCCL_DEBUG=WARN")
    cluster.add_command("export PYTHONFAULTHANDLER=1")
    cluster.add_command("export NCCL_SOCKET_IFNAME=eth0")
    cluster.add_command("export NCCL_IB_DISABLE=1")

    # configure cluster
    cluster.per_experiment_nb_nodes = 1
    cluster.per_experiment_nb_gpus = 2
    cluster.memory_mb_per_node = 32000

    cluster.optimize_parallel_cluster_gpu(
        main, nb_trials=5, job_name="gridSearch")
