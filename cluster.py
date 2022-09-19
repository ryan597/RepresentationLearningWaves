import pytorch_lightning as pl
from test_tube import SlurmCluster, HyperOptArgumentParser
from train import main

if __name__ == "__main__":

    parser = HyperOptArgumentParser(strategy="grid_search", add_help=False)

    parser.add_argument("--train_path", default="data",
                        help="Path to directory of training datasets")
    parser.add_argument("--valid_path", default="data/test",
                        help="Path to directory of validation datasets")
    parser.add_argument("--test_path",
                        help="Path to directory of testing datasets",
                        default=None)
    parser.add_argument("--batch_size", default=10,
                        help="Number of samples to include in each batch")
    parser.add_argument("--masks",
                        help="Train for segmentation or frame prediciton",
                        default=False)
    parser.add_argument("--seq_length",
                        help="Length of sequence to use",
                        default=2)
    parser.add_argument("--checkpoint",
                        help="Path to checkpoint",
                        default=False)
    parser.opt_list("--backbone", default="resnet", type=str,
        options=["resnet", "baseline", "resunet"],
        help="Backbone of model, resnet or resunet")
    parser.opt_list("--size", default=256, type=int,
        options=[128, 256])
    parser.opt_list("--freeze", default=5, type=int,
        options=[0, 2, 4, 6, 8], tunable=True)
    parser.opt_list("--lr", default=0.001, type=float,
        options=[1e-4], tunable=True)
    parser.opt_list("--layers", default=50, type=int,
        options=[50], tunable=True)

    parser = pl.Trainer.add_argparse_args(parser)

    hyperparams = parser.parse_args()

    cluster = SlurmCluster(
        hyperparam_optimizer=hyperparams,
        log_path="outputs",
    )

    # configure cluster
    cluster.job_time = "10:00:00"
    cluster.per_experiment_nb_nodes = 1
    cluster.per_experiment_nb_gpus = 2
    cluster.memory_mb_per_node = 0  # use all available memory
    cluster.add_slurm_cmd(cmd="account", value="ndmat033a", comment="")
    cluster.add_slurm_cmd(cmd="partition", value="GpuQ", comment="")

    cluster.load_modules(["intel/2019u5", "cuda/11.3", "cudnn", "conda/2"])
    cluster.add_command("source activate rlwave")
    cluster.add_command("export NCCL_DEBUG=WARN")
    cluster.add_command("export PYTHONFAULTHANDLER=1")
    cluster.add_command("export NCCL_SOCKET_IFNAME=eth0")
    cluster.add_command("export NCCL_IB_DISABLE=1")

    cluster.optimize_parallel_cluster_gpu(
        main, nb_trials=15, job_name="gridSearch")
