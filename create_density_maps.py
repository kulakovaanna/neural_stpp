import argparse
import numpy as np
import os
import sys
import time

import torch
import datasets
import matplotlib.pyplot as plt
from matplotlib import patheffects
from iterators import EpochBatchIterator
from models import CombinedSpatiotemporalModel, JumpCNFSpatiotemporalModel, SelfAttentiveCNFSpatiotemporalModel, JumpGMMSpatiotemporalModel
from models.spatial import GaussianMixtureSpatialModel, IndependentCNF, JumpCNF, SelfAttentiveCNF
from models.spatial.cnf import TimeVariableCNF
from models.temporal import HomogeneousPoissonPointProcess, HawkesPointProcess, SelfCorrectingPointProcess, NeuralPointProcess
from models.temporal.neural import ACTFNS as TPP_ACTFNS
from models.temporal.neural import TimeVariableODE
import toy_datasets
import utils
from viz_dataset import load_data, MAPS, BBOXES

from train_stpp import *

torch.backends.cudnn.benchmark = True

FIGSIZE = 10
DPI = 300
SAVEPATH = "/home/a.kylakova/mipt/nir/neural_stpp/data_maps"
EXP = True

def plot_density(loglik_fn, spatial_locations, S_mean, S_std, dataset_name, device, text=None, fp64=False, N=50, model_name="model", exp=EXP):

    x = np.linspace(BBOXES[dataset_name][0], BBOXES[dataset_name][1], N)
    y = np.linspace(BBOXES[dataset_name][2], BBOXES[dataset_name][3], N)
    s = np.stack([x, y], axis=1)

    X, Y = np.meshgrid(s[:, 0], s[:, 1])
    S = np.stack([X.reshape(-1), Y.reshape(-1)], axis=1)
    S = torch.tensor(S).to(device)
    S = S.double() if fp64 else S.float()
    S = (S - S_mean.to(S)) / S_std.to(S)
    logp = loglik_fn(S)

    if MAPS[dataset_name]:
        map_img = plt.imread(MAPS[dataset_name])
        fig, ax = plt.subplots(figsize=(FIGSIZE, FIGSIZE * map_img.shape[0] / map_img.shape[1]))
        ax.imshow(map_img, zorder=0, extent=BBOXES[dataset_name])
    else:
        fig, ax = plt.subplots(figsize=(FIGSIZE, FIGSIZE))

    if exp:
        Z = logp.exp().detach().cpu().numpy().reshape(N, N)
        model_name = f"{model_name}_exp"
    else:
        Z = logp.detach().cpu().numpy().reshape(N, N)
        model_name = f"{model_name}_log"
    
    ax.contourf(X, Y, Z, levels=20, alpha=0.7, cmap='RdGy')

    spatial_locations = spatial_locations * np.array(S_std) + np.array(S_mean)
    ax.scatter(spatial_locations[:, 0], spatial_locations[:, 1], s=20**2, alpha=1.0, marker="x", color="k")

    ax.set_xlim(BBOXES[dataset_name][0], BBOXES[dataset_name][1])
    ax.set_ylim(BBOXES[dataset_name][2], BBOXES[dataset_name][3])

    if text:
        txt = ax.text(0.15, 0.9, text,
                      horizontalalignment="center",
                      verticalalignment="center",
                      transform=ax.transAxes,
                      size=16,
                      color='white')
        txt.set_path_effects([patheffects.withStroke(linewidth=5, foreground='black')])

    plt.axis('off')
    os.makedirs(os.path.join(SAVEPATH), exist_ok=True)
    plt.savefig(os.path.join(SAVEPATH, f"density_{model_name}.png"), bbox_inches='tight', dpi=DPI)
    plt.close()
    
    return X, Y, Z

def get_dim(data):
    if data == "gmm":
        return 1
    elif data == "fmri":
        return 3
    else:
        return 2

def plot_main(args):
    if args.resume is None:
        print("resume path is empty")
        sys.exit(1)

    device = f"cuda:{args.device}"
    print(f"device: {device}")
    print(f"model_name: {args.model_name}")

    x_dim = get_dim(args.data)
    
    if args.model == "jumpcnf" and args.tpp == "neural":
        model = JumpCNFSpatiotemporalModel(dim=x_dim,
                                            hidden_dims=list(map(int, args.hdims.split("-"))),
                                            tpp_hidden_dims=list(map(int, args.tpp_hdims.split("-"))),
                                            actfn=args.actfn,
                                            tpp_cond=args.tpp_cond,
                                            tpp_style=args.tpp_style,
                                            tpp_actfn=args.tpp_actfn,
                                            share_hidden=args.share_hidden,
                                            solve_reverse=args.solve_reverse,
                                            tol=args.tol,
                                            otreg_strength=args.otreg_strength,
                                            tpp_otreg_strength=args.tpp_otreg_strength,
                                            layer_type=args.layer_type,
                                            ).to(device)
    elif args.model == "attncnf" and args.tpp == "neural":
        model = SelfAttentiveCNFSpatiotemporalModel(dim=x_dim,
                                                    hidden_dims=list(map(int, args.hdims.split("-"))),
                                                    tpp_hidden_dims=list(map(int, args.tpp_hdims.split("-"))),
                                                    actfn=args.actfn,
                                                    tpp_cond=args.tpp_cond,
                                                    tpp_style=args.tpp_style,
                                                    tpp_actfn=args.tpp_actfn,
                                                    share_hidden=args.share_hidden,
                                                    solve_reverse=args.solve_reverse,
                                                    l2_attn=args.l2_attn,
                                                    tol=args.tol,
                                                    otreg_strength=args.otreg_strength,
                                                    tpp_otreg_strength=args.tpp_otreg_strength,
                                                    layer_type=args.layer_type,
                                                    lowvar_trace=not args.naive_hutch,
                                                    ).to(device)
    elif args.model == "cond_gmm" and args.tpp == "neural":
        model = JumpGMMSpatiotemporalModel(dim=x_dim,
                                            hidden_dims=list(map(int, args.hdims.split("-"))),
                                            tpp_hidden_dims=list(map(int, args.tpp_hdims.split("-"))),
                                            actfn=args.actfn,
                                            tpp_cond=args.tpp_cond,
                                            tpp_style=args.tpp_style,
                                            tpp_actfn=args.tpp_actfn,
                                            share_hidden=args.share_hidden,
                                            tol=args.tol,
                                            tpp_otreg_strength=args.tpp_otreg_strength,
                                            ).to(device)
    else:
        # Mix and match between spatial and temporal models.
        if args.tpp == "poisson":
            tpp_model = HomogeneousPoissonPointProcess()
        elif args.tpp == "hawkes":
            tpp_model = HawkesPointProcess()
        elif args.tpp == "correcting":
            tpp_model = SelfCorrectingPointProcess()
        elif args.tpp == "neural":
            tpp_hidden_dims = list(map(int, args.tpp_hdims.split("-")))
            tpp_model = NeuralPointProcess(
                cond_dim=x_dim, hidden_dims=tpp_hidden_dims, cond=args.tpp_cond, style=args.tpp_style, actfn=args.tpp_actfn,
                otreg_strength=args.tpp_otreg_strength, tol=args.tol)
        else:
            raise ValueError(f"Invalid tpp model {args.tpp}")

        if args.model == "gmm":
            model = CombinedSpatiotemporalModel(GaussianMixtureSpatialModel(), tpp_model).to(device)
        elif args.model == "cnf":
            model = CombinedSpatiotemporalModel(
                IndependentCNF(dim=x_dim, hidden_dims=list(map(int, args.hdims.split("-"))),
                                layer_type=args.layer_type, actfn=args.actfn, tol=args.tol, otreg_strength=args.otreg_strength,
                                squash_time=True),
                tpp_model).to(device)
        elif args.model == "tvcnf":
            model = CombinedSpatiotemporalModel(
                IndependentCNF(dim=x_dim, hidden_dims=list(map(int, args.hdims.split("-"))),
                                layer_type=args.layer_type, actfn=args.actfn, tol=args.tol, otreg_strength=args.otreg_strength),
                tpp_model).to(device)
        elif args.model == "jumpcnf":
            model = CombinedSpatiotemporalModel(
                JumpCNF(dim=x_dim, hidden_dims=list(map(int, args.hdims.split("-"))),
                        layer_type=args.layer_type, actfn=args.actfn, tol=args.tol, otreg_strength=args.otreg_strength),
                tpp_model).to(device)
        elif args.model == "attncnf":
            model = CombinedSpatiotemporalModel(
                SelfAttentiveCNF(dim=x_dim, hidden_dims=list(map(int, args.hdims.split("-"))),
                                    layer_type=args.layer_type, actfn=args.actfn, l2_attn=args.l2_attn, tol=args.tol, otreg_strength=args.otreg_strength),
                tpp_model).to(device)
        else:
            raise ValueError(f"Invalid model {args.model}")

    params = []
    attn_params = []
    for name, p in model.named_parameters():
        if "self_attns" in name:
            attn_params.append(p)
        else:
            params.append(p)

    optimizer = torch.optim.AdamW([
        {"params": params},
        {"params": attn_params}
    ], lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.98))

    checkpt_path = args.resume
    if os.path.exists(checkpt_path):
        print(f"Resuming checkpoint from {checkpt_path}")
        checkpt = torch.load(checkpt_path, "cpu")

    model.load_state_dict(checkpt["state_dict"])
    optimizer.load_state_dict(checkpt["optim_state_dict"])

    model.to(device)
    model.eval()

    args_data = args.data
    t0, t1 = map(lambda x: cast(x, device), get_t0_t1(args_data))
    train_set = load_data(args_data, split="train")
    train_date_sequence_dict = dict(zip(train_set.file_splits["train"], train_set))
    train_date = np.sort(list(train_date_sequence_dict.keys()))[1]

    with torch.no_grad():
        sequence = train_date_sequence_dict[train_date].to(device)
        
        print(f"train_date: {train_date}, {train_date_sequence_dict[train_date].shape}")
        print(f"args.data: {args_data}, sequence.shape: {sequence.shape}, len(train_set): {len(train_set)}")

        torch.cuda.empty_cache()

        model_logprob_fn = model.spatial_conditional_logprob_fn(
            t=torch.tensor(60., device=device), 
            event_times=sequence[:, 0], 
            spatial_locations=sequence[:, 1:3], 
            t0=t0, 
            t1=t1,
            # aux_state=None
        )

        print("density map", end=" ")
        x, y, p = plot_density(
            loglik_fn=model_logprob_fn, 
            spatial_locations=sequence[:, 1:3].cpu(), 
            index="", 
            S_mean=train_set.S_mean.cpu(), 
            S_std=train_set.S_std.cpu(), 
            savepath="", 
            dataset_name="earthquakes_orig_magn_3_5", 
            device=device, 
            text=None, 
            fp64=False,
            model_name=args.model_name,
            N=args.num
        )
        print("saved")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, choices=MAPS.keys(), default="earthquakes_jp")

    parser.add_argument("--model", type=str, choices=["cond_gmm", "gmm", "cnf", "tvcnf", "jumpcnf", "attncnf"], default="gmm")
    parser.add_argument("--tpp", type=str, choices=["poisson", "hawkes", "correcting", "neural"], default="poisson")
    parser.add_argument("--actfn", type=str, default="swish")
    parser.add_argument("--tpp_actfn", type=str, choices=TPP_ACTFNS.keys(), default="softplus")
    parser.add_argument("--hdims", type=str, default="64-64-64")
    parser.add_argument("--layer_type", type=str, choices=["concat", "concatsquash"], default="concat")
    parser.add_argument("--tpp_hdims", type=str, default="32-32")
    parser.add_argument("--tpp_nocond", action="store_false", dest='tpp_cond')
    parser.add_argument("--tpp_style", type=str, choices=["split", "simple", "gru"], default="gru")
    parser.add_argument("--no_share_hidden", action="store_false", dest='share_hidden')
    parser.add_argument("--solve_reverse", action="store_true")
    parser.add_argument("--l2_attn", action="store_true")
    parser.add_argument("--naive_hutch", action="store_true")
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--otreg_strength", type=float, default=1e-4)
    parser.add_argument("--tpp_otreg_strength", type=float, default=1e-4)

    parser.add_argument("--warmup_itrs", type=int, default=0)
    parser.add_argument("--num_iterations", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--gradclip", type=float, default=0)
    parser.add_argument("--max_events", type=int, default=4000)
    parser.add_argument("--test_bsz", type=int, default=32)

    parser.add_argument("--experiment_dir", type=str, default="experiments")
    parser.add_argument("--experiment_id", type=str, default=None)
    parser.add_argument("--ngpus", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--logfreq", type=int, default=10)
    parser.add_argument("--testfreq", type=int, default=100)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--num", type=int, default=50)

    parser.add_argument("--model_name", type=str, default="model")
    args = parser.parse_args()

    if args.port is None:
        args.port = int(np.random.randint(10000, 20000))

    if args.experiment_id is None:
        args.experiment_id = time.strftime("%Y%m%d_%H%M%S")

    experiment_name = f"{args.model}"
    if args.model in ["cnf", "tvcnf", "jumpcnf", "attncnf"]:
        experiment_name += f"{args.hdims}"
        experiment_name += f"_{args.layer_type}"
        experiment_name += f"_{args.actfn}"
        experiment_name += f"_ot{args.otreg_strength}"

    if args.model == "attncnf":
        if args.l2_attn:
            experiment_name += "_l2attn"
        if args.naive_hutch:
            experiment_name += "_naivehutch"

    if args.model in ["cnf", "tvcnf", "jumpcnf", "attncnf"]:
        experiment_name += f"_tol{args.tol}"

    experiment_name += f"_{args.tpp}"
    if args.tpp in ["neural"]:
        experiment_name += f"{args.tpp_hdims}"
        experiment_name += f"{args.tpp_style}"
        experiment_name += f"_{args.tpp_actfn}"
        experiment_name += f"_ot{args.tpp_otreg_strength}"
        if args.tpp_cond:
            experiment_name += "_cond"
    if args.share_hidden and args.model in ["jumpcnf", "attncnf"] and args.tpp == "neural":
        experiment_name += "_sharehidden"
    if args.solve_reverse and args.model == "jumpcnf" and args.tpp == "neural":
        experiment_name += "_rev"
    experiment_name += f"_lr{args.lr}"
    experiment_name += f"_gc{args.gradclip}"
    experiment_name += f"_bsz{args.max_events}x{args.ngpus}_wd{args.weight_decay}_s{args.seed}"
    experiment_name += f"_{args.experiment_id}"
    savepath = os.path.join(args.experiment_dir, experiment_name)

    if args.gradclip == 0:
        args.gradclip = 1e10

    plot_main(args=args)