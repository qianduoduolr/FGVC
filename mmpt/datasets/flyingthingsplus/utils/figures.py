import argparse
import os
import pickle
import warnings
from collections import namedtuple
from itertools import cycle
from typing import Dict, List, Tuple

import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

from .util import get_str_formatted_time, ensure_dir

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
sns.set_style("ticks")
sns.set_palette("flare")

plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})
plt.rc('text', usetex=True)
plt.rcParams.update({
    # 'figure.figsize': (15, 5),
    'figure.titlesize': '20',
    'axes.titlesize': '22',
    'legend.title_fontsize': '16',
    'legend.fontsize': '14',
    'axes.labelsize': '18',
    'xtick.labelsize': '16',
    'ytick.labelsize': '16',
    'figure.dpi': 200,
})


def average_displacement_error(trajectory_a: torch.Tensor, trajectory_b: torch.Tensor) -> float:
    """
    Computes the average displacement error between two trajectory tensors.

    Parameters
    ----------
    trajectory_a : torch.Tensor
        A 2D tensor representing the first trajectory. Its shape should be (S, D),
        where S is the number of time steps and D is the number of dimensions.
    trajectory_b : torch.Tensor
        A 2D tensor representing the second trajectory. Its shape should be (S, D),
        where S is the number of time steps and D is the number of dimensions.

    Returns
    -------
    float
        The average displacement error between the two trajectories, computed as the
        mean L2 norm of the element-wise difference between the two trajectories.

    Raises
    ------
    AssertionError
        If either of the input tensors is not a 2D tensor, or if they do not have
        the same shape.

    Examples
    --------
    >>> trajectory_a = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    >>> trajectory_b = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    >>> average_displacement_error(trajectory_a, trajectory_b)
    1.4142135381698608
    """
    assert trajectory_a.ndim == trajectory_b.ndim == 2, "Input tensors must be 2D tensors"
    assert trajectory_a.shape == trajectory_b.shape, "Input tensors must have the same shape"
    return (trajectory_a - trajectory_b).norm(dim=1).mean().item()


def extract_visible_trajectory(trajectory_a: torch.Tensor, trajectory_b: torch.Tensor,
                               visibility: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extracts the visible portion of two trajectory tensors according to a visibility mask.

    Parameters
    ----------
    trajectory_a : torch.Tensor
        A 2D tensor representing the first trajectory. Its shape should be (S, D),
        where S is the number of time steps and D is the number of dimensions.
    trajectory_b : torch.Tensor
        A 2D tensor representing the second trajectory. Its shape should be (S, D),
        where S is the number of time steps and D is the number of dimensions.
    visibility : torch.Tensor
        A 1D tensor representing the visibility of each time step. Its length should be S.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        A tuple of two 2D tensors representing the visible portion of the input trajectories.
        The output tensor shapes are (N, D), where N is the number of visible time steps
        and D is the number of dimensions.

    Examples
    --------
    >>> trajectory_a = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    >>> trajectory_b = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    >>> visibility = torch.tensor([1, 0, 1])
    >>> extract_visible_trajectory_chain(trajectory_a, trajectory_b, visibility)
    (tensor([[0., 0.], [2., 2.]]), tensor([[1., 1.], [3., 3.]]))
    """
    return trajectory_a[visibility == 1], trajectory_b[visibility == 1]


def extract_visible_trajectory_chain(
        trajectory_a: torch.Tensor,
        trajectory_b: torch.Tensor,
        visibility: torch.Tensor,
        query_point: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extracts the visible portion of two trajectory tensors around the query point.

    Parameters
    ----------
    trajectory_a : torch.Tensor
        A 2D tensor representing the first trajectory. Its shape should be (S, D),
        where S is the number of time steps and D is the number of dimensions.
    trajectory_b : torch.Tensor
        A 2D tensor representing the second trajectory. Its shape should be (S, D),
        where S is the number of time steps and D is the number of dimensions.
    visibility : torch.Tensor
        A 1D tensor representing the visibility of each time step. Its length should be S.
    query_point : torch.Tensor
        A 1D tensor representing the query point, (t,x,y). Its shape should be (D+1,).

    Returns
    -------
    tuple
        A tuple of two 2D tensors representing the visible portion of the input trajectories.
        If the entire trajectories are visible, this is just the original input trajectories.

    Raises
    ------
    AssertionError
        If the visibility tensor is not a 1D tensor, or if it does not have the same length as
        the input trajectories.

    Examples
    --------
    >>> trajectory_a = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    >>> trajectory_b = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    >>> visibility = torch.tensor([1, 1, 0])
    >>> query_point = torch.tensor([0.0, 0.0, 0.0])
    >>> extract_visible_trajectory_chain(trajectory_a, trajectory_b, visibility, query_point)
    (tensor([[0., 0.], [1., 1.]]), tensor([[1., 1.], [2., 2.]]))
    """
    assert visibility.ndim == 1, "Visibility tensor must be a 1D tensor"
    assert len(visibility) == len(trajectory_a) == len(trajectory_b), "Input tensors must have the same length"

    t = query_point[0].item()
    t = int(t)
    assert visibility[t] == 1, "Query point must be visible"

    occluded_indices = (visibility == 0).nonzero()

    occluded_indices_after_query_point = occluded_indices[occluded_indices > t]
    if len(occluded_indices_after_query_point) > 0:
        idx = occluded_indices_after_query_point[0].item()
        trajectory_a = trajectory_a[:idx]
        trajectory_b = trajectory_b[:idx]

    occluded_indices_until_query_point = occluded_indices[occluded_indices < t]
    if len(occluded_indices_until_query_point) > 0:
        idx = occluded_indices_until_query_point[-1].item()
        trajectory_a = trajectory_a[idx + 1:]
        trajectory_b = trajectory_b[idx + 1:]

    return trajectory_a, trajectory_b


def compute_summary(results: Dict, query_mode='first') -> Dict:
    """
    Computes a summary of the trajectory prediction results.

    Parameters
    ----------
    results : Dict
        A dictionary containing the trajectory prediction results. It should have the following keys:
        - 'trajectory_gt': A 2D tensor representing the ground-truth trajectory. Its shape should be (S, D),
          where S is the number of time steps and D is the number of dimensions.
        - 'trajectory_pred': A 2D tensor representing the predicted trajectory. Its shape should be (S, D),
          where S is the number of time steps and D is the number of dimensions.
        - 'visibility_gt': A 1D tensor of length S representing the visibility of each time step.
        - 'visibility_pred': A 1D tensor of length S representing the predicted visibility of each time step.

    Returns
    -------
    Dict
        A dictionary containing the computed summary statistics, with the following keys:
        - 'idx': A string representing the trajectory index, in the format "<iter>--<video_idx>--<point_idx_in_video>".
        - 'ade': The average displacement error between the ground-truth and predicted trajectories.
        - 'ade_visible': The average displacement error between the visible portion
           of the ground-truth and predicted trajectories.
        - 'ade_visible_chain': The average displacement error between the first visible chain
           of the ground-truth and predicted trajectories.
        - 'n_timesteps': The length of the trajectory.
        - 'n_timesteps_visible': The number of visible points in the ground-truth trajectory.
        - 'n_timesteps_visible_chain': The number of visible points in the ground-truth trajectory, assuming a chain structure.
        - 'occlusion_accuracy': The ratio of correctly predicted point visibilities
        - 'jaccard_i': TODO Define, for i in [1,2,4,8,16]
        - 'average_jaccard': TODO Define
        - 'pts_within_i': PCK with a threshold of 'i' that does not scale the threshold relative to a body
           (e.g., a human), but measures in pixels. Computed for i in [1, 2, 4, 8, 16]
        - 'average_pts_within_thresh': The average of 'pts_within_i', for all i.

    Examples
    --------
    >>> results = {
    ...     'iter': 123,
    ...     'video_idx': 2,
    ...     'point_idx_in_video': 31,
    ...     'trajectory_gt': torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]),
    ...     'trajectory_pred': torch.tensor([[0.0, 0.0], [2.0, 2.0], [3.0, 3.0]]),
    ...     'visibility_gt': torch.tensor([True, True, False])
    ...     'visibility_pred': torch.tensor([True, True, True]),
    ... }
    >>> compute_summary(results)
    {
        'idx': '123--2--31',
        'ade': 0.9428090453147888,
        'ade_visible': 0.7071067690849304,
        'ade_visible_chain': 0.7071067690849304,
        'n_timesteps': 3,
        'n_timesteps_visible': 2,
        'n_timesteps_visible_chain': 2,
        'occlusion_accuracy': 0.5,
        'jaccard_1': 0.0,
        'jaccard_2': 0.5,
        'jaccard_4': 0.5,
        'jaccard_8': 0.5,
        'jaccard_16': 0.5,
        'average_jaccard': 0.4,
        'pts_within_1': 0.0,
        'pts_within_2': 1.0,
        'pts_within_4': 1.0,
        'pts_within_8': 1.0,
        'pts_within_16': 1.0,
        'average_pts_within_thresh': 0.8
    }
    """
    traj_gt = results["trajectory_gt"]
    traj_pred = results["trajectory_pred"]
    vis_gt = results["visibility_gt"]
    query_point = results["query_point"]

    traj_gt_visible, traj_pred_visible = extract_visible_trajectory(traj_gt, traj_pred, vis_gt)
    traj_gt_visible_chain, traj_pred_visible_chain = extract_visible_trajectory_chain(traj_gt, traj_pred, vis_gt,
                                                                                      query_point)

    assert vis_gt.sum() == len(traj_gt_visible)
    assert vis_gt.sum() >= len(traj_gt_visible_chain)

    summary = {
        "idx": f'{results["iter"]}--{results["video_idx"]}--{results["point_idx_in_video"]}',
        "ade": average_displacement_error(traj_gt, traj_pred),
        "ade_visible": average_displacement_error(traj_gt_visible, traj_pred_visible),
        "ade_visible_chain": average_displacement_error(traj_gt_visible_chain, traj_pred_visible_chain),
        "n_timesteps": len(traj_gt),
        "n_timesteps_visible": len(traj_gt_visible),
        "n_timesteps_visible_chain": len(traj_gt_visible_chain),
    }

    # TODO ad hoc fix for results saved to disk without the visibility_pred field
    if "visibility_pred" not in results:
        warnings.warn("The 'visibility_pred' field is missing from the results for idx={idx}. "
                      "It will be set to all ones.")
        results["visibility_pred"] = torch.ones_like(results["visibility_gt"])

    # TODO Rescale the trajectories of all datasets (e.g., FLT++) to 256x256, as suggested in the TAP-Vid paper
    from mmpt.datasets.tapvid_evaluation_datasets import compute_tapvid_metrics
    tapvid_metrics = compute_tapvid_metrics(
        query_points=results["query_point"][None, None, :].numpy(),
        gt_occluded=~results["visibility_gt"][None, None, :].numpy(),
        gt_tracks=results["trajectory_gt"][None, None, :, :].numpy(),
        pred_occluded=~results["visibility_pred"][None, None, :].numpy(),
        pred_tracks=results["trajectory_pred"][None, None, :, :].numpy(),
        query_mode=query_mode,
        additional_pck_thresholds=[
            0.01, 0.05,
            *[0.1 * (i + 1) for i in range(10)],
            *[(i + 1) for i in range(10)],
            # 15, 20, 50, 100, 256,
        ],
    )
    tapvid_metrics = {k: v.item() * 100 for k, v in tapvid_metrics.items()}

    summary.update(tapvid_metrics)
    return summary


def compute_summary_df(results_list: List[Dict]) -> pd.DataFrame:
    """
    Computes a summary dataframe of the results of multiple trajectory prediction experiments.

    Parameters:
        results_list (List[Dict]): a list of dictionaries, where each dictionary has the same format as the dictionary
                                   returned by the compute_summary function.

    Returns:
        summary_df (pd.DataFrame): a dataframe containing the metrics returned by <see cref="compute_summary">,
                                   with the metrics being the columns (e.g., 'idx' (str), 'ade' (float),
                                   'n_timesteps' (int), etc.)
    """
    summaries = []
    for results in results_list:
        summaries += [compute_summary(results)]
    return pd.DataFrame.from_records(summaries)


def figure1(
        df: pd.DataFrame,
        output_path: str,
        log_y: bool = False,
        save_pdf: bool = True,
        name: str = "figure1",
        title: str = rf"ADE per visible chain length (w/ 95\% CI)",
        scale: float = 1.0,
) -> None:
    df = df.copy()
    df = df[df.n_timesteps_visible_chain > 1]

    fig, ax = plt.subplots(figsize=(6 * scale, 4 * scale))
    fig.suptitle(title)
    sns.lineplot(
        df,
        x="n_timesteps_visible_chain",
        y="ade_visible_chain",
        hue="name",
        palette=cycle(["GoldenRod", "r", "forestgreen", "yellow"]),
        linestyle="-",
        linewidth=2,
        errorbar=("ci", 95),
        markers=True,
        dashes=False,
        err_style="band",
        # err_style="bars", err_kws={"fmt": 'o', "linewidth": 2, "capsize": 6},
        alpha=1,
        ax=ax,
    )
    ax.set_xlabel(rf"visibility chain length")
    ax.set_ylabel(rf"ADE")
    if log_y:
        plt.yscale('log')
    ax.legend_.set_title(None)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"{name}.png"))
    if save_pdf:
        plt.savefig(os.path.join(output_path, f"PDF__{name}.pdf"))
    plt.show()
    plt.close()
    plt.clf()


def figure2(
        df: pd.DataFrame,
        output_path: str,
        ade_metric: str = "ade_visible",
        log_y: bool = False,
        save_pdf: bool = True,
        name: str = "figure2",
        title: str = rf"ADE for mostly visible (w/ 95\% CI)"
) -> None:
    df = df.copy()
    df_list = []
    for mostly_visible_threshold in range(1, int(df.n_timesteps_visible.max()) + 1):
        df_ = df.copy()
        df_["mostly_visible_threshold"] = mostly_visible_threshold
        df_["mostly_visible"] = (df_.n_timesteps_visible >= mostly_visible_threshold).apply(
            lambda x: "Mostly Visible" if x else "Mostly Occluded")
        df_list += [df_]

        df_ = df.copy()
        df_["mostly_visible_threshold"] = mostly_visible_threshold
        df_["mostly_visible"] = "All"
        df_list += [df_]

    df = pd.concat(df_list).reset_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(title)
    mostly_visible_types = ["Mostly Visible", "Mostly Occluded", "All"]
    mostly_visible_linestlye = ["--", ":", "-"]
    for mostly_visible, linestyle in zip(mostly_visible_types, mostly_visible_linestlye, strict=True):
        sns.lineplot(
            df[df.mostly_visible == mostly_visible],
            x="mostly_visible_threshold",
            y=ade_metric,
            hue="name",
            palette=cycle(["GoldenRod", "r", "forestgreen", "yellow"]),
            linestyle=linestyle,
            linewidth=2,
            errorbar=("ci", 95),
            markers=True,
            dashes=False,
            err_style="band",
            alpha=1,
            legend=mostly_visible == "All",
            ax=ax,
        )
    texts = [t.get_text() for t in ax.get_legend().get_texts()] + mostly_visible_types
    lines = ax.get_legend().get_lines() + [Line2D([0, 10], [0, 10], linewidth=2, color="black", linestyle=linestyle)
                                           for linestyle in mostly_visible_linestlye]
    new_legend = plt.legend(lines, texts, loc="center left")
    ax.add_artist(new_legend)
    ax.set_xlabel(rf"mostly visible threshold")
    ax.set_ylabel(rf"ADE")
    if log_y:
        plt.yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"{name}_{ade_metric}.png"))
    if save_pdf:
        plt.savefig(os.path.join(output_path, f"PDF__{name}_{ade_metric}.pdf"))
    plt.show()
    plt.close()
    plt.clf()


def figure3(
        df: pd.DataFrame,
        output_path: str,
        log_y: bool = False,
        save_pdf: bool = True,
        name: str = "figure3",
        title: str = rf"ADE per number of visible points (w/ 95\% CI)",
        scale: float = 1.0,
) -> None:
    df = df.copy()
    fig, ax = plt.subplots(figsize=(6 * scale, 4 * scale))
    fig.suptitle(title)
    MetricLabelStyle = namedtuple("MetricLabelStle", ["metric", "label", "style"])
    metric_label_style_list = [MetricLabelStyle("ade", "ADE", "-"), MetricLabelStyle("ade_visible", "ADE Visible", ":")]
    for metric_label_style in metric_label_style_list:
        sns.lineplot(
            df,
            x="n_timesteps_visible",
            y=metric_label_style.metric,
            hue="name",
            palette=cycle(["GoldenRod", "r", "forestgreen", "yellow"]),
            linestyle=metric_label_style.style,
            linewidth=2,
            errorbar=("ci", 95),
            markers=True,
            dashes=False,
            err_style="band",
            alpha=1,
            legend=metric_label_style.metric == "ade",
            ax=ax,
        )
    texts = [t.get_text() for t in ax.get_legend().get_texts()] + [mls.label for mls in metric_label_style_list]
    lines = ax.get_legend().get_lines() + [Line2D([0, 10], [0, 10], linewidth=2, color="black", linestyle=mls.style)
                                           for mls in metric_label_style_list]
    new_legend = plt.legend(lines, texts, loc="center left")
    ax.add_artist(new_legend)
    ax.set_xlabel(rf"number of visible points")
    ax.set_ylabel(rf"ADE")
    if log_y:
        plt.yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"{name}.png"))
    if save_pdf:
        plt.savefig(os.path.join(output_path, f"PDF__{name}.pdf"))
    plt.show()
    plt.close()
    plt.clf()


def figure4(
        df: pd.DataFrame,
        output_path: str,
        log_y: bool = False,
        save_pdf: bool = True,
        name: str = "figure4",
        title: str = rf"PCK per threshold across scenes (w/ 95\% CI)",
        scale: float = 1.0,
) -> None:
    df = df.copy()

    pts_cols = [c for c in df.columns if c.startswith("pts_within_")]
    df = df[["name"] + pts_cols]
    df = df.melt("name", var_name="threshold", value_name="pts_within_threshold")
    df.threshold = df.threshold.apply(lambda x: float(x.replace("pts_within_", "")))

    fig, ax = plt.subplots(figsize=(6 * scale, 4 * scale))
    fig.suptitle(title)
    sns.lineplot(
        df,
        x="threshold",
        y="pts_within_threshold",
        hue="name",
        palette=cycle(["GoldenRod", "r", "forestgreen", "yellow"]),
        linestyle="-",
        linewidth=2,
        errorbar=("ci", 95),
        markers=True,
        dashes=False,
        err_style="bars",
        alpha=1,
        legend=True,
        ax=ax,
    )
    ax.set_xlabel(rf"threshold in pixels")
    ax.set_ylabel(rf"PCK")
    if log_y:
        plt.yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"{name}.png"))
    if save_pdf:
        plt.savefig(os.path.join(output_path, f"PDF__{name}.pdf"))
    plt.show()
    plt.close()
    plt.clf()


def table1(
        df: pd.DataFrame,
        output_path: str,
        ade_metric: str = "ade_visible",
        mostly_visible_threshold: int = 4,
        name: str = "table1"
) -> None:
    mostly_visible_indices = df.n_timesteps_visible >= mostly_visible_threshold
    mostly_visible_ade_df = df[mostly_visible_indices][["name", ade_metric]].groupby("name").mean()
    mostly_occluded_ade_df = df[~mostly_visible_indices][["name", ade_metric]].groupby("name").mean().rename(
        columns={ade_metric: ade_metric + "_occluded"})
    table_df = pd.merge(mostly_visible_ade_df, mostly_occluded_ade_df, left_index=True, right_index=True)
    table_df = table_df.sort_values(ade_metric, ascending=False)
    table_df.to_csv(os.path.join(output_path, f"{name}_threshold-{mostly_visible_threshold}_metric-{ade_metric}.csv"))
    print(f"TABLE: '{name}' (metric={ade_metric})")
    print(table_df)
    print()


def table2(
        df: pd.DataFrame,
        output_path: str,
        mostly_visible_threshold: int = 4,
        add_legacy_metrics: bool = False,
        add_tapvid_metrics: bool = True,
        create_heatmap: bool = True,
        name: str = "table2-selected-metrics"
) -> None:
    df = df.copy()
    df_list = []

    df_global_ade = df[["name", "ade", "ade_visible", "ade_occluded", "ade_visible_chain"]].groupby(["name"]).mean()
    df_list += [df_global_ade]

    for chain_length in [2, 4, 8]:
        if chain_length not in df.n_timesteps_visible_chain.unique():
            continue
        df_chain_ade = df[df.n_timesteps_visible_chain == chain_length][["name", "ade_visible_chain"]].groupby(
            ["name"]).mean().rename(columns={"ade_visible_chain": f"ade_visible_chain_{chain_length}"})
        df_list += [df_chain_ade]

    # Legacy metrics (reported in paper for FlyingThings++ with threshold 4 and for CroHD with threshold 8):
    #    1. ADE of Mostly Visible Trajectories (reported as "Vis." in paper)
    #    2. ADE of Mostly Occluded Trajectories (reported as "Occ." in paper)
    # These legacy metrics were computed as the average of pre-iteration metrics
    if add_legacy_metrics:
        df["iter"] = df.idx.apply(lambda x: int(x.split("--")[0]))
        df["mostly_visible"] = df.n_timesteps_visible >= mostly_visible_threshold
        df_average_per_iter = df.groupby(["iter", "mostly_visible", "name"]).mean().reset_index()
        df_mostly_visible_ade = df_average_per_iter[df_average_per_iter.mostly_visible][["name", "ade"]].groupby(
            "name").mean().rename(columns={"ade": "ade_mostly_visible"})
        df_mostly_occluded_ade = df_average_per_iter[~df_average_per_iter.mostly_visible][["name", "ade"]].groupby(
            "name").mean().rename(columns={"ade": "ade_mostly_occluded"})
        df_list += [df_mostly_visible_ade, df_mostly_occluded_ade]

    if add_tapvid_metrics:
        df_tapvid_metrics = df[[
            "name", "dataset", "model", "jaccard_1", "jaccard_2", "jaccard_4", "jaccard_8", "jaccard_16",
            "pts_within_0.01", "pts_within_0.1", "pts_within_0.5",
            "pts_within_1", "pts_within_2", "pts_within_4", "pts_within_8", "pts_within_16",
            "occlusion_accuracy", "average_jaccard", "average_pts_within_thresh",
        ]].groupby(["name", "dataset", "model"]).mean()
        df_list += [df_tapvid_metrics]

        df[["name", "dataset", "model", "jaccard_1"]].groupby(["name", "dataset", "model"]).mean()

    table_df = df_list[0]
    for df_i in df_list[1:]:
        table_df = pd.merge(table_df, df_i, left_index=True, right_index=True)
        assert len(table_df) == len(df_list[0])

    table_df.reset_index(inplace=True)
    table_df.set_index(["dataset", "model"], inplace=True)
    table_df = table_df.sort_values("name", ascending=True)
    table_df.drop(columns=["name"], inplace=True)
    table_df = table_df.transpose()

    table_df.to_csv(os.path.join(output_path, f"{name}_threshold-{mostly_visible_threshold}.csv"))

    print(f"TABLE: '{name}'")
    print(table_df.to_markdown())
    print()

    if create_heatmap:
        fig, ax = plt.subplots(figsize=(6 + 1 * len(table_df), 12))
        fig.suptitle(name)
        sns.heatmap(table_df, annot=True, linewidths=0.3, fmt=".2f", norm=LogNorm(vmin=1, vmax=100))
        fig.autofmt_xdate()
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f"{name}.png"))
        plt.show()
        plt.close()
        plt.clf()


def table3(
        df: pd.DataFrame,
        output_path: str,
        create_heatmap: bool = False,
        name: str = "table3-pck"
) -> None:
    df = df.copy()
    df["iter"] = df.idx.apply(lambda x: x.split("--")[0])
    table_df = df[[
        "iter", "name", "dataset", "model",
        "ade", "ade_visible", "ade_occluded", "ade_visible_chain",
        "jaccard_1", "jaccard_2", "jaccard_4", "jaccard_8", "jaccard_16",
        "pts_within_0.01", "pts_within_0.1", "pts_within_0.5",
        "pts_within_1", "pts_within_2", "pts_within_4", "pts_within_8", "pts_within_16",
        "average_jaccard", "average_pts_within_thresh", "occlusion_accuracy",
    ]]
    table_df = table_df.groupby(["name", "dataset", "model", "iter"]).mean()
    table_df = table_df.groupby(["name", "dataset", "model"]).mean()
    table_df = table_df.sort_values("name", ascending=True)
    table_df = table_df.transpose()

    table_df.to_csv(os.path.join(output_path, f"{name}.csv"))
    print(f"TABLE: '{name}'")
    print(table_df.to_markdown())

    if create_heatmap:
        fig, ax = plt.subplots(figsize=(9, 3 + 1 * len(table_df)))
        fig.suptitle(name)
        sns.heatmap(table_df, annot=True, linewidths=0.3, fmt=".2f", norm=LogNorm(vmin=1, vmax=100))
        fig.autofmt_xdate()
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f"{name}.png"))
        plt.show()
        plt.close()
        plt.clf()
        
    result = {}
    result['average_jaccard'] = table_df.iat[-3,0]
    result['average_pts_within_thresh'] = table_df.iat[-2,0]
    result['occlusion_accuracy'] = table_df.iat[-1,0]

    return result

def ad_hoc_ade_fix(df):
    # TODO Ad hoc fix: ADE only for non-query points
    df.ade = df.ade * df.n_timesteps / (df.n_timesteps - 1)
    df.ade_visible = df.ade_visible * df.n_timesteps_visible / (df.n_timesteps_visible - 1)
    df.ade_visible_chain = df.ade_visible_chain * df.n_timesteps_visible_chain / (df.n_timesteps_visible_chain - 1)


def add_ade_occluded_to_df(df):
    df["ade_occluded"] = (df.ade * df.n_timesteps - df.ade_visible * df.n_timesteps_visible) / (
            df.n_timesteps - df.n_timesteps_visible)


def make_figures(df, output_path, mostly_visible_threshold):
    df = df.copy()
    # table1(df, output_path, "ade", name="withquery__table1")
    # table1(df, output_path, "ade_visible", name="withquery__table1")
    # figure1(df, output_path, name="withquery__figure1")
    # figure2(df, output_path, "ade", name="withquery__figure2")
    # figure2(df, output_path, "ade_visible", name="withquery__figure2")
    # figure3(df, output_path, name="withquery__figure3")

    # ad_hoc_ade_fix(df)
    add_ade_occluded_to_df(df)

    # table1(df, output_path, "ade", name="table1")
    # table1(df, output_path, "ade_visible", name="table1")
    # table2(df, output_path, mostly_visible_threshold, name="table2-selected-metrics")
    result = table3(df, output_path, name="table3-pck-metrics")

    # figure1(df, output_path, name="figure1", scale=3)
    # figure1(df, output_path, name="log__figure1")
    # figure2(df, output_path, "ade", name="figure2")
    # figure2(df, output_path, "ade_visible", name="figure2")
    # figure2(df, output_path, "ade", log_y=True, name="log__figure2")
    # figure2(df, output_path, "ade_visible", log_y=True, name="log__figure2")
    # figure3(df, output_path, name="figure3", scale=3)
    # figure4(df, output_path, name="figure4", scale=3)

    print(f"Done. Figures saved to: {output_path}")
    
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_path_list", nargs='+', required=True,
                        help="List of paths that point to either a precomputed dataframe stored in a CSV file or a "
                             "raw evaluation result in a pickle file containing information such as trajectories and "
                             "visibilities. CSV files are faster as they do not require any additional computation, "
                             "while pickle files are generally larger and need to be computed to obtain the "
                             "dataframe. For consistency in the computation of metrics, it is suggested to use either "
                             "all pickle files or all dataframes.")
    parser.add_argument("--mostly_visible_threshold", type=int, default=4,
                        help='Threshold used to define when a trajectory is "mostly visible" '
                             'compared to being "mostly occluded". The trajectory is mostly visible '
                             'if there is a number of visible points greater or equal to the threshold. '
                             'Otherwise, it is considered as mostly occluded. '
                             'This value was set to 4 for the FlyingThings++ dataset and to 8 for the CroHD dataset, '
                             'in the numbers reported in the paper')
    parser.add_argument("--output_path", type=str, default=f"logs/figures/{get_str_formatted_time()}",
                        help="path to folder to save gifs to")
    args = parser.parse_args()

    assert len(args.results_path_list) > 0, "At least one result path must be provided"
    ensure_dir(args.output_path)

    results_df_list = []
    for path_idx, path in enumerate(args.results_path_list):

        # Extract metadata from path
        # Example of path: logs/1_8_None_pips_72_tapvid_davis_first_2023.03.27_06.36.35/results_list.pkl
        # TODO save the metadata to a file and read it from there
        try:
            path_fixed = path.replace("rgb_stacking",
                                      "rgb-stacking")  # TODO Ad hoc fix to make split work for rgb_stacking
            path_parts = os.path.basename(os.path.dirname(path_fixed)).split("_")
            batch_size, pips_window, n_points, modeltype, seed, dataset_type, subset, query_mode, date, time = path_parts

            name = f"{modeltype}_{dataset_type}_{subset}"
            dataset = f"{dataset_type}_{subset}" if dataset_type != "tapvid" else f"{subset}".upper()
            model = f"{modeltype}-{pips_window}" if modeltype == "pips" else modeltype

        except ValueError:
            name = os.path.basename(os.path.dirname(path))
            dataset = "unknown"
            model = "unknown"
            query_mode = "unknown"

        if path.endswith(".csv"):
            df = pd.read_csv(path)
        elif path.endswith(".pkl"):
            with open(path, "rb") as f:
                results_list = pickle.load(f)

                print(f"*** Results dictionary for the first datapoint of {name}:")
                for k, v in results_list[0].items():
                    print(f"{k}: {v}")
                print()
                print(f"*** Summary metrics for the first datapoint of {name}:")
                for k, v in compute_summary(results_list[0]).items():
                    print(f"{k}: {v}")
                print()

                print(f"Recomputing summary for {name}...")
                df = compute_summary_df(results_list)

                for results in results_list:
                    if results["query_point"][0] != 0:
                        print(f"*** Results dictionary for the first datapoint with a non-first query, for {name}:")
                        for k, v in results.items():
                            print(f"{k}: {v}")
                        print()
                        print(f"*** Summary metrics for the first datapoint with a non-first query, for {name}:")
                        for k, v in compute_summary(results).items():
                            print(f"{k}: {v}")
                        print()
                        break

        else:
            raise ValueError(f"The results path provided does not point to a `.csv` or `.pkl` file. "
                             f"The given path was: `{path}`")

        df["name"] = f"{path_idx}__{name}"
        df["model"] = model
        df["dataset"] = dataset
        df["query_mode"] = query_mode

        print(f"Loaded results df with name `{name}` from path `{path}`.")
        print(df.describe().transpose())
        print()
        results_df_list += [df]

    df = pd.concat(results_df_list)
    make_figures(df, args.output_path, args.mostly_visible_threshold)