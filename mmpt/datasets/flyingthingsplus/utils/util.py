import math
import pathlib
import random
from datetime import datetime
from typing import List, Any

import numpy as np
import torch
import wandb
from torch.optim.lr_scheduler import LambdaLR


class Object(object):
    """
    Empty class for use as a default placeholder object.
    """
    pass


def get_str_formatted_time() -> str:
    """
    Returns the current time in the format of '%Y.%m.%d_%H.%M.%S'.
    Returns
    -------
    str
        The current time
    """
    return datetime.now().strftime('%Y.%m.%d_%H.%M.%S')


HORSE = """               .,,.
             ,;;*;;;;,
            .-'``;-');;.
           /'  .-.  /*;;
         .'    \\d    \\;;               .;;;,
        / o      `    \\;    ,__.     ,;*;;;*;,
        \\__, _.__,'   \\_.-') __)--.;;;;;*;;;;,
         `""`;;;\\       /-')_) __)  `\' ';;;;;;
            ;*;;;        -') `)_)  |\\ |  ;;;;*;
            ;;;;|        `---`    O | | ;;*;;;
            *;*;\\|                 O  / ;;;;;*
           ;;;;;/|    .-------\\      / ;*;;;;;
          ;;;*;/ \\    |        '.   (`. ;;;*;;;
          ;pip;'. ;   |          )   \\ | ;;;;;;
          ,;*;s;;\\/   |.        /   /` | ';;;*;
           ;;;**;/    |/       /   /__/   ';;;
           '*;;;/     |       /    |      ;*;
                `""""`        `""""`     ;'"""


def nice_print(msg, last=False):
    """
    Print a message in a nice format.
    Parameters
    ----------
    msg : str
        The message to be printed
    last : bool, optional
        Whether to print a blank line at the end, by default False
    Returns
    -------
    None
    """
    print()
    print("\033[0;35m" + msg + "\033[0m")
    if last:
        print()


class AttrDict(dict):
    """
    A dictionary class that can be accessed with attributes.
    Note that the dictionary keys must be strings and
    follow attribute naming rules to be accessible as attributes,
    e.g., the key "123xyz" will give a syntax error.
    Usage:
    ```
    x = AttrDict()
    x["jure"] = "mate"
    print(x.jure)
    # "mate"
    x[123] = "abc"
    x.123
    # SyntaxError: invalid syntax
    ```
    """

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def ensure_dir(dirname):
    """
    Ensure that a directory exists. If it doesn't, create it.
    Parameters
    ----------
    dirname : str or pathlib.Path
        The directory, the existence of which will be ensured
    Returns
    -------
    None
    """
    dirname = pathlib.Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def batchify_list(datapoints: List[Any], batch_size: int, batch_post_processing_fn=None) -> List[Any]:
    """
    Batchify a list of anything into batches of given batch size,
    apply an optional post-processing function on top of each batch.

    Parameters
    ----------
    datapoints: List[Any]
        The list of datapoints.
    batch_size: int
        The size of batches.
    batch_post_processing_fn: function, optional
        An optional post-processing function that will be applied on top of the batches.

    Returns
    -------
    A list of batches. The batch might be anything after post-processing, thus List[Any] is returned.
    """
    assert batch_size > 0
    num_batches = math.ceil(len(datapoints) / batch_size)
    batches = [datapoints[i * batch_size:(i + 1) * batch_size] for i in range(num_batches)]
    assert len(datapoints) == sum([len(batch) for batch in batches])
    if batch_post_processing_fn is not None:
        batches = [batch_post_processing_fn(batch) for batch in batches]
    return batches


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

    Source:
    https://github.com/huggingface/transformers/blob/820c46a707ddd033975bc3b0549eea200e64c7da/src/transformers/optimization.py#L75

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def zip_strict(*lists):
    """
    Zip lists, with strict equality of length.

    Given an arbitrary number of lists, zip_strict ensures that all lists have the same length before returning the
    zipped object.

    Parameters
    ----------
    *lists : list of lists
        An arbitrary number of lists.

    Returns
    -------
    zipped_lists : zip object
        An object representing a sequence of tuples, where the i-th tuple contains the i-th element from each of the
        input lists.

    Raises
    ------
    AssertionError
        If not all input lists have the same length.

    Examples
    --------
    >>> list(zip_strict([1, 2, 3], [4, 5, 6]))
    [(1, 4), (2, 5), (3, 6)]

    >>> list(zip_strict([1, 2, 3], [4, 5, 6], [7, 8, 9]))
    [(1, 4, 7), (2, 5, 8), (3, 6, 9)]

    >>> list(zip_strict([1, 2, 3], [4, 5]))
    AssertionError:
    """
    lengths = [len(_list) for _list in lists]
    assert all([length == lengths[0] for length in lengths]), "All input lists must have the same length."
    return zip(*lists)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_seed_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def setup_wandb(entity, project, experiment):
    timestamp = datetime.now().strftime('%Y.%m.%d_%H.%M.%S')
    if experiment is None:
        experiment = timestamp
    else:
        experiment += f"_{timestamp}"
    wandb.init(entity=entity, project=project, name=experiment)


def log_video_to_wandb(log_key, frames, step=None, fmt="gif", fps=4):
    frames_4d = np.stack(frames, axis=0)
    frames_4d = frames_4d.transpose((0, 3, 1, 2))
    wandb.log({log_key: wandb.Video(frames_4d, format=fmt, fps=fps)}, step=step)