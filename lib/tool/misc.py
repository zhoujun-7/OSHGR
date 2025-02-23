import os
import sys
import logging
import time
import torch
import random
import numpy as np
import types
from natsort import natsorted


def ddp_info(self, s):
    if self.RANK == 0:
        self.info(s)


def setup_DDP_logger(final_output_dir, rank=0):
    time_str = time.strftime("%Y-%m-%d---%H-%M-%S")
    phase = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    log_file = "{}/{}/log.log".format(phase, time_str, rank)
    final_log_file = os.path.join(final_output_dir, log_file)
    final_log_dir = os.path.dirname(final_log_file)
    head = "%(asctime)-15s %(message)s"
    # logging.basicConfig(format=head)
    if rank == 0:
        os.makedirs(os.path.dirname(final_log_file), exist_ok=True)
        logging.basicConfig(filename=str(final_log_file), format=head)
    else:
        logging.basicConfig(format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger("").addHandler(console)
    logger.RANK = rank
    logger.ddp_info = types.MethodType(ddp_info, logger)
    return logger, time_str, final_log_dir


def count_acc(logits, label):
    m = label < 1000
    if m.sum() > 0 and isinstance(logits, torch.Tensor):
        logits_ = logits[m]
        label_ = label[m]
        pred = torch.argmax(logits_, dim=1)
        if torch.cuda.is_available():
            return (pred == label_).type(torch.cuda.FloatTensor).mean().item() * 100
        else:
            return (pred == label_).type(torch.FloatTensor).mean().item() * 100
    else:
        return 0


def count_acc_with_multi_proto(logits, label):
    m = label < 1000
    if m.sum() > 0:
        logits_ = logits[m]
        label_ = label[m]
        pred = torch.argmax(logits_, dim=1)

        n_cls = label_.max() + 1
        pred = pred % n_cls

        if torch.cuda.is_available():
            return (pred == label_).type(torch.cuda.FloatTensor).mean().item() * 100
        else:
            return (pred == label_).type(torch.FloatTensor).mean().item() * 100
    else:
        return 0


def set_seed(seed):
    if seed == 0:
        print(" random seed")
        torch.backends.cudnn.benchmark = True
    else:
        print("manual seed:", seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def instance_to_dict(instance):
    dt = {}
    for k in dir(instance):
        if k[:2] == "__" and k[-2:] == "__":
            continue

        is_save = False
        if str(type(getattr(instance, k))) == "<class 'builtin_function_or_method'>":
            continue

        if isinstance(getattr(instance, k), int):
            is_save = True
        elif isinstance(getattr(instance, k), float):
            is_save = True
        elif isinstance(getattr(instance, k), list):
            is_save = True
        elif isinstance(getattr(instance, k), tuple):
            is_save = True
        elif isinstance(getattr(instance, k), dict):
            is_save = True
        elif isinstance(getattr(instance, k), set):
            is_save = True
        elif isinstance(getattr(instance, k), str):
            is_save = True
        elif isinstance(getattr(instance, k), np.ndarray):
            is_save = True
        elif isinstance(getattr(instance, k), torch.Tensor):
            is_save = True
        else:
            pass

        if is_save:
            dt[k] = getattr(instance, k)
        else:
            dt[k] = instance_to_dict(getattr(instance, k))
    return dt


class MultiLoopTimeCounter:
    def __init__(self, loop_times_ls):
        self.loop_times_np = np.array(loop_times_ls)
        self.equal_min_loop_np = np.cumprod(self.loop_times_np[::-1])[::-1]
        self.accum_min_loop = 0
        self.time_start = time.time()

    def step(self):
        current_time = time.time()
        self.accum_min_loop += 1
        avg_time_min_loop = (current_time - self.time_start) / self.accum_min_loop
        self.time_all_loop_np = self.equal_min_loop_np * avg_time_min_loop
        self.time_avg_loop_np = self.time_all_loop_np / self.loop_times_np
        self.rem_time_cur_loop_np = (self.accum_min_loop % self.equal_min_loop_np) * avg_time_min_loop
        return self.time_all_loop_np, self.rem_time_cur_loop_np, self.time_avg_loop_np

    def read(self):
        return self.time_all_loop_np, self.rem_time_cur_loop_np, self.time_avg_loop_np


def search_files(root_dir, file_ext_ls=[]):
    f_ls = natsorted(os.listdir(root_dir))
    f_ls = [os.path.join(root_dir, f) for f in f_ls]

    all_f_ls = []
    for f in f_ls:
        if os.path.isdir(f):
            ff_ls = search_files(f, file_ext_ls)
            all_f_ls.extend(ff_ls)
        else:
            n, ext = os.path.splitext(f)
            if isinstance(file_ext_ls, list):
                if ext in file_ext_ls:
                    all_f_ls.append(f)
            elif file_ext_ls == "any":
                all_f_ls.append(f)
    return all_f_ls
