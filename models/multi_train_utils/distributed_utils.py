import os

import torch
import torch.distributed as dist


def init_distributed_mode(args):
    # 判断是否处于分布式模式
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # 获取环境变量RANK和WORLD_SIZE的值，分别赋值给args.rank和args.world_size
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        # 获取环境变量SLURM_PROCID的值，赋值给args.rank
        args.rank = int(os.environ['SLURM_PROCID'])
        # 使用args.rank取模torch.cuda.device_count()的值，赋值给args.gpu
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        # 打印提示信息，表示不使用分布式模式
        print('Not using distributed mode')
        # 设置args.distributed为False
        args.distributed = False
        return

    # 设置args.distributed为True
    args.distributed = True

    # 设置当前进程使用的GPU设备
    torch.cuda.set_device(args.gpu)
    # 设置通信后端为nccl，nvidia GPU推荐使用NCCL
    args.dist_backend = 'nccl'
    # 打印分布式初始化信息
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    # 初始化进程组，使用args.dist_backend作为通信后端，args.dist_url作为初始化方法，args.world_size作为world_size，args.rank作为rank
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)
    # 等待所有进程到达此处
    dist.barrier()



def cleanup():
    dist.destroy_process_group()


def is_dist_avail_and_initialized():
    """检查是否支持分布式环境"""
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def reduce_value(value, average=True):
    world_size = get_world_size()
    if world_size < 2:  # 单GPU的情况
        return value

    with torch.no_grad():
        dist.all_reduce(value)
        if average:
            value /= world_size

        return value
