import argparse

def get_option(parser=argparse.ArgumentParser()):
    """
    为命令行解析器添加参数选项，并返回解析后的参数

    Args:
        parser (argparse.ArgumentParser): 命令行解析器对象，默认为argparse.ArgumentParser()

    Returns:
        argparse.Namespace: 解析后的参数对象
    """
    parser.add_argument('--batch_size', type=int, default=32, help="输入批量大小，默认为64")
    parser.add_argument('--val_batch_size', type=int, default=32, help="输入批量大小，默认为32")
    parser.add_argument('--epochs', type=int, default=100, help='训练的轮数')
    parser.add_argument("--seed", type=int, default=66, help="随机种子")
    parser.add_argument("--class_num", type=int, default=7, help="分类类别数量")
    parser.add_argument("--tensorboard_path", type=str, default="./runs/log", help="TensorBoard日志路径")
    parser.add_argument("--img_size", type=tuple, default=(224, 224), help="输入图像大小")
    parser.add_argument("--show", action="store_true", default=False, help="是否显示参数信息")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoint", help="检查点保存路径")
    parser.add_argument("--log_file", type=str, default="./train_logger.txt", help="训练日志文件路径")
    parser.add_argument("--GPUS", type=int, default=1, help="使用的GPU数量")
    args = parser.parse_args()

    if args.show:
        print(f"批量大小 {args.batch_size}")
        print(f"随机种子 {args.seed}")
        print(f"分类类别数量 {args.class_num}")
        print(f"log路径 {args.log_path}")

    return args



if __name__ == '__main__':
    ops = get_option()
    print(ops)
