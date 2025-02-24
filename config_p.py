import argparse
import json

class Options():
    def __init__(self):
        self.initialized = False

    def add_arguments_parser(self, parser: argparse.ArgumentParser):
        """Add a set of arguments to the parser.
        Parameters
        ----------
        parser : argparse.ArgumentParser
            Parser to add arguments to.
        Returns
        -------
        parser : argparse.ArgumentParser
            Parser with added arguments.
        """
        parser.add_argument('--batch_size', type=int, default=32, help="input batch size,default = 64")
        parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train for, default=10')
        parser.add_argument("--seed", type=int, default=66, help="random seed")
        parser.add_argument("--class_num", type=int, default=30, help="classification category,default 10")
        parser.add_argument("--log_path", type=str, default="./runs/log")
        parser.add_argument("--img_size", type=tuple, default=(224, 224), help="input image size")
        parser.add_argument("--show", action="store_true", default=False)
        self.initialized = True

        return parser

    def _initialize_options(self):
        """Initialize a namespace that store options.
        Returns
        -------
        opt: argparse.Namespace
            Namespace with options.
        """
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.add_arguments_parser(parser)
            self.parser = parser

        else:
            print("WARNING: Options was already initialized before")

        return self.parser.parse_args()

    def parse(self):
        """Initialize a namespace that store options.
        Returns
        -------
        opt: argparse.Namespace
            Namespace with options.
        """
        opt = self._initialize_options()
        return opt

    def print_options(self, opt: argparse.Namespace):
        """Print all options and the default values (if changed).
        Parameters
        ----------
        opt : argparse.Namespace
            Namespace with options to print.
        """
        # create a new parser with default arguments
        parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser = self.add_arguments_parser(parser)

        message = '----------------- Options ---------------\n'
        for key, value in sorted(vars(opt).items()):
            comment = ''
            default = parser.get_default(key)
            if value != default:
                comment = f'(default {default})'
            key, value = str(key), str(value)
            message += f'{key}: {value} {comment}\n'
        print(message)

    def save_options(self, opt: argparse.Namespace, path: str):
        """Save options to a json file.
        Parameters
        ----------
        opt : argparse.Namespace
            Namespace with options to save.
        path : str
            Path to save the options (.json extension will
            be automatically added at the end if absent).
        """
        if not path.endswith('.json'):
            path += '.json'
        with open(path, 'w') as f:
            f.write(json.dumps(vars(opt), indent=4))

    def load_options(self, path:str):
        """
        从json文件中加载选项

        Args:
            path (str): 加载选项的路径（如果不存在.json扩展名，将会自动添加）

        Returns:
            argparse.Namespace: 加载后的选项命名空间
        """
        if not path.endswith('.json'):
            path += '.json'
        # 初始化一个新的命名空间，使用默认参数
        parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        opt = self.add_arguments_parser(parser).parse_args([])

        variables = json.load(open(path, 'r'))
        for key, value in variables.items():
            setattr(opt, key, value)
        print("------------------------------")
        return opt


if __name__ == '__main__':
    options = Options()
    opt = options.load_options("./options.json")
    if not opt.show:
        options.print_options(opt)