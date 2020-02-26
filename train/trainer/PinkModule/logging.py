import shutil
import sys


class PinkBlackLogger:
    def __init__(self, fp, stream=sys.stdout):
        self.stream = stream
        self.fp = fp

    def write(self, message):
        self.fp.write(message)
        self.fp.flush()
        self.stream.write(message)

    def flush(self):
        self.stream.flush()


def padding(arg, width, pad=' '):
    if isinstance(arg, float):
        return '{:.6f}'.format(arg).center(width, pad)
    elif isinstance(arg, int):
        return '{:6d}'.format(arg).center(width, pad)
    elif isinstance(arg, str):
        return arg.center(width, pad)
    elif isinstance(arg, tuple):
        if len(arg) != 2:
            raise ValueError('Unknown type: {}'.format(type(arg), arg))
        if not isinstance(arg[1], str):
            raise ValueError('Unknown type: {}'
                             .format(type(arg[1]), arg[1]))
        return padding(arg[0], width, pad=pad)
    else:
        raise ValueError('Unknown type: {}'.format(type(arg), arg))


def print_row(kwarg_list=[], pad=' '):
    len_kwargs = len(kwarg_list)
    term_width = shutil.get_terminal_size().columns
    width = min((term_width - 1 - len_kwargs) * 9 // 10, 150) // len_kwargs
    row = '|{}' * len_kwargs + '|'
    columns = []
    for kwarg in kwarg_list:
        columns.append(padding(kwarg, width, pad=pad))
    print(row.format(*columns))
