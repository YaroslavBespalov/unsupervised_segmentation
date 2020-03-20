from argparse import ArgumentParser


class GanParameters(ArgumentParser):

    def __init__(self, *args, **kwargs):
        super().__init__(add_help=False, *args, **kwargs)
        self.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
        self.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
        self.add_argument('--L1', type=float, default=2, help='L1 loss weight')
        self.add_argument('--noise_size', type=float, default=50)

