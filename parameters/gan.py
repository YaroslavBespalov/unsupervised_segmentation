from argparse import ArgumentParser


class GanParameters(ArgumentParser):

    def __init__(self, *args, **kwargs):
        super().__init__(add_help=False, *args, **kwargs)
        self.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
        self.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
        self.add_argument('--L1', type=float, default=2, help='L1 loss weight')
        self.add_argument('--noise_size', type=float, default=256)


class MunitParameters(ArgumentParser):

    def __init__(self, *args, **kwargs):
        super().__init__(add_help=False, *args, **kwargs)
        self.add_argument('--input_dim', type=int, default=3)
        self.add_argument('--dim', type=int, default=16)
        self.add_argument('--style_dim', type=int, default=256)
        self.add_argument('--n_downsample', type=int, default=2)
        self.add_argument('--n_res', type=int, default=4)
        self.add_argument('--activ', type=str, default="lrelu")
        self.add_argument('--pad_type', type=str, default="replicate")
        self.add_argument('--norm', type=str, default="none")
        self.add_argument('--mlp_dim', type=int, default=256)
        self.add_argument('--n_layer', type=int, default=8)
        self.add_argument('--num_scales', type=int, default=3)