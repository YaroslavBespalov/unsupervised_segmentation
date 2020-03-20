from argparse import ArgumentParser

class DeformationParameters(ArgumentParser):

    def __init__(self, *args, **kwargs):
        super().__init__(add_help=False, *args, **kwargs)
        self.add_argument('--ct', type=float, default=0.0001, help='transition loss weight')
        self.add_argument('--ca', type=float, default=0.0002, help='linear deformation loss weight')
        self.add_argument('--cw', type=float, default=0.002, help='non linear deformation loss weight')
