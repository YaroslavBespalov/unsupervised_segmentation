from abc import ABC, abstractmethod


class PathProvider(ABC):

    @staticmethod
    @abstractmethod
    def board() -> str: pass

    @staticmethod
    @abstractmethod
    def models() -> str: pass

    @staticmethod
    @abstractmethod
    def data() -> str: pass


class ZhoresPath(PathProvider):

    homa = "/trinity/home/n.buzun"
    ausland = "/gpfs/gpfs0/n.buzun"

    @staticmethod
    def board() -> str:
        return ZhoresPath.homa + "/runs"

    @staticmethod
    def models() -> str:
        return ZhoresPath.homa + "/PycharmProjects/saved"

    @staticmethod
    def data() -> str:
        return ZhoresPath.ausland


class DGXPath(PathProvider):

    homa = "/home/ibespalov/pomoika"
    ausland = "/raid/data"

    @staticmethod
    def board() -> str:
        return DGXPath.homa

    @staticmethod
    def models() -> str:
        return DGXPath.homa

    @staticmethod
    def data() -> str:
        return DGXPath.ausland


class Paths:

    default: PathProvider = DGXPath







