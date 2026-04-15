import sys

from src import dataframe_sampler as _package
from src.dataframe_sampler import *  # noqa: F401,F403
from src.dataframe_sampler import anonymization as _anonymization
from src.dataframe_sampler import cli as _cli
from src.dataframe_sampler import encoding as _encoding
from src.dataframe_sampler import io as _io
from src.dataframe_sampler import knn as _knn
from src.dataframe_sampler import llm as _llm
from src.dataframe_sampler import metrics as _metrics
from src.dataframe_sampler import neighbours as _neighbours
from src.dataframe_sampler import sampler as _sampler
from src.dataframe_sampler import utils as _utils
from src.dataframe_sampler import vectorizer as _vectorizer
from src.dataframe_sampler.cli import main

__path__ = _package.__path__

sys.modules.setdefault("dataframe_sampler.anonymization", _anonymization)
sys.modules.setdefault("dataframe_sampler.cli", _cli)
sys.modules.setdefault("dataframe_sampler.encoding", _encoding)
sys.modules.setdefault("dataframe_sampler.io", _io)
sys.modules.setdefault("dataframe_sampler.knn", _knn)
sys.modules.setdefault("dataframe_sampler.llm", _llm)
sys.modules.setdefault("dataframe_sampler.metrics", _metrics)
sys.modules.setdefault("dataframe_sampler.neighbours", _neighbours)
sys.modules.setdefault("dataframe_sampler.sampler", _sampler)
sys.modules.setdefault("dataframe_sampler.utils", _utils)
sys.modules.setdefault("dataframe_sampler.vectorizer", _vectorizer)


if __name__ == "__main__":
    main()
