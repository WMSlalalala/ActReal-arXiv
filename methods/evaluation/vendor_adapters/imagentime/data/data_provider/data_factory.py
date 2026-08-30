"""The ETT loader `utils/utils_data.py` imports but this work never reaches.

The repository ships `data/` as an empty package -- the corpora are distributed
separately -- yet `utils/utils_data.py` imports this name at module scope, so
nothing in the repo can be imported at all until it exists.  Supplying it here
keeps that import satisfied without editing the authors' file.

It is deliberately not implemented.  `data_provider` is reached only from the
`ETTh1/ETTh2/ETTm1/ETTm2` branch of `gen_dataloader`, and HMOG windows go
through the `fred_md` branch into `data/long_range.py`.  If this ever runs, the
dataset name was wrong and the run should stop rather than quietly load
something else.
"""


def data_provider(args, flag):
    raise NotImplementedError(
        "data_provider serves the ETT datasets only; HMOG windows are routed "
        "through data/long_range.py via the fred_md branch of gen_dataloader"
    )
