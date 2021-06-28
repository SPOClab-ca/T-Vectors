import argparse
import numpy as np
from pathlib import Path
from dn3.configuratron import ExperimentConfig
from dn3.trainable.experimental import TVector

from dn3_ext import load_datasets, create_numpy_formatted_tvectors


def extract_vectors(args):
    assert args.config is not None
    assert args.model is not None

    experiment = ExperimentConfig(args.config)
    validating = load_datasets(experiment, return_training=False, return_validating=True, only1020=args.limit_channels)

    # Load TVector model, number of targets is currently arbitrary
    t_vectors = TVector(2, hidden_size=args.hidden_size, channels=77, ignored_inds=list()).to('cuda')
    t_vectors.load(args.model)
    t_vectors.train(False)
    if args.session_id:
        validating.update_id_returns(session=True)
    return create_numpy_formatted_tvectors(validating, t_vectors)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Creates T-Vector validation data.")
    parser.add_argument('vectors', help='Location of the extracted vectors, if file is specified and does'
                                                  'not exist, model and config must be specified to extract t-vector '
                                                  'formatted validation data.')
    parser.add_argument('model', help="Filename of saved t-vector model to load.")
    parser.add_argument('--limit-channels', action='store_true', help="Whether to limit the used channels to simply  "
                                                                      "the 10-20 channel set.")
    parser.add_argument('--session-id', action='store_true', help="Whether to also save session_ids")
    parser.add_argument('--config', default='configs/pretraining.yml', help="Configuratron file with experiment "
                                                                            "params.")
    parser.add_argument("--hidden-size", default=384, type=int, help="Length of resulting T-Vector.")
    args = parser.parse_args()

    print('Extracting T-Vectors from validation dataset.')
    t_vectors, extras = extract_vectors(args)
    extras = {c: extras[c].values for c in list(extras.columns)}
    Path(args.vectors).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.vectors, t_vectors=t_vectors, **extras)
