import argparse
import os
import torch
import tqdm

from torch.utils.data import WeightedRandomSampler
from torch.optim.lr_scheduler import MultiStepLR

from dn3.configuratron import ExperimentConfig
from dn3.trainable.experimental import TVector
from dn3.transforms.instance import Deep1010ToEEG
from dn3.transforms.batch import RandomTemporalCrop
from dn3.trainable.processes import StandardClassification, get_label_balance

import numpy as np
from sklearn import neighbors, model_selection
from dn3_ext import load_datasets, create_numpy_formatted_tvectors, WandBLogging

# Since we are doing a lot of loading, this is nice to suppress some tedious information.
# Keep in mind, removing this might help debug data loading problems
import mne
mne.set_log_level(False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pre-trains t-vector model.")
    parser.add_argument('--config', default='configs/pretraining.yml', help="Configuratron file with experiment "
                                                                            "params.")
    parser.add_argument("--hidden-size", default=384, type=int, help="Length of resulting T-Vector.")
    parser.add_argument('--sample-crop', default=0, type=int, help="How many samples to sacrifice for crop and resample"
                                                                   " augmentation.")
    parser.add_argument('--batch-crop-max', default=0.8, type=float, help='The maximum fraction of temporal cropping '
                                                                          'for retrieved batches.')
    parser.add_argument("--log-interval", default=100, type=int, help="How many batches between logging output.")
    parser.add_argument("--checkpoint-dir", default="checkpoints/", help="Directory to store checkpoints. Must exist.")
    parser.add_argument('--median-sampling', action='store_true', help="There is a large imbalance between the least "
                                                                       "and most represented people. This under/over "
                                                                       "samples each person to the median "
                                                                       "representation.")
    parser.add_argument('--lr-drop-milestones', type=float, nargs='+', default=[0.5, 0.75],
                        help="Percentage (between 0 and 1) of total epochs for when to drop the learning rate "
                             "by 0.1.")
    parser.add_argument('--knn-val', default=5, type=int, help="Whether to run KNN on the validation datasets. Where "
                                                               "argument here is `K`.")
    parser.add_argument('--no-save', action='store_true', help="Don't save epoch checkpoints while training (will"
                                                               " still save final model just in case).")
    args = parser.parse_args()
    checkpoint_file = os.path.join(args.checkpoint_dir)
    experiment = ExperimentConfig(args.config)

    training, target_thinkers, validating = load_datasets(experiment, return_validating=True, return_training=True,
                                                          sample_crop=args.sample_crop)

    # Print some dataset metrics
    sample_weights, counts = get_label_balance(training)
    target_count = counts.sum(-1)
    print("Training points/person: mean {:.2f}, min {}, max {}, ".format(target_count.mean(), target_count.min(),
                                                                         target_count.max()))
    args.lr_drop_milestones = [round(experiment.training_params.epochs * min(1.0, max(p, 0)))
                               for p in args.lr_drop_milestones]

    # Connect W and B
    logging_remote = WandBLogging(args.checkpoint_dir)

    # Load TVector model for training thinkers and create process
    t_vectors = TVector(target_thinkers, hidden_size=args.hidden_size, channels=77, ignored_inds=list())

    # Check if restarting job
    state = logging_remote.load_state()
    start_epoch = None
    if state is not None:
        t_vectors.load(os.path.join(args.checkpoint_dir, 't-vectors.pt'))
        start_epoch = state['epoch']
    logging_remote.watch(t_vectors, log_freq=args.log_interval)

    # Decide whether to use LDAM loss to compensate for training label imbalance, or simple resampling.
    if args.median_sampling:
        sampler = WeightedRandomSampler(sample_weights, len(counts) * int(np.median(counts)), replacement=True)
        print("Sampling {} points from each person.".format(np.median(counts)))
    else:
        sampler = None

    process = StandardClassification(t_vectors)
    process.add_batch_transform(RandomTemporalCrop(max_crop_frac=args.batch_crop_max))
    process.set_scheduler(MultiStepLR(process.optimizer, args.lr_drop_milestones))
    process.set_optimizer(torch.optim.Adam(process.parameters(), lr=experiment.training_params.lr,
                                           weight_decay=experiment.training_params.l2))

    def knn_validation(metrics):
        logging_remote.training_callback(metrics)
        if args.knn_val > 0:
            validating.clear_transforms()
            validating.add_transform(Deep1010ToEEG())
            x, extras = create_numpy_formatted_tvectors(validating, t_vectors)
            y = extras['person_ids'].astype(int).real
            scores = list()
            for train, test in model_selection.StratifiedKFold(n_splits=5).split(x, y):
                model = neighbors.KNeighborsClassifier(args.knn_val)
                model.fit(x[train, :], y[train].squeeze())
                score = model.score(x[test, :], y[test].squeeze())
                tqdm.tqdm.write("{:.1%}".format(score))
                scores.append(score)
            logging_remote.validation_callback({'KNN-Val': np.mean(scores)})


    def save_checkpoint(metrics):
        logging_remote.save_state(epoch=metrics['epoch'])
        if not args.no_save:
            t_vectors.save(experiment.t_vector_weights.format(metrics['epoch']))
        knn_validation(metrics)


    process.fit(training, epochs=experiment.training_params.epochs, batch_size=experiment.training_params.batch_size,
                train_log_interval=args.log_interval, epoch_callback=save_checkpoint, retain_best=None, num_workers=4,
                resume_epoch=start_epoch, sampler=sampler, log_callback=knn_validation)

    t_vectors.save(experiment.t_vector_weights.format("end"))


