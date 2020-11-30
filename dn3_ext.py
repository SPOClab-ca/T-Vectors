import bisect
import torch
import tqdm
import numpy as np

from pandas import DataFrame

from torch import nn
from torch.utils.data import ConcatDataset, DataLoader

from dn3.transforms.instance import InstanceTransform, CropAndResample, Deep1010ToEEG, UniformTransformSelection
from dn3.transforms.channels import DEEP_1010_CHS_LISTING
from dn3.trainable.experimental import TVector


CH_NAMES_1020 = ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
                 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']


class Only1020EEG(InstanceTransform):

    USED_INDS = [i for i, ch in enumerate(DEEP_1010_CHS_LISTING) if ch in CH_NAMES_1020]

    def __call__(self, x):
        return x[self.USED_INDS, :]

    def new_channels(self, old_channels):
        return len(self.USED_INDS)


class RandomCrop(InstanceTransform):

    def __init__(self, cropped_length):
        """
        Pad the number of samples.

        Parameters
        ----------
        cropped_length : int
                         The cropped sequence length (in samples).
        """
        super().__init__()
        self._new_length = cropped_length

    def __call__(self, x):
        assert x.shape[-1] >= self._new_length
        start_offset = np.random.choice(x.shape[-1] - self._new_length)
        return x[..., start_offset:start_offset + self._new_length]

    def new_sequence_length(self, old_sequence_length):
        return self._new_length


def load_datasets(experiment, return_training=True, return_validating=False, sample_crop=None, only1020=False):
    training = list()
    validating = None
    total_thinkers = 0

    def _load_dataset(name, ds):
        print("Constructing " + name)
        return ds.auto_construct_dataset(return_person_id=True)

    for name, ds in experiment.datasets.items():
        if name != experiment.training_params.validation_dataset and return_training:
            loaded = _load_dataset(name, ds)
            if only1020:
                print("Limiting to: {}".format(CH_NAMES_1020))
                loaded.add_transform(Only1020EEG())
            else:
                loaded.add_transform(Deep1010ToEEG())
            if isinstance(sample_crop, int) and sample_crop > 0:
                loaded.add_transform(UniformTransformSelection([
                    CropAndResample(experiment.global_samples - sample_crop, stdev=sample_crop / 4),
                    RandomCrop(experiment.global_samples - sample_crop),
                    ], weights=[0.25, 0.75])
                )
            total_thinkers += len(loaded.get_thinkers())

            training.append(loaded)

        elif return_validating and name == experiment.training_params.validation_dataset:
            validating = _load_dataset(name, ds)
            validating.add_transform(Deep1010ToEEG())

    returning = list()
    if return_training:
        print("Training TVectors using {} people's data across {} datasets.".format(total_thinkers, len(training)))
        returning.append(PersonIDAggregator(training))
        returning.append(total_thinkers)
    if return_validating:
        returning.append(validating)

    if len(returning) == 1:
        return returning[0]
    return returning


def create_numpy_formatted_ds(dataset, tvector_model: nn.Module, batch_size=128):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    tvectors = list()
    columns = list()
    if dataset.return_person_id:
        columns.append('person_ids')
    if dataset.return_session_id:
        columns.append('session_ids')
    if dataset.get_targets() is not None:
        columns.append('target')

    extras = DataFrame(columns=columns)

    tvector_model.train(False)
    with torch.no_grad():
        for batch in tqdm.tqdm(loader, desc="Creating T-Vectors"):
            batch[0] = batch[0].to('cuda')
            tvectors.append(tvector_model(batch[0])[1].detach().cpu().numpy())
            batch_extras = DataFrame(np.column_stack([b.cpu().numpy().astype(int) for b in batch[1:]]), columns=columns)
            extras = extras.append(batch_extras, ignore_index=True)

    return np.vstack(tvectors), extras


class PersonIDAggregator(ConcatDataset):

    def __init__(self, datasets):
        super().__init__(datasets)
        self.people_offset = self.cumsum([d.get_thinkers() for d in datasets])
        self._val_start = self.people_offset[-1]

    def num_people(self):
        return self.people_offset[-1]

    def get_targets(self):
        i = 0
        targets = list()
        for ds in self.datasets:
            for th in ds.get_thinkers():
                targets += [i] * len(ds.thinkers[th])
                i += 1
        return np.array(targets)

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        fetched = self.datasets[dataset_idx][sample_idx]
        x, ds_p_id = fetched[:2]
        return [x, ds_p_id + bisect.bisect_right(self.people_offset, ds_p_id), *fetched[2:]]


class AsTVector(InstanceTransform):

    def __init__(self, tvector_model, hidden_size=384, cuda=False):
        super().__init__()
        if isinstance(tvector_model, str):
            self.tvector = TVector(hidden_size=hidden_size)
            self.tvector.load(tvector_model, freeze_features=True)
        elif callable(tvector_model):
            self.tvector = tvector_model
        else:
            raise ValueError("Could not interpret tvector_model")

        self.cuda = False
        if isinstance(self.tvector, TVector):
            self.tvector.return_features = True
            if cuda:
                self.tvector.cuda()
                self.cuda = True

    def __call__(self, x):
        if self.cuda:
            x = x.cuda()
        return self.tvector(x)[1].cpu()


class ScoreLabels(InstanceTransform):

    def __init__(self, score_df: DataFrame, ordered_people: list, person_column='Person', score_column='Accuracy',
                 score_threshold=0.82):
        super().__init__(only_trial_data=False)
        self.person_column = person_column
        self.score_column = score_column
        self.score_threshold = score_threshold
        self.score_df = score_df
        self.ordered_people = ordered_people

    def __call__(self, *args):
        args = list(args)
        label = int(self.score_df[self.score_df['Person'] == self.ordered_people[args[1].item()]]
                    [self.score_column].values[0] > self.score_threshold)
        return [args[0], label]
