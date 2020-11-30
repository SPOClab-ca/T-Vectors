import tqdm
from pandas import DataFrame

from dn3.configuratron import ExperimentConfig
from dn3.trainable.processes import StandardClassification
# from dn3.trainable.models import TIDNet
from dn3.transforms.instance import CropAndResample

from dn3_ext import Deep1010ToEEG, FilteredTIDNet, FilteredEEGNet, RandomCrop, UniformTransformSelection


# Since we are doing a lot of loading, this is nice to suppress some tedious information
import mne
mne.set_log_level(False)

if __name__ == '__main__':
    experiment = ExperimentConfig('configs/classification.yml')

    dataset = experiment.datasets[list(experiment.datasets.keys())[0]]
    dataset = dataset.auto_construct_dataset()
    # dataset.add_transform(Deep1010ToEEG())

    results = list()
    for fold, (training, validation, test) in enumerate(tqdm.tqdm(dataset.lmso(experiment.training_params.folds),
                                                        total=experiment.training_params.folds,
                                                        desc="LMSO", unit='fold')):

        # training.add_transform(CropAndResample(dataset.sequence_length - 16, 4))
        training.add_transform(UniformTransformSelection([
            CropAndResample(dataset.sequence_length - 128, stdev=8),
            RandomCrop(dataset.sequence_length - 128)
        ], weights=[0.25, 0.75]))
        validation.add_transform()

        tidnet = FilteredTIDNet.from_dataset(dataset)
        # tidnet = FilteredEEGNet.from_dataset(dataset)

        process = StandardClassification(tidnet, learning_rate=experiment.training_params.lr)
        process.fit(training_dataset=training, validation_dataset=validation, epochs=experiment.training_params.epochs,
                    batch_size=experiment.training_params.batch_size, num_workers=0)

        for _, _, test_thinker in test.loso():
            summary = {'Person': test_thinker.person_id,
                       'Fold': fold+1,
                       "Accuracy": process.evaluate(test_thinker)['Accuracy']}
            results.append(summary)
        _res = DataFrame(results)
        tqdm.tqdm.write(str(_res[_res['Fold'] == fold+1].describe()))

    df = DataFrame(results)
    print(df.describe())
    # df.to_csv('baseline_mmi.csv')
