import argparse
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import torch

import sklearn
from sklearn import manifold, cluster, metrics, neighbors, decomposition, preprocessing, model_selection


def iterated_concat(current, to_add):
    if len(current) == 0:
        return to_add
    return np.concatenate([current, to_add], axis=0)

# Since we are doing a lot of loading, this is nice to suppress some tedious information.
# Keep in mind, removing this might help debug data loading problems
import mne
mne.set_log_level(False)

PRIMARY_CHOICES = ['subject', 'target', 'dataset']
SECONDARY_CHOICES = ['session', 'dataset', 'target', 'None']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analysis of T-Vectors experiments. While the rest of this project "
                                                 "serves as a demonstration of how to use DN3, this is much more "
                                                 "haphazard.")
    parser.add_argument('vectors', help='Location of the extracted vectors.',
                        nargs='+')
    parser.add_argument("--hidden-size", default=384, type=int, help="Length of resulting T-Vector.")

    # Analysis options
    parser.add_argument('--pool-vectors', type=int, default=1, help="Average pool vectors to stabilize representation "
                                                                    "and minimize outliers.")
    # t-SNE args
    parser.add_argument('--tsne', type=int, default=0, help='Whether to run a (2 or 3)D t-SNE visualization of the '
                                                            'T-vector manifold, where N is specified with this argument.',
                        choices=[2, 3])
    parser.add_argument('--tsne-primary', choices=PRIMARY_CHOICES, default='subject',
                        help='What the primary (colour) label represents on the t-SNE plot.')
    parser.add_argument('--tsne-secondary', choices=SECONDARY_CHOICES, default='None',
                        help='What the secondary (marker) label represents on the t-SNE plot')
    parser.add_argument('--primary-labels', nargs='+', default=None, help='Human-readable labels for primary '
                                                                           'when making the t-SNE legend.')
    parser.add_argument('--secondary-labels', nargs='+', default=None, help='Human-readable labels for the people '
                                                                            'when making the t-SNE legend.')

    parser.add_argument('--pca', type=int, default=0, help='View T-Vectors in terms of their N principle components. '
                                                           'If 2 or 3 will create a visualization.')
    parser.add_argument('--cluster', type=int, default=0, help="Simple k-means clustering evaluation to see how "
                                                               "readily clusterable the individual people are.")
    parser.add_argument('--knn', type=int, default=0, help="Run nearest-neighbours classification of the T-vector "
                                                           "represented trials with respect to people ids.")
    parser.add_argument('--svm', action='store_true')

    parser.add_argument('--predict-performance', default=None, help="Changes the targets in use to the performances "
                                                                    "listed in a provided csv file. Person id is "
                                                                    "swapped for classes divided according to "
                                                                    "'--predict-thresholds'")
    parser.add_argument('--predict-dataset', action='store_true')
    parser.add_argument('--predict-session', action='store_true')
    parser.add_argument('--predict-events', action='store_true')
    parser.add_argument('--predict-thresholds', nargs='+', type=float, default=[0.8],
                        help="The thresholds to use to divide performance classes.")
    parser.add_argument('--save-predictions', default=None, help="Filename to save predictions (and targets) to.")
    args = parser.parse_args()

    t_vectors = list()
    person_ids = list()
    session_ids = list()
    dataset_ids = list()
    event_ids = list()
    for ds_id, dataset in enumerate(args.vectors):
        print("Loading ", dataset)
        loaded = np.load(dataset, allow_pickle=True)
        t_vectors = iterated_concat(t_vectors, loaded['t_vectors'])
        person_ids = iterated_concat(person_ids, loaded['person_ids'].astype(int) + len(person_ids))
        session_ids = iterated_concat(session_ids, loaded['session_ids'].astype(int))
        dataset_ids = iterated_concat(dataset_ids, ds_id * np.ones_like(loaded['person_ids'].astype(int)))
        if 'target' in loaded.files:
            event_ids = iterated_concat(event_ids, loaded['target'].astype(int))
        print("Loaded.")

    if args.predict_dataset:
        targets = dataset_ids
        print("Labelling {} datasets.".format(targets.max() + 1))
    elif args.predict_session:
        targets = session_ids
        print("Labelling {} session.".format(targets.max() + 1))
    elif args.predict_events:
        if len(event_ids) == 0:
            raise ValueError("Events not provided")
        targets = event_ids
        print("Labelling {} events.".format(targets.max() + 1))
    else:
        targets = person_ids
        print("Labelling {} people.".format(targets.max() + 1))

    if args.pool_vectors > 1:
        print("Averaging every {} vectors".format(args.pool_vectors))
        new_t_vectors = list()
        new_person_ids = list()
        new_target_ids = list()
        for person_id in range(person_ids.max() + 1):
            for target_id in range(targets.max() + 1):
                full_mask = np.logical_and(targets == target_id, person_ids == person_id)
                if not np.any(full_mask):
                    if targets is not person_ids and not args.predict_dataset:
                        print("Warning: Subject {} has no {} target.".format(person_id, target_id))
                    continue

                pooled = torch.nn.functional.avg_pool1d(
                    torch.from_numpy(t_vectors[full_mask]).transpose(1, 0).unsqueeze(0), args.pool_vectors
                ).permute([0, 2, 1]).reshape(-1, args.hidden_size).numpy()

                new_t_vectors.append(pooled)
                new_person_ids.append([person_id]*pooled.shape[0])
                new_target_ids.append([target_id] * pooled.shape[0])

        t_vectors = np.concatenate(new_t_vectors, axis=0)
        person_ids = np.concatenate(new_person_ids, axis=0)
        targets = np.concatenate(new_target_ids, axis=0)
        print("{} Unique points after averaging".format(len(t_vectors)))

    if args.pca > 0:
        from mpl_toolkits.mplot3d import Axes3D
        markers = ['*', '+', '^', '.', '-']

        print("PCA - {} components.".format(args.pca))
        pca = decomposition.PCA(args.pca)
        t_vectors = pca.fit_transform(t_vectors)

        if args.pca in [2, 3]:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d' if args.pca == 3 else None)

            def scatter_plot(data, color=None, marker=None, label=None):
                if args.pca == 3:
                    ax.scatter(data[:, 0], data[:, 1], data[:, 2], marker=marker, color=color, label=label)
                else:
                    ax.scatter(data[:, 0], data[:, 1], marker=marker, color=color, label=label)

            for person_id in range(person_ids.max() + 1):
                for i, sess in enumerate(np.unique(session_ids[person_ids == person_id])):
                    scatter_plot(t_vectors[np.logical_and(person_ids == person_id, session_ids == sess), :],
                                 marker=markers[i], color='C{}'.format(person_id), label=str(person_id))

            if person_ids.max() < 10:
                plt.legend()
            plt.show()

    # Consider in terms of predictive performance
    if args.predict_performance is not None:
        assert str(args.predict_performance).endswith('.csv')
        df = pd.read_csv(args.predict_performance)
        assert len(df) == person_ids.max()+1
        print("Loaded performances from: ", args.predict_performance)

        acc = df['Accuracy'].values
        args.predict_thresholds = [0, *args.predict_thresholds, 1]
        bins = np.digitize(acc, args.predict_thresholds)
        for person_id, person_bin in enumerate(bins):
            targets[person_ids == person_id] = person_bin
        print("Divided performance into {} bins. Class counts: {}".format(len(args.predict_thresholds)-1,
                                                                               np.histogram(df['Accuracy'].values,
                                                                                            args.predict_thresholds
                                                                                            )[0]))

    if args.tsne > 0:
        from mpl_toolkits.mplot3d import Axes3D
        markers = ['*', '+', '^', '.', '-']
        primary = [person_ids, targets, dataset_ids][PRIMARY_CHOICES.index(args.tsne_primary)]
        secondary = [session_ids, dataset_ids, targets, None][SECONDARY_CHOICES.index(args.tsne_secondary)]
        if secondary is not None and len(secondary) > len(primary):
            secondary = secondary[np.arange(0, stop=len(secondary), step=args.pool_vectors)]

        if args.primary_labels is None:
            args.primary_labels = np.arange(primary.max() + 1)
        if args.secondary_labels is None and secondary is not None:
            args.secondary_labels = np.arange(secondary.max() + 1)
        assert len(args.primary_labels) == primary.max() + 1
        if secondary is not None:
            assert len(args.secondary_labels) == secondary.max() + 1

        print("TSNE with {} components".format(args.tsne))
        tsne = manifold.TSNE(n_components=args.tsne, init='random', random_state=0, verbose=1)
        tsne_rep = tsne.fit_transform(t_vectors)

        fig = plt.figure(figsize=[11, 8.5])
        ax = fig.add_subplot(111, projection='3d' if args.tsne == 3 else None)

        def scatter_plot(data, color=None, marker=None, label=None):
            if args.tsne == 3:
                ax.scatter(data[:, 0], data[:, 1], data[:, 2], marker=marker, color=color, label=label)
            else:
                ax.scatter(data[:, 0], data[:, 1], marker=marker, color=color, label=label)

        for prime in range(primary.max() + 1):
            if secondary is not None:
                for sec in range(secondary.max() + 1):
                    scatter_plot(tsne_rep[np.logical_and(primary == prime, secondary == sec), :],
                                 marker=markers[sec], color='C{}'.format(prime),
                                 label='{}-{}'.format(args.primary_labels[prime], args.secondary_labels[sec]))
            else:
                scatter_plot(tsne_rep[prime == primary, :], marker='.', color='C{}'.format(prime),
                             label=args.primary_labels[prime])

        if primary.max() < 10:
            plt.legend(prop={'size': 12})
        title = 't-SNE Represented T-Vectors: {}{}'.format(
            args.tsne_primary.capitalize(),
            # "Dataset",
            ' and {}'.format(args.tsne_secondary.capitalize()) if secondary is not None else ''
        )
        plt.title(title)
        plt.show()

    if args.cluster > 0:
        assignment_ami = list()

        for _ in range(args.cluster):
            kmeans = cluster.MiniBatchKMeans(n_clusters=targets.max()+1)
            cluster_labels = kmeans.fit_predict(t_vectors)
            assignment_ami.append(metrics.adjusted_mutual_info_score(targets, cluster_labels))

        print(assignment_ami)
        print("Cluster assignment accuracy over {} runs: mean={}, std={}".format(args.cluster,
                                                                                 np.mean(assignment_ami),
                                                                                 np.std(assignment_ami)))

    classifier_models = list()
    if args.knn > 0:
        classifier_models.append(neighbors.KNeighborsClassifier(args.knn))
    if args.svm:
        classifier_models.append(sklearn.svm.SVC())

    for model in classifier_models:
        print("Evaluating ", str(model))
        if args.predict_performance is not None:
            split = model_selection.GroupKFold(n_splits=5).split(t_vectors, targets, groups=person_ids)
        else:
            split = model_selection.StratifiedKFold(n_splits=5).split(t_vectors, targets)

        # Stratify by person_ids
        score = list()
        target_wise = list()
        saved = [list(), list()]
        for train, test in split:
            model.fit(t_vectors[train, :], targets[train].squeeze())
            score.append(model.score(t_vectors[test, :], targets[test].squeeze()))
            print("Overall Accuracy: ", score[-1])
            predictions = model.predict(t_vectors[test, :])
            cmat = metrics.confusion_matrix(targets[test].squeeze(), predictions)
            target_wise.append(cmat.diagonal() / cmat.sum(0))
            print(target_wise[-1])
            saved[0].append(predictions)
            saved[1].append(targets[test].squeeze())
            # print("{:.1%}".format(results[-1]))
        print("Average: {:.1%}".format(np.mean(score)))
        predictions = np.concatenate(saved[0])
        targets = np.concatenate(saved[1])
        cmat = metrics.confusion_matrix(targets, predictions)
        print("Overall target-wise:", cmat.diagonal() / cmat.sum(0))
