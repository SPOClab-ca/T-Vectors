# T-Vectors

A method for generating person identifying vectors from arbitrary lengths
of compatible EEG. See .

Please cite this article if you continue to explore and adapt the use of T-Vectors.

This work was created in large part using the DN3 library: https://github.com/SPOClab-ca/dn3. If
confused about the implementation please refer to DN3 for more information.

Currently, the configuration files found in `configs/` do not have any associated `toplevel` entries.
This is because you will need to point the configuratron to wherever you have downloaded your data.

Once this is done, pretraining can be accomplished by running:

```shell script
.../T-vectors/$ python3 pretrain_tvectors.py --median-sampling
```

T-Vectors can be extracted for each dataset `ds`, with T-Vector model weights `t-weights.pt` by running:

```shell script
.../T-vectors/$ mkdir -p extracted_vectors/
.../T-vectors/$ python3 extract_vectors.py extracted_vectors/ds.npz t-weights.pt --session-id
```

If using provided pre-trained model weights (available for download in the `Releases` section of this repo), 
results and the figures found in the associated publication can be generated as follows:

```shell script
.../T-vectors/$ ./make_predictions.sh
.../T-vectors/$ ./make_plots.sh
```

Each python module has a variety of options to change how training, extraction and analysis can be done,
have a look by adding `--help` to any invocation.