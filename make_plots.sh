#!/bin/bash

# ------------------------------------------
BCI_PERSON_LABELS="A01 A02 A03 A04 A05 A06 A07 A08 A09"

python3 analysis.py extracted_vectors/bci_iv_2a.npz --tsne 2 --primary-labels "$BCI_PERSON_LABELS" \
        --tsne-primary subject --tsne-secondary session --secondary-labels T E

python3 analysis.py extracted_vectors/bci_iv_2a.npz --tsne 2 --primary-labels "$BCI_PERSON_LABELS" \
        --tsne-primary subject --tsne-secondary session --pool-vectors 4 --secondary-labels T E

# ------------------------------------------

python3 analysis.py extracted_vectors/bci_iv_2a.npz extracted_vectors/mmi.npz --tsne 2 --tsne-primary target \
        --primary-labels BCI-IV-2a MMIdb --pool-vectors 4 --predict-dataset

python3 analysis.py extracted_vectors/bci_iv_2a.npz extracted_vectors/mmi_1020.npz --tsne 2 --tsne-primary target \
        --primary-labels BCI-IV-2a MMIdb --pool-vectors 4 --predict-dataset