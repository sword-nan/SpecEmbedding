## SpecEmbedding

Chinese Version: [SpecEmbedding](./docs/README_zh.md)

SpecEmbedding is a deep learning model designed specifically for MS/MS spectral embedding. It combines sinusoidal positional encoding with a supervised contrastive learning framework to improve performance in compound identification and structural similarity retrieval tasks.

The model was trained and evaluated on the GNPS, MoNA, and MTBLS1572 datasets, initially preprocessed by the MSBERT team. To further improve data quality, we removed entries with malformed or invalid SMILES strings. All cleaned data, along with the preprocessing scripts and 10-fold query/reference splits used for evaluation, are available on [figshare](https://doi.org/10.6084/m9.figshare.28876751.v2).

To assess the model's robustness and generalizability on high-quality data, we additionally tested MassBank and MassSpecGym, two curated spectral libraries. SpecEmbedding achieved consistently strong performance on these datasets as well.

To ensure a fair and reproducible evaluation, we strictly retained the original training set split used by MSBERT and only applied random splitting to the test sets. Final results are reported as the average and standard deviation across the 10 splits.

Details on the hyperparameter search space, training procedure, ablation studies, and full benchmark results across multiple tasks are also available on [figshare](https://doi.org/10.6084/m9.figshare.28876751.v2) for reproducibility and further research.

### 1. Environment

OS: Linux Ubuntu 20.04

Python: 3.12

PyTorch: 2.6.0 with CUDA 12.4

> ⚠️ Note for Windows Users: When running on Windows, you may encounter numerical errors during cosine similarity computation. This is caused by @njit decorators from the numba library. You can fix it by commenting out all @njit decorators in the code.

### 2. Demo

#### 2.1 Compute Cosine Similarity Matrix Between Query and Reference Spectra

```python
import sys
sys.path.append("..")

from SpecEmbedding.utils.model import embedding, cosine_similarity, load_tanimoto_supcon_aug_model
from SpecEmbedding.utils.clean import read_raw_spectra
from SpecEmbedding.trainer.trainer import ModelTester
from SpecEmbedding.data.tokenizer import Tokenizer

# Load query and reference spectra
q = read_raw_spectra("./q.msp")
r = read_raw_spectra("./r.msp")

# Initialize tokenizer and device
tokenizer = Tokenizer(100, True)
device = "cpu"

# Define the SiameseModel architecture
model = load_tanimoto_supcon_aug_model(device)

# Initialize the ModelTester
tester = ModelTester(model, device, True)

# Generate embeddings for query and reference spectra
q, _ = embedding(tester, tokenizer, 512, q, True)
r, _ = embedding(tester, tokenizer, 512, r, True)

# Compute the cosine similarity matrix
cosine_scores = cosine_similarity(q, r)
```

#### 2.2 Get the Top-1 Most Similar Compound for Each Query Spectrum

```python
import sys
sys.path.append("..")

from SpecEmbedding.utils.model import embedding, cosine_similarity, load_tanimoto_supcon_aug_model, top_k_indices
from SpecEmbedding.utils.clean import read_raw_spectra
from SpecEmbedding.trainer.trainer import ModelTester
from SpecEmbedding.data.tokenizer import Tokenizer

# Disable progress bar for simplicity
show_progress_bar = False

# Load query and reference spectra
q_spectra = read_raw_spectra("./q.msp")
r_spectra = read_raw_spectra("./r.msp")

# Initialize tokenizer and device
tokenizer = Tokenizer(100, True)
device = "cpu"

# Define the SiameseModel architecture
model = load_tanimoto_supcon_aug_model(device)

# Initialize the ModelTester
tester = ModelTester(model, device, show_progress_bar)

# Generate embeddings for query and reference spectra
q, _ = embedding(tester, tokenizer, 512, q_spectra, show_progress_bar)
r, _ = embedding(tester, tokenizer, 512, r_spectra, show_progress_bar)

# Compute the cosine similarity matrix
cosine_scores = cosine_similarity(q, r)

# Retrieve the indices of the top-1 candidates
indices = top_k_indices(cosine_scores, 1)
for i, index in enumerate(indices[:, 0]):
    print(f"The {i + 1}-th spectra with SMILES {q_spectra[i].get('smiles')} most similar compound is {r_spectra[index].get('smiles')}")
```

In addition, we provide full training and evaluation scripts.
Users can refer to the following Jupyter notebooks for details on model training and performance evaluation:

[Model Training Script](./demo/train_model.ipynb)
[Model Model Evaluation and Metrics](./hit_metric/GNPS&MoNA&MTBLS1572.ipynb)

### 3. Web Service

An online web interface is available for demonstration and public use: [SpecEmbedding](https://huggingface.co/spaces/xp113280/SpecEmbedding)