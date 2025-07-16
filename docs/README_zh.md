## SpecEmbedding

SpecEmbedding 是一个专注于 MS/MS 光谱嵌入的深度学习模型，结合了正弦位置编码与监督对比学习策略，旨在提升化合物鉴定与结构相似性检索的性能。

模型的训练与测试基于经过 MSBERT 团队初步清洗的 GNPS、MoNA 和 MTBLS1572 数据集。在此基础上，我们进一步去除了一些格式错误和无效的 SMILES 序列，以提升整体数据质量。所有清洗后的数据及相关处理脚本现已发布至 [figshare](https://doi.org/10.6084/m9.figshare.28876751.v2)，包括用于模型评估的 10 次查询集与参考集随机划分结果。

为了验证模型在高质量谱图数据上的表现，我们还使用了 MassBank 和 MassSpecGym 两个标准库进行测试，模型在这些数据集上同样展现出优异的性能和良好的泛化能力。

为确保实验评估的公正性，我们严格保留了 MSBERT 设定的训练集划分，仅在测试集上进行了多次随机划分，并基于这些划分结果报告了 hit@k 的平均值与标准差。

此外，关于超参数搜索空间、模型训练细节、消融实验结果，以及各模型在不同任务下的完整评估指标，也一并公开存储在 [figshare](https://doi.org/10.6084/m9.figshare.28876751.v2) 中，供他人复现与参考。

更多关于评估的结果以及细节可参照 [SpecEmbedding-Comparison](https://github.com/sword-nan/SpecEmbedding-Comparison)

### 1. 环境配置

操作系统：Linux Ubuntu 20.04

Python：3.12

PyTorch：2.6.0 + CUDA 12.4

> ⚠️ 注意：在 Windows 系统上运行时，如发现计算余弦相似度矩阵出现异常结果，请将 @njit 装饰器注释掉（该装饰器来自 numba），即可恢复正常。

### 2. 示例

#### 2.1 计算 query 和 reference 的余弦相似度矩阵
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

#### 2.2 计算 Top1 的候选化合物

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

此外，我们提供了完整的训练脚本与测试脚本。

用户可以通过阅读以下两个 Jupyter Notebook 文件了解详细的训练流程与评估方法：

[训练脚本](../demo/train_model.ipynb)
[评估脚本](../hit_metric/GNPS&MoNA&MTBLS1572.ipynb)

### 3. Web 服务

我们也为用户提供了一个 web 服务，每个人都可以通过访问网址 [SpecEmbedding](https://huggingface.co/spaces/xp113280/SpecEmbedding) 使用。