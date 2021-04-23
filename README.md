# Introduction
-----
OpenDebias is a Debias framework for Natural Language Inference research with Pytorch. OpenDebias aims to accelerate research cycle in debias learning: from designing a new debias learning method to debias training a new model on new dataset with the exsiting debias learning methods. Key features include:

- **Reproducible implementation of SOTA in Debias**

- **Benchmark suite:**

- **Ease of Usability**

- **Modular:** Easy to design new tasks and reuse the existing components from other tasks. The modular components are simple drop-in replacements in jsonnet config files.

# Setup
## Install Dependencies
`conda create -n opendebias python=3.6`
`pip install -r requirement.txt`

## Example Usage

### Example. 1
Debiased Train a BERT-based NLI model on MNLI dataset with word-overlap bias in a two-stage manner, and evaluate on HANS.

1. Download MNLI dataset and HANS datset, put them in the path as in examples/configs/dataset/mnli_train.jsonnet, mnli_dev.jsonnet, hans.jsonnet
2. Download pretrained BERT weights, config and vocabulary, put them in the path as in examples/configs/main_model/basic_bert_classifier.jsonnet
3. Run the shel
```bash
bash examples/scripts/two_stage_train_mnli_on_wordoverlap.sh
```

### Example. 2
Debiased Train a BERT-based NLI model on MNLI dataset with hypothesis-only bias in a one-stage manner, and evaluate on MNLI-Hard.
1. Download MNLI dataset and MNLI-hard datset, put them in the path as in examples/configs/dataset/mnli_train.jsonnet, mnli_dev.jsonnet, mnli_hard.jsonnet, mnli_m_hard.jsonnet
2. Download pretrained BERT weights, config and vocabulary, put them in the path as in examples/configs/main_model/basic_bert_classifier.jsonnet
3. Run the shel
```bash
bash examples/scripts/one_stage_train_mnli_on_hyponly.sh
```