## small-language-model-experiments

Made as a part of CS689 HW6 on experimentation and building of small language models.

### Objective:

- Design, train and evaluate a seqeunce of increasingly capable small-language-model
and investigate,empirically, how architectural components and computational budget affect model quality.

### Datasets:

- Used several small,commonly used datasets for next-token prediction:
  - Tiny Shakespeare : A 1Mb character-level dataset of Shakespeare's works.
  - WikiText-2 : A word-levl dataset derived from cleaned Wikipedia articles.
  - Penn TreeBank (PTB): A classic word-level dataset consisting of Wall Street Journal Text.


### Models Used:

- Linear Predictor -> A single linear (softmax regression) layer.
- Multi-Layer Perceptron (MLP) -> At least 3 layers, with non-linear activations.
- Multi-head self-attention model -> Implement one or more self-attention layers with configurable
numbers of heads.
- Multi-layer transformer -> A small transformer with multiple blocks.

### Evaluation Metrics:

- For each model we are experimenting with:
  - Context Length (sequence length)
  - MLP hyperparameters
    - number of hiden units 
    - number of layers 
    - activation functions 

  - Self Attention Hyperparameters:
    - Number of Heads 
    - Head Dimension 

  - Transformer Hyperparameters:
    - number of layers 
    - embedding size 
    - MLP width 

  - Optimization choices:
    - learning rate 
    - scheduler 
    - optimizer (Adam vs SGD)
    - Batch size 

- Other Strategies:
  - Alternative embedding strategies 
  - positional encodings 
  - regularization tehcniques 
  - weight tying 
  - modifying training objectives.

