# Transformer-based-language-model-from-scratch

This project implements a decoder-only Transformer-based language model from scratch. The model is trained using a character-level tokenizer on a dataset containing one million characters sourced from the works of Shakespeare.

## Features
- **Transformer Architecture**: The model utilizes a decoder-only Transformer architecture, comprising layers of self-attention mechanisms.
  
- **Optimization Techniques**: The implementation incorporates residual networks, LayerNorm, and dropout layers as optimization techniques to improve training efficiency and achieve better performance. These techniques have contributed to reducing the validation loss from 2.08 to 1.5.
  
- **Hyperparameter Tuning**: Extensive hyperparameter tuning has been conducted to optimize various parameters, including layer size, number of multi-heads, learning rate, and embedding size. This ensures that the model achieves optimal performance on the given dataset.

This project is presented as a Jupyter notebook. To use it:

1. **Download the Notebook**: Download the `gpt-dev.ipynb` notebook file.

2. **Open in Jupyter Notebook**: Launch Jupyter Notebook on your local machine and open the downloaded notebook file.

3. **Run the Notebook**: Execute each cell in the notebook sequentially to run the code and observe the model's training process and results.

## Requirements

- Python 3.x
- PyTorch
- NumPy
- tqdm


## Acknowledgements

The implementation of the decoder-only Transformer architecture draws inspiration from the seminal work of Vaswani et al. in the paper `Attention Is All You Need`.
