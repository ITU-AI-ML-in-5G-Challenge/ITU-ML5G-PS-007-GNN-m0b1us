# Enhanced GNN generalization to Real Network Dataset by Attention Mechanism
## 1st place solution of Graph Neural Networking Challenge 2023

## The problem
The goal of this challenge is to create a Network Digital Twin based on neural networks which can accurately estimate QoS performance metrics given the network state and the input traffic. More in detail, solutions must predict the resulting per-flow mean delay given: (i) a network topology (L2 and L3), (ii) a set of input flow packet traces, and (iii) a routing configuration. The following figure presents a schematic representation.


## Instructions to Replicate our Solution
To use this code, you should follow the tutorial described below. 

### Creating the Anaconda Virtual Environment
Considering that one has the miniconda or the Anaconda virtual environment already installed in a
Linux machine, to recreate the environment used in our solution, below command should be used to create the environment:

```console
conda env create -f environment.yml
```

After that, you can activate this environment using the next command:
```console
conda activate gnn_2023_m0b1us
```

### Training the m0b1us model
Considering the current directory is the m0b1us model, to perform the training, it is necessary to
execute the command below. The below command will train the CBR+MB model considering the dataset at
`datasets/data_cbr_mb_13`. Where the `--ckpt-path` flag defines the weights directory name.

```console
python3 std_train.py -ds CBR+MB --ckpt-path best_model
```

### Perfoming the predictions
ntended to create the predictions utilized in our best solution, the command below should be used.
This command takes into account the CBR+MB model, the best weight `150-15.7196`, the training and
testing datasets at `datasets/data_cbr_mb_13` and dataset/data_test, respectively.

```console 
python3 std_predict.py -ds CBR+MB --ckpt-path weights/150-15.7196 \
--tr-path datasets/data_cbr_mb_13_cv/0/training/ \
--te-path datasets/data_test/
```

# Credits
This project would not have been possible without the contribution of:

[Cláudio Matheus Modesto](https://github.com/claudio966) - [LASSE](https://github.com/lasseufpa), Federal University of Pará

[Rebecca Aben-Athar](https://github.com/rebeccaathar) - [LASSE](https://github.com/lasseufpa), Federal University of Pará

Andrey Silva - Ericsson

Silvia Lins - Ericsson



