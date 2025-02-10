# Enhanced GNN generalization to Real Network Dataset by Attention Mechanism
## 1st place solution of Graph Neural Networking Challenge 2023

## The problem
This challenge aims to create a Network Digital Twin based on neural networks that can accurately estimate QoS performance metrics given the network state and the input traffic. More in detail, solutions must predict the resulting per-flow mean delay given: (i) a network topology (L2 and L3), (ii) a set of input flow packet traces, and (iii) a routing configuration. The following figure presents a schematic representation.

## Presentation and Awards Ceremony
[<img src="https://raw.githubusercontent.com/ITU-AI-ML-in-5G-Challenge/ITU-ML5G-PS-007-GNN-m0b1us/main/thumbnail_presentation_awards.jpg" width="50%">](https://youtu.be/ebaKcQV2Zok)

## Instructions to Replicate our Solution
To use this code, you should follow the tutorial described below. 

### Creating the Anaconda Virtual Environment
Considering that one has the Miniconda or the Anaconda virtual environment already installed in a
Linux machine, to recreate the environment used in our solution, the below command should be used to create the environment:

```console
conda env create -f environment.yml
```

After that, you can activate this environment using the next command:
```console
conda activate gnn_2023_m0b1us
```

### Training the m0b1us model
Considering the current directory is the m0b1us model, it is necessary to
execute the command below to perform the training. The below command will train the CBR+MB model considering the dataset at
`datasets/data_cbr_mb_13`. Where the `--ckpt-path` flag defines the weights directory name.

```console
python3 std_train.py -ds CBR+MB --ckpt-path best_model
```

### Performing the predictions
intended to create the predictions utilized in our best solution, the command below should be used.
This command considers the CBR+MB model, the best weight `150-15.7196`, the training and
testing datasets at `datasets/data_cbr_mb_13` and dataset/data_test, respectively.

```console 
python3 std_predict.py -ds CBR+MB --ckpt-path weights/150-15.7196 \
--tr-path datasets/data_cbr_mb_13_cv/0/training/ \
--te-path datasets/data_test/
```

# Credits
If you benefit from this work, please cite on your publications using:

```
@ARTICLE{Modesto2024gnn,
  title     = "Delay estimation based on multiple stage message passing with
               attention mechanism using a real network communication dataset",
  author    = "Modesto, Cl{\'a}udio and Aben-Athar, Rebecca and Silva, Andrey
               and Lins, Silvia and Gon{\c c}alves, Glauco and Klautau,
               Aldebaro",
  journal   = "ITU J. (Geneva)",
  publisher = "International Telecommunication Union",
  volume    =  5,
  number    =  4,
  pages     = "465--477",
  month     =  dec,
  year      =  2024,
  language  = "en"
}
```



