# In-Context Learning for Zero-Shot Speed Estimation of BLDC motors

This repository contains the Python and Matlab code to reproduce the results of the paper [In-Context Learning for Zero-Shot Speed Estimation of BLDC motors](https://arxiv.org/abs/2504.00673)
by Alessandro Colombo, Riccardo Busetto, Valentina Breschi, Marco Forgione, Dario Piga, and Simone Formentin

We propose a learning framework in which a meta model is trained to describe the *class* of BLDC motors. It receives as input a windows the last __H__ samples of the voltages applied to the motor and the measured currents, and estimates the current motor speed.
By training the model on a set of simulated experiments, in which the motor parameters were randomly perturbed, the model is able to comprehend the intrinsic parameters of the system and adapt to different motor configurations.

# Main files

## Simulation

In the *matlab_simulator* folder:
 * __BLDC_simulator.slx__ contains the controller simulator
 * __BLDC_simulator_init[...].m__ files are used to generate the dataset
 * __BLDC_simulator_test[...].m__ were used to tune the nominal parameters of the simulator.

In *matlab_simulator/Model_id_scripts* and *matlab_simulator/EKF_scripts* the code for generating and tuning the EKFs for the different motor configurations can be found.

## Training

In the *speed_estimator* folder:
* __transformer_zerostep.py__ contains the architecture for the meta model
* __dataset.py__ is used to extract data batches from the dataset
* __train_zerostep.py__ is used to train the meta model
* __test_zerostep.py__ test the metamodel against the dataset

The code was originally taken and modified from [sysid-transformers](https://github.com/forgi86/sysid-transformers)


# Software requirements
Simulation was performed on Matlab R2024a

Model training was performed on a Python 3.12 environment with:

 * numpy
 * scipy
 * matplotlib
 * python-control
 * pytorch (v2.7.0)
 * Cuda 12.6

These dependencies may be installed through the commands:

```
pip install numpy scipy matplotlib control
```

For more details on pytorch installation options (e.g. support for CUDA acceleration), please refer to the official [installation instructions](https://pytorch.org/get-started/locally/).

The following packages are also useful:

```
pip install jupyter # (optional, to run the test jupyter notebooks)
pip install wandb # (optional, for experiment logging & monitoring)
```

# Hardware requirements
While all the scripts should be able run on CPU, execution may be frustratingly slow. For faster training, a GPU is highly recommended.
To run the paper's examples, we used a pc equipped with an nVidia RTX 3070ti GPU.




# Citing

If you find this project useful, we encourage you to cite the [paper](https://arxiv.org/abs/2504.00673) 
```
@article{colombo2025context,
  title={In-Context Learning for Zero-Shot Speed Estimation of BLDC motors},
  author={Colombo, Alessandro and Busetto, Riccardo and Breschi, Valentina and Forgione, Marco and Piga, Dario and Formentin, Simone},
  journal={arXiv preprint arXiv:2504.00673},
  year={2025}
}
```

# Look at:
- speed_estimator/transformer_zerostep.py - architettura Transformer (decoder-only) per stima @k dal contesto. 
- speed_estimator/train_zerostep.py - training loop; costruisce il batch, gestisce l'autoregressione (ri- . iniezione di @k-1 come 5° canale), loss, scheduler. 
- speed_estimator/dataset.py - normalizzazioni e finestre temporali (contesto lungo H); prepara (u, y) per il modello.
- speed_estimator/torch_utils.py - utility (device/GPU, ecc.). 
- speed_estimator/deprecated/KalmanFilter.py - baseline (E)KF per confronto con ICL (richiamo allo stato dell'arte).