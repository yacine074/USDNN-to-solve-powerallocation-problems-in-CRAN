# Unsupervised-deep-learning-to-solve-powerallocation-problems-in-cognitive-relay-networks
## Abstract
In  this  paper,  an  unsupervised  deep  learning  approach  is  proposed  to  solve  the  constrained  and  non-convex rate maximization problem in a relay-aided cognitive   radio network. The network under study is composed by a primary, respectively a secondary, user–destination pair and a secondary full-duplex relay performing Decode-and-Forward. The primary communication is protected by imposing a Quality of Service(QoS) constraint in terms of primary Shannon rate degradation. The relaying operation is highly non linear  and leads to a non-convex objective function and primary QoS constraint, which makes deep learning approaches relevant and promising to solve such a difficult problem. For this,  we propose a fully-connected neural network and a custom loss function to be minimized during training. Our numerical experiments show that the proposed neural network has a high generalization capability on unseen data, with no over-fitting effects. Also, the predicted solution performs close to the optimal one obtained by bruteforce.


## Setup
<ul>
  <li>Install latest version of python</li>
  <li>Run DNN_training.ipynb to generate the dataset and simulate the training phase</li>
  <li>Run DNN_Performance_on_test_set.ipynb to evaluate the DNN over the test set </li>
  <li>Run DNN_Architecture_choice.ipynb for architecture choice experiments </li>
  <li> Run Impact_of_relay_position.ipynb to see the impact of the relay position </li>
</ul>

## Dataset is avalaible on [Google Drive](https://drive.google.com/drive/folders/169LxP_d4DewpBP8tgEQdOWIoroqihzkx?usp=sharing).

## Weights and model history are avalaible on [Google Drive](https://drive.google.com/file/d/1StOpo4eztMm2OE9YCSi6Fqc3SNI-hbHv/view?usp=sharing).

## Data for architecture choice is avalaible on [Google Drive](https://drive.google.com/drive/folders/1MCpUiI_Z35Ocft-mTbsfv-rS8wK5VBFb?usp=sharing).



## How to cite this code

When writing a paper that uses this code, we would appreciate it if you could cite the paper [HAL](https://hal.archives-ouvertes.fr/hal-03534545/).

## References

Yacine Benatia, Anne Savard, R. Negrel, and E. Veronica Belmega, "Unsupervised deep learning to solve powerallocation problems in cognitive relay networks".


## Contact

yacine.ben-atia@imt-nord-europe.fr or benatia_yacine@hotmail.fr

