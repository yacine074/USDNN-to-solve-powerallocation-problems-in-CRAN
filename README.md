# Unsupervised-deep-learning-to-solve-powerallocation-problems-in-cognitive-relay-networks
## Abstract
In  this  paper,  an  unsupervised  deep  learning  approach  is  proposed  to  solve  the  constrained  and  non-convex rate maximization problem in a relay-aided cognitive   radio network. The network under study is composed by a primary, respectively a secondary, user–destination pair and a secondary full-duplex relay performing Decode-and-Forward. The primary communication is protected by imposing a Quality of Service(QoS) constraint in terms of primary Shannon rate degradation. The relaying operation is highly non linear  and leads to a non-convex objective function and primary QoS constraint, which makes deep learning approaches relevant and promising to solve such a difficult problem. For this,  we propose a fully-connected neural network and a custom loss function to be minimized during training. Our numerical experiments show that the proposed neural network has a high generalization capability on unseen data, with no over-fitting effects. Also, the predicted solution performs close to the optimal one obtained by bruteforce.


## Setup
<ul>
  <li>Install latest version of python</li>
  <li>Run DNN_training.ipynb to generate the dataset and simulate the training phase</li>
  <li>Run DNN_test.ipynb to evaluate the DNN over the test set </li>
  <li>Run DNN_size.ipynb for architecture choice experiments </li>
  <li> Run Impact_of_relay_position.ipynb to see the impact of the relay position </li>
</ul>


## How to cite this code

When writing a paper that uses this code, we would appreciate it if you could cite the paper [HAL](https://hal.archives-ouvertes.fr/hal-03534545/).

## References

Benatia Y, Savard A, Negrel R, Belmega EV. Unsupervised deep learning to solve power allocation problems in cognitive relay networks. In2022 IEEE International Conference on Communications Workshops (ICC Workshops) 2022 May 16 (pp. 331-336). IEEE.


## Contact

yacine.ben-atia@imt-nord-europe.fr or benatia_yacine@hotmail.fr

