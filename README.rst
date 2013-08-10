################
Project Proposal
################

:Authors:
    Bill Chickering (bchick),
    Charles Celerier (cceleri)

A fundamental challenge to supervised machine learning is that of feature
selection. There exist many strategies for determining a set of variables
derived from an original representation to accurately classify domain data.
Ideally, the selected variables minimize redundant and/or irrelevant information
and thereby reduce the risk of overfitting. Feature selection is often addressed
using dimension reduction techniques such as principal component anaysis (PCA)
or factor analysis. Recently it was found that K-means clustering offers another
straightforward method for determining an efficacious feature space (e.g. the
CS221 Visual Cortex programming assigment). As our final project, we intend to
implement another popular approach to feature extraction that utilizes deep
neural networks constructed from sparse autoencoders.

Specifically, we plan to use sparse autoencoders to extract a feature set for
the USPS handwritten digit data (http://www.gaussianprocess.org/gpml/data/).
This dataset consists of training and testing data each comprising 4649 16x16
pixel greyscale images. Following the methods of Hinton and Salakhutdinov [1]_,
we will "pretrain" individual restricted Boltzmann machines (RBMs, i.e. two
layer bipartite neural networks) and thereby learn associated weights and
biases. The output, or learned features, of one RBM may then be used as the
input of another RBM. Each detector layer, or output, of an RBM will consist of
fewer nodes than the input layer. By layering several RBMs in this way, a deep
autoencoder can be constructed. The innermost hidden layer of such an
autoencoder then provides a relatively high-level feature set from which a
classifier can be trained via supervised learning.

Once our feature set along with a method for its extaction is determined via
this unsupervised learning method, we will construct a naive Bayes model that
relates the handwritten digit category (e.g. 0-9) to each feature. The training
data can then be used to determine prior and conditional probability density
functions. These probabilities, in turn, will allow us to develop a classifier,
which can be evaluated using the testing data.

Our first milestone will be the implementation and training of a single RBM. The
input nodes of this RBM will directly map to the 256 raw image pixels. The
output will provide an initial feature set from which we will develop a
classifier as described above. We expect that using a shallow neural network in
this way we will achieve a testing error that is significantly smaller than that
obtained from uniformly at random selecting digit classifactions. Assuming this
hypothesis is confirmed, we will proceed to implement and train a second RBM
that uses the output of our first RBM as its input. We will then develop a
classifier using the output of the second RBM as features. We expect that the
accuracy achieved with this new classifier will exceed that of the first
classifier. In this way, we intend to demonstrate the value of deep neural
networks for feature selection.

.. [1] G.E. Hinton and R.R. Salakhutdinov. Science, 313:504, 2006.

..
   vim: set tw=80 ts=3 sw=3 expandtab:

