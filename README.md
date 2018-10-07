Introduction
============

The objective of this problem set was to construct four models of text
classifiers in Pytorch to classify the sentiment of input sentences from
the Stanford Sentiment Treebank (SST-2) (Socher et al., 2013) dataset as positive or
negative. In particular, we implement and successfully train a naive
Bayes unigram classifier as described by (Wang and Manning, 2012), a
logistic regression model over word types, a continuous bag-of-word
(CBOW) neural network with embeddings similar to that of
(Mikolov et al., 2013), and a variant of a simple convolutional neural network (CNN)
from (Kim, 2014). After replicating the models as described in the
literature, we then experiment with modifications to the CBOW and CNN
architectures that improve performance, and also propose a variation of
the CNN architecture that significantly outperforms any of the other
models.

Problem Description
===================

In the Stanford Sentiment Treebank sentiment classification task, we are
provided with a corpus of sentences taken from movie reviews. Each
sentence has been tagged as either positive, negative, or neutral; we
follow (Kim, 2014) in removing the neutral examples and formulating the
task as a binary decision between positive and negative sentences.

Model and Algorithms
====================

For each model variant, we formalize prediction <img src="/tex/deceeaf6940a8c7a5a02373728002b0f.svg?invert_in_darkmode&sanitize=true" align=middle width=8.649225749999989pt height=14.15524440000002pt/> for test case <img src="/tex/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode&sanitize=true" align=middle width=9.075367949999992pt height=22.831056599999986pt/> as

<img src="/tex/df20549a30b3ad782db4ceb40d7e29d5.svg?invert_in_darkmode&sanitize=true" align=middle width=158.59744394999998pt height=29.190975000000005pt/>

where <img src="/tex/8cda31ed38c6d59d14ebefa440099572.svg?invert_in_darkmode&sanitize=true" align=middle width=9.98290094999999pt height=14.15524440000002pt/> is an activation function, <img src="/tex/a14a3f2d05c75b9ef74274b694eafda3.svg?invert_in_darkmode&sanitize=true" align=middle width=19.80585089999999pt height=22.55708729999998pt/> are our learned
weights, <img src="/tex/02eea2c618080adc916045fbbf4a5711.svg?invert_in_darkmode&sanitize=true" align=middle width=27.517177049999987pt height=29.190975000000005pt/> is our feature vector for input <img src="/tex/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode&sanitize=true" align=middle width=9.075367949999992pt height=22.831056599999986pt/>, and <img src="/tex/4bdc8d9bcfb35e1c9bfb51fc69687dfc.svg?invert_in_darkmode&sanitize=true" align=middle width=7.054796099999991pt height=22.831056599999986pt/>
is a bias vector. Note that in this problem set, <img src="/tex/7deabf463449f0cf052858d7ee9e1674.svg?invert_in_darkmode&sanitize=true" align=middle width=26.189268599999988pt height=29.190975000000005pt/> <img src="/tex/5ba9e09976f6a5a8919c63baa6f2fbe7.svg?invert_in_darkmode&sanitize=true" align=middle width=10.95894029999999pt height=17.723762100000005pt/>
<img src="/tex/85aa349fbcaeefc01598d8e652098008.svg?invert_in_darkmode&sanitize=true" align=middle width=40.18271399999999pt height=24.65753399999998pt/> for all <img src="/tex/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode&sanitize=true" align=middle width=9.075367949999992pt height=22.831056599999986pt/> since we only consider positive and
negative sentiment inputs in the SST-2 dataset. We use Pytorch for all
model implementations, and all models are trained for 10 epochs each
using batches of size 10, a learning rate of 1e-4, the Adam optimizer,
and the negative log likelihood loss function. The only exception to
this setup was for multinomial naive Bayes, which was fit in one epoch
with learning parameter <img src="/tex/c745b9b57c145ec5577b82542b2df546.svg?invert_in_darkmode&sanitize=true" align=middle width=10.57650494999999pt height=14.15524440000002pt/> = 1.0.

Multinomial Naive Bayes
-----------------------

Let <img src="/tex/653c478e9734abe66ae33dbb01f04154.svg?invert_in_darkmode&sanitize=true" align=middle width=24.57604049999999pt height=29.190975000000005pt/> <img src="/tex/5ba9e09976f6a5a8919c63baa6f2fbe7.svg?invert_in_darkmode&sanitize=true" align=middle width=10.95894029999999pt height=17.723762100000005pt/> <img src="/tex/518b9f574e8c21c3049ff2c6b88fd3f2.svg?invert_in_darkmode&sanitize=true" align=middle width=30.274122449999986pt height=29.190975000000005pt/> be the
feature count vector for training case <img src="/tex/77a3b857d53fb44e33b53e4c8b68351a.svg?invert_in_darkmode&sanitize=true" align=middle width=5.663225699999989pt height=21.68300969999999pt/> with classification label
<img src="/tex/708d9d53037c10f462707daa2370b7df.svg?invert_in_darkmode&sanitize=true" align=middle width=23.57413739999999pt height=29.190975000000005pt/>. <img src="/tex/a9a3a4a202d80326bda413b5562d5cd1.svg?invert_in_darkmode&sanitize=true" align=middle width=13.242037049999992pt height=22.465723500000017pt/> is the set of features, and <img src="/tex/95ab258cd940f4adfc24ff07414898ef.svg?invert_in_darkmode&sanitize=true" align=middle width=22.49335769999999pt height=34.337843099999986pt/>
represents the number of occurrences of feature <img src="/tex/2b1d9e493415b82798e1b94394f4e37d.svg?invert_in_darkmode&sanitize=true" align=middle width=15.69356084999999pt height=22.465723500000017pt/> in training case
<img src="/tex/77a3b857d53fb44e33b53e4c8b68351a.svg?invert_in_darkmode&sanitize=true" align=middle width=5.663225699999989pt height=21.68300969999999pt/>. Define the count vectors of <img src="/tex/154ede110e001f129e857ce348e9c246.svg?invert_in_darkmode&sanitize=true" align=middle width=22.49335769999999pt height=29.190975000000005pt/> as <img src="/tex/980fcd4213d7b5d2ffcc82ec78c27ead.svg?invert_in_darkmode&sanitize=true" align=middle width=10.502226899999991pt height=14.611878600000017pt/> =
<img src="/tex/c745b9b57c145ec5577b82542b2df546.svg?invert_in_darkmode&sanitize=true" align=middle width=10.57650494999999pt height=14.15524440000002pt/> + <img src="/tex/9f7a5b19b125ff1b0251199da054bbda.svg?invert_in_darkmode&sanitize=true" align=middle width=99.41521259999999pt height=29.190975000000005pt/>, for the smoothing
parameter <img src="/tex/c745b9b57c145ec5577b82542b2df546.svg?invert_in_darkmode&sanitize=true" align=middle width=10.57650494999999pt height=14.15524440000002pt/>. We follow Wang and Manning in binarizing the counts;
<img src="/tex/ee547e7b457838d33d63517b730d2bb3.svg?invert_in_darkmode&sanitize=true" align=middle width=28.10730449999999pt height=29.190975000000005pt/> = <img src="/tex/396ac573e737dfe2f2af06b7e4c7ac91.svg?invert_in_darkmode&sanitize=true" align=middle width=9.452005199999991pt height=21.18721440000001pt/>
<img src="/tex/5e5db6fc5fdad3bbeda0a3b5a0cca011.svg?invert_in_darkmode&sanitize=true" align=middle width=75.24539879999999pt height=29.190975000000005pt/>. With regard to Equation (1),
<img src="/tex/380c103b60c66d6420ec8923cdc6e6e8.svg?invert_in_darkmode&sanitize=true" align=middle width=19.80585089999999pt height=22.55708729999998pt/> is the log-count ratio between the number of positive and
negative examples, <img src="/tex/4bdc8d9bcfb35e1c9bfb51fc69687dfc.svg?invert_in_darkmode&sanitize=true" align=middle width=7.054796099999991pt height=22.831056599999986pt/> = log(<img src="/tex/e20f85a81984688a096d5d260d98c6f3.svg?invert_in_darkmode&sanitize=true" align=middle width=23.29914179999999pt height=22.465723500000017pt/>/<img src="/tex/622e8ec6c860226008ede1c559a6c983.svg?invert_in_darkmode&sanitize=true" align=middle width=23.48178854999999pt height=22.465723500000017pt/>) where <img src="/tex/e20f85a81984688a096d5d260d98c6f3.svg?invert_in_darkmode&sanitize=true" align=middle width=23.29914179999999pt height=22.465723500000017pt/> and <img src="/tex/622e8ec6c860226008ede1c559a6c983.svg?invert_in_darkmode&sanitize=true" align=middle width=23.48178854999999pt height=22.465723500000017pt/> are the
number of positive and negative training cases in the training dataset,
<img src="/tex/5fd7788d73b576ccd3de6b7fca34d87b.svg?invert_in_darkmode&sanitize=true" align=middle width=25.10848724999999pt height=29.190975000000005pt/> is the number of occurrences of input <img src="/tex/9fc20fb1d3825674c6a279cb0d5ca636.svg?invert_in_darkmode&sanitize=true" align=middle width=14.045887349999989pt height=14.15524440000002pt/>, and
<img src="/tex/396ac573e737dfe2f2af06b7e4c7ac91.svg?invert_in_darkmode&sanitize=true" align=middle width=9.452005199999991pt height=21.18721440000001pt/> is a binary indicator function that maps <img src="/tex/5fd7788d73b576ccd3de6b7fca34d87b.svg?invert_in_darkmode&sanitize=true" align=middle width=25.10848724999999pt height=29.190975000000005pt/>
to 1 if greater than 0 and 0 otherwise. We consider only unigrams as
features.

Logistic Regression
-------------------

We learn weight and bias matrices <img src="/tex/380c103b60c66d6420ec8923cdc6e6e8.svg?invert_in_darkmode&sanitize=true" align=middle width=19.80585089999999pt height=22.55708729999998pt/> and <img src="/tex/4bdc8d9bcfb35e1c9bfb51fc69687dfc.svg?invert_in_darkmode&sanitize=true" align=middle width=7.054796099999991pt height=22.831056599999986pt/> such to optimize

<p align="center"><img src="/tex/3f02fce8188b5edfdd2024832e9b073c.svg?invert_in_darkmode&sanitize=true" align=middle width=154.0928499pt height=36.6554298pt/></p>

where <img src="/tex/9fc20fb1d3825674c6a279cb0d5ca636.svg?invert_in_darkmode&sanitize=true" align=middle width=14.045887349999989pt height=14.15524440000002pt/> is <img src="/tex/4716870f3f422e3e30698c2aa13917a6.svg?invert_in_darkmode&sanitize=true" align=middle width=22.37447849999999pt height=24.65753399999998pt/>-dimensional vector representing bag-of-words
unigram counts for each training sample. In our implementation, we
represent <img src="/tex/380c103b60c66d6420ec8923cdc6e6e8.svg?invert_in_darkmode&sanitize=true" align=middle width=19.80585089999999pt height=22.55708729999998pt/> and <img src="/tex/4bdc8d9bcfb35e1c9bfb51fc69687dfc.svg?invert_in_darkmode&sanitize=true" align=middle width=7.054796099999991pt height=22.831056599999986pt/> with a single fully-connected layer,
which maps directly to a two unit output layer under sigmoid activation.

Continuous Bag-of-Words
-----------------------

In the CBOW architecture, each word in a sentence input of word-length
<img src="/tex/55a049b8f161ae7cfeb0197d75aff967.svg?invert_in_darkmode&sanitize=true" align=middle width=9.86687624999999pt height=14.15524440000002pt/> is mapped to a <img src="/tex/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode&sanitize=true" align=middle width=9.075367949999992pt height=22.831056599999986pt/>-dimensional embedding vector. The embedding
vectors for all words are averaged to produce a single feature vector
<img src="/tex/7ce60fa175bb49eb2f7e059689024ca0.svg?invert_in_darkmode&sanitize=true" align=middle width=8.66435294999999pt height=14.611878600000017pt/> that represents the entire input. In particular,

<p align="center"><img src="/tex/f511ee09005f22fe52fe76d864a00cbf.svg?invert_in_darkmode&sanitize=true" align=middle width=96.06540735pt height=49.9887465pt/></p>

where <img src="/tex/7ce60fa175bb49eb2f7e059689024ca0.svg?invert_in_darkmode&sanitize=true" align=middle width=8.66435294999999pt height=14.611878600000017pt/> <img src="/tex/5ba9e09976f6a5a8919c63baa6f2fbe7.svg?invert_in_darkmode&sanitize=true" align=middle width=10.95894029999999pt height=17.723762100000005pt/> <img src="/tex/4b312fec5a3abfa6cb179842a5bda867.svg?invert_in_darkmode&sanitize=true" align=middle width=25.55179274999999pt height=30.4110543pt/> and <img src="/tex/028da4cb240f456a705bdcc3b0a46df2.svg?invert_in_darkmode&sanitize=true" align=middle width=13.315249199999991pt height=14.611878600000017pt/> <img src="/tex/5ba9e09976f6a5a8919c63baa6f2fbe7.svg?invert_in_darkmode&sanitize=true" align=middle width=10.95894029999999pt height=17.723762100000005pt/>
<img src="/tex/68c15762df23dac6c55c680d5852ac8f.svg?invert_in_darkmode&sanitize=true" align=middle width=21.44403689999999pt height=30.4110543pt/>, and <img src="/tex/ec71f47b6aee7b3cd545386b93601915.svg?invert_in_darkmode&sanitize=true" align=middle width=13.20877634999999pt height=22.831056599999986pt/> = <img src="/tex/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode&sanitize=true" align=middle width=9.075367949999992pt height=22.831056599999986pt/> for all <img src="/tex/77a3b857d53fb44e33b53e4c8b68351a.svg?invert_in_darkmode&sanitize=true" align=middle width=5.663225699999989pt height=21.68300969999999pt/> are the dimensions of the
sentence embedding and word embeddings, respectively. This encoding
<img src="/tex/7ce60fa175bb49eb2f7e059689024ca0.svg?invert_in_darkmode&sanitize=true" align=middle width=8.66435294999999pt height=14.611878600000017pt/> is then passed into a single fully-connected layer that
maps directly to two output units, representing output classes, under
softmax activation.

Convolutional Neural Network
----------------------------

Let <img src="/tex/c416d0c6d8ab37f889334e2d1a9863c3.svg?invert_in_darkmode&sanitize=true" align=middle width=14.628015599999989pt height=14.611878600000017pt/> <img src="/tex/5ba9e09976f6a5a8919c63baa6f2fbe7.svg?invert_in_darkmode&sanitize=true" align=middle width=10.95894029999999pt height=17.723762100000005pt/> <img src="/tex/76b11a20d53ed4d10c9d38e8b4ecd46a.svg?invert_in_darkmode&sanitize=true" align=middle width=19.13820809999999pt height=27.91243950000002pt/> be the <img src="/tex/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode&sanitize=true" align=middle width=9.075367949999992pt height=22.831056599999986pt/>-dimensional word
vector correspond to the <img src="/tex/77a3b857d53fb44e33b53e4c8b68351a.svg?invert_in_darkmode&sanitize=true" align=middle width=5.663225699999989pt height=21.68300969999999pt/>-th word in the input sentence. After
padding all sentences in an input batch to the same length <img src="/tex/55a049b8f161ae7cfeb0197d75aff967.svg?invert_in_darkmode&sanitize=true" align=middle width=9.86687624999999pt height=14.15524440000002pt/>, where
<img src="/tex/55a049b8f161ae7cfeb0197d75aff967.svg?invert_in_darkmode&sanitize=true" align=middle width=9.86687624999999pt height=14.15524440000002pt/> is the maximum length sentence of all sentences in the batch, each
sentence is then represented as

<p align="center"><img src="/tex/95e6739128498ae9459d6a100d53fe47.svg?invert_in_darkmode&sanitize=true" align=middle width=177.89524335pt height=12.05477955pt/></p>

where <img src="/tex/45848451c711deba755da6422f9e68c6.svg?invert_in_darkmode&sanitize=true" align=middle width=12.785434199999989pt height=19.1781018pt/> is the concatenation operation. Let <img src="/tex/bfddb4c677ca74c5212b9bdbe4532f68.svg?invert_in_darkmode&sanitize=true" align=middle width=39.19628294999999pt height=14.611878600000017pt/>
represent the concatenation of words <img src="/tex/c416d0c6d8ab37f889334e2d1a9863c3.svg?invert_in_darkmode&sanitize=true" align=middle width=14.628015599999989pt height=14.611878600000017pt/>, <img src="/tex/48daf924d1e550eb78217f1e0884411d.svg?invert_in_darkmode&sanitize=true" align=middle width=31.27193519999999pt height=14.611878600000017pt/>,
..., <img src="/tex/c44f404c5862ec20b77e284ed02e857b.svg?invert_in_darkmode&sanitize=true" align=middle width=30.82389749999999pt height=14.611878600000017pt/>. In Convolutional neural networks, we apply
convolution operations <img src="/tex/5ddc1b22140b2658931d8962d8c90c33.svg?invert_in_darkmode&sanitize=true" align=middle width=13.91546639999999pt height=14.611878600000017pt/> <img src="/tex/5ba9e09976f6a5a8919c63baa6f2fbe7.svg?invert_in_darkmode&sanitize=true" align=middle width=10.95894029999999pt height=17.723762100000005pt/> <img src="/tex/2c5a948318138412ea3a0dec0a6d7290.svg?invert_in_darkmode&sanitize=true" align=middle width=30.738368099999988pt height=27.91243950000002pt/> with filter
size <img src="/tex/2ad9d098b937e46f9f58968551adac57.svg?invert_in_darkmode&sanitize=true" align=middle width=9.47111549999999pt height=22.831056599999986pt/> to produce features, where the filter size is effectively the
window size of words to convolve over. Let <img src="/tex/3bc6fc8b86b6c61889f4e572c7546b8e.svg?invert_in_darkmode&sanitize=true" align=middle width=11.76470294999999pt height=14.15524440000002pt/> be a feature generated
by this operation. Then

<p align="center"><img src="/tex/0ce891ff1a4f93f91f68cf06f3da3be3.svg?invert_in_darkmode&sanitize=true" align=middle width=168.47671499999998pt height=16.438356pt/></p>

where <img src="/tex/4bdc8d9bcfb35e1c9bfb51fc69687dfc.svg?invert_in_darkmode&sanitize=true" align=middle width=7.054796099999991pt height=22.831056599999986pt/> is a bias term and <img src="/tex/190083ef7a1625fbc75f243cffb9c96d.svg?invert_in_darkmode&sanitize=true" align=middle width=9.81741584999999pt height=22.831056599999986pt/> is the rectified linear unit (ReLU)
function. Applying filter length size <img src="/tex/31fae8b8b78ebe01cbfbe2fe53832624.svg?invert_in_darkmode&sanitize=true" align=middle width=12.210846449999991pt height=14.15524440000002pt/> over all possible windows of
the words in our input sentence produces the feature map

<p align="center"><img src="/tex/8d71ca3d40ca59ad73cf8f5faad703c3.svg?invert_in_darkmode&sanitize=true" align=middle width=165.36159585pt height=16.438356pt/></p>

In our implementation, we convolve over filter sizes <img src="/tex/2ad9d098b937e46f9f58968551adac57.svg?invert_in_darkmode&sanitize=true" align=middle width=9.47111549999999pt height=22.831056599999986pt/> <img src="/tex/5ba9e09976f6a5a8919c63baa6f2fbe7.svg?invert_in_darkmode&sanitize=true" align=middle width=10.95894029999999pt height=17.723762100000005pt/>
<img src="/tex/deab25011f1573e316f68b62accb9567.svg?invert_in_darkmode&sanitize=true" align=middle width=55.70780654999999pt height=24.65753399999998pt/> and then concatenate the features of each
<img src="/tex/f5ec0198af7987f6245d92311996f877.svg?invert_in_darkmode&sanitize=true" align=middle width=16.09780754999999pt height=14.611878600000017pt/> into a single vector. We apply a max-over-time pooling
operation (Collobert et al, 2011) to this vector of concatenated feature
maps, denoted <img src="/tex/e74308ca1bd81a80819135589e16d2e6.svg?invert_in_darkmode&sanitize=true" align=middle width=8.40178184999999pt height=14.611878600000017pt/>, and get <img src="/tex/0038bd66465254af4225aa31848b342b.svg?invert_in_darkmode&sanitize=true" align=middle width=8.579777249999989pt height=22.831056599999986pt/> = max(<img src="/tex/e74308ca1bd81a80819135589e16d2e6.svg?invert_in_darkmode&sanitize=true" align=middle width=8.40178184999999pt height=14.611878600000017pt/>). We
then apply dropout with <img src="/tex/2ec6e630f199f589a2402fdf3e0289d5.svg?invert_in_darkmode&sanitize=true" align=middle width=8.270567249999992pt height=14.15524440000002pt/> = 0.50 to <img src="/tex/0038bd66465254af4225aa31848b342b.svg?invert_in_darkmode&sanitize=true" align=middle width=8.579777249999989pt height=22.831056599999986pt/> as regularization
measure against overfitting, pass this into a fully-connected layer and
compute the softmax over the output.

Modified CNN
------------

Finally, we also implemented with a series of modifications to the CNN
architecture to give a slight performance improvement on the SST-2
dataset. In this implementation, we utilize Stanford’s GloVe pre-trained
vectors (Pennington et al., 2014), we make these changes:

-   Following (Kim, 2014), we use two copies of word embedding table
    during the convolution and max-pooling steps – one that is
    non-static, or updated during training as a regular module in the
    model, and another that is omitted from the optimizer and preserved
    as static throughout the training run. In the forward pass of the
    model, these two sets of embeddings are concatenated together along
    the “channel” dimension, and then passed into the three
    convolutional layers as a single tensor, with two values for each of
    the 300 dimensions in GloVe model.

-   After producing the combined feature vector representing the
    max-pooled features from the three convolutional kernels, we simply
    add the non-padded word count of the input as a single extra
    dimension, producing a 301-dimension tensor which then gets mapped
    to the 2-unit output. From an engineering standpoint, we find that
    this marginally improves performance on the SST-2 dataset, where, on
    average, positive sentences are slightly longer than negative ones –
    19.41 words versus 19.17. It’s not clear whether this would hold
    across different data sets, or if it’s specific to SST-2. (Though
    it’s also not entirely clear that wouldn’t, and seems to imply an
    interesting corpus-linguistic question – are “positive” sentences
    generally longer than “negative” ones?)

Experiments
===========

In addition to the two changes described above, we also experimented
with a wide range of other modifications to the CNN architecture,
including:

-   Combining the CBOW model with the CNN architecture by concatenating
    the maxpooled CNN vectors with the averaged CBOW vector before
    mapping to the final output units.

-   Adding more linear layers between the convolutions and the final
    output units.

-   Using larger feature maps in the convolutional layers ([200-500],
    instead of 100.)

-   A series of hand-engineered features that tried to capture
    character-level, syntactical, or corpus-linguistic information that
    might get missed by the embeddings – the average length of words in
    the sentence; the average frequency of the words (queried via the
    `wordfreq` Python package); the relative offset inside the sentence
    of the root verb in the dependency tree; whether or not the
    dependency tree includes a negated root verb.

-   Static and non-static channels of pre-trained as well as randomly
    initialized word embeddings, and all combinations of these varieties
    (e.g. static randomized, non-static randomized, static pre-trained)
    via multichannel word embeddings.

Although many of these affected the speed at which the model fit the
training data, none of them improved the classification accuracy on the
validation set – the best performance came from just the two changes
described in section 3.5. Below, we report the classification accuracies
achieved by each of the models outlaid in Section 3 on the test set:

*Model* | *Epoch 1* |
:---: | :---: |
Logisitic Regression | 78.03
Unigram Naive Bayes | 82.48
CBoW | 83.75
CNN | 85.34
Modified CNN | 85.94

Though the modified CNN was the best-performing model, it only beats the
vanilla CNN by about half a percent, even though the multi-channel
embedding architectures takes about twice as long to train. It’s also
notable how competitive the unigram naive Bayes baseline is, especially
given that it fits in about two seconds on a 2017 MacBook Pro, compared
to 20-30 minutes for the CNN architectures. We suspect that adding
bigrams and trigrams as features to the Naive Bayes would make it an
even more competitive baseline than simply using unigrams as features,
as per (Wang and Manning, 2012).

Conclusion
==========

We trained 5 classes of models – logistic regression and naive Bayes
baselines, a simple CBOW architecture, a replication of Kim’s CNN
architecture, and a modified CNN that makes a handful of small tweaks
that improve performance on the SST-2 data. Though our results are
competitive with reported scores in the literature, we were unable to
match the performance reported by Kim (88.1%), using the architecture
that we followed for the CNN.

References
==========

R. Collobert, J. Weston, L. Bottou, M. Karlen, P. Kavukcuoglu. “Natural
Language Processing (almost) from Scratch.” Journal of Machine Learning
Research, pages 2493-2537. 2011.

Y. Kim. “Convolutional Neural Networks for Sentence Classification.”
Proceedings of the 2014 Conference on Empirical Methods in Natural
Language Processing (EMNLP), pages 1746–1751. 2014.

T. Mikolov, K. Chen, G. Corrado, J. Dean. “Efficient estimation of word
representations in vector space.” arXiv preprint arXiv:1301.3781. 2013.

J. Pennington, R. Socher, C. Manning. “GloVe: Global Vectors for
Word Representation.” Proceedings of the 2014 Conference on Empirical Methods in Natural
Language Processing (EMNLP), pages 1532-1543. 2014.

R. Socher, A. Perelygin, J. Wu, J. Chuang, C. Manning, A. Ng, C. Potts. 2013. "Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank." Proceedings of the 2013 Conference on Empirical Methods in Natural
Language Processing (EMNLP), pages 1631-1642. 2013.

S. Wang, C. Manning. “Baselines and Bigrams: Simple, Good Sentiment and
Topic Classification.” Proceedings of the 50th Annual Meeting of the
Association for Computational Linguistics, pages 90–94. 2012.
