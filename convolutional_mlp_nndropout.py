import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from logistic_sgd import LogisticRegression, load_data
from cnn_tools import HiddenLayer, _dropout_from_layer, DropoutHiddenLayer, LeNetConvPoolLayer, DropoutLeNetConvPoolLayer


def evaluate_lenet5(initial_learning_rate=0.01, learning_rate_decay = 1, dropout_rates = [0.2, 0.2, 0.2, 0.5], n_epochs=300,
                    dataset='file_622.pkl.gz',
                    nkerns=[40, 80,160], batch_size=500):
    """ Demonstrates lenet on MNIST dataset

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type learning_rate_decay: float
    :param learning_rate_decay: learning rate decay used (1 means learning rate decay is deactivated)

    :type dropout_rates: list of float
    :param dropout_rates: dropout rate used for each layer (input layer, 1st filtered layer, 2nd filtered layer, fully connected layer)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: path to the dataset used for training /testing (MNIST here)

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer
    """

    rng = numpy.random.RandomState(23455)

    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size
    n_valid_batches /= batch_size
    n_test_batches /= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    epoch = T.scalar()

    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    learning_rate = theano.shared(numpy.asarray(initial_learning_rate,
        dtype=theano.config.floatX))

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # Reshape matrix of rasterized images of shape (batch_size, 28 * 28)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    # (28, 28) is the size of MNIST images.
    layer0_input = x.reshape((batch_size, 1, 100, 46))
   
    layer0 = LeNetConvPoolLayer(
      rng,
      input=layer0_input,
      image_shape=(batch_size, 1, 100, 46),
      filter_shape=(nkerns[0], 1, 5, 5),
      poolsize=(2, 2),
    )
    
    layer1 = LeNetConvPoolLayer(
      rng,
      input=layer0.output,
      image_shape=(batch_size, nkerns[0], 48, 21),
      filter_shape=(nkerns[1], nkerns[0], 5, 5),
      poolsize=(2, 2),
    )
	
    layer1_1 = LeNetConvPoolLayer(
      rng,
      input=layer1.output,
      image_shape=(batch_size, nkerns[1], 22, 8),
      filter_shape=(nkerns[2], nkerns[1], 5, 5),
      poolsize=(2, 2),
    )
	

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4),
    # or (500, 50 * 4 * 4) = (500, 800) with the default values.
    layer2_dropout_input = layer1_1.output.flatten(2)
    layer2_input = layer1_1.output.flatten(2) 

    # construct a fully-connected sigmoidal layer
    layer2_dropout = DropoutHiddenLayer( #drop out
        rng,
        input=layer2_dropout_input,
        n_in=nkerns[2] * 9 * 2,
        n_out=800,
        activation=T.tanh,
        dropout_rate = dropout_rates[3]
    )

    layer2 = HiddenLayer( #connected
      rng,
      input=layer2_input,
      n_in=nkerns[1] * 9 * 2,
      n_out=800,
      activation=T.tanh,
      W=layer2_dropout.W * (1 - dropout_rates[2]),
      b=layer2_dropout.b
    )


    # classify the values of the fully-connected sigmoidal layer
    layer3_dropout = LogisticRegression(
      input = layer2_dropout.output,
      n_in = 800, n_out = 5)

    layer3 = LogisticRegression(
      input=layer2.output,
      n_in=800, n_out=5,
      W=layer3_dropout.W * (1 - dropout_rates[-1]),
      b=layer3_dropout.b
    )


    # the cost we minimize during training is the NLL of the model
    cost = layer3.negative_log_likelihood(y)
    dropout_cost = layer3_dropout.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # create a list of all model parameters to be fit by gradient descent
    params = layer3_dropout.params + layer2_dropout.params + layer1_1.params + layer1.params + layer0.params

    # create a list of gradients for all model parameters
    grads = T.grad(dropout_cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    train_model = theano.function(
        [index],
        dropout_cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # Theano function to decay the learning rate
    decay_learning_rate = theano.function(inputs=[], outputs=learning_rate,
            updates={learning_rate: learning_rate * learning_rate_decay})

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'
    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print 'training @ iter = ', iter
            cost_ij = train_model(minibatch_index)

            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [
                        test_model(i)
                        for i in xrange(n_test_batches)
                    ]
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break
        new_learning_rate = decay_learning_rate()

    end_time = timeit.default_timer()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

if __name__ == '__main__':
    evaluate_lenet5()
