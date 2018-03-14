import numpy as np


def model_predict(W, b, x):
    """Prediction function.

    Note that this function can be merged with model_loss, but is written
    as a separate function simply to enhance readibility.

    Parameters
    ----------
    W : ndarray
        The weight parameters of the linear classifier. D x C, where C is the
        number of classes, and D is the dimenstion of input data.

    b : ndarray
        The bias parameters of the linear classifier. C, where C is the number
        of classes.

    x : ndarray
        Input data that we want to predic the labels of. NxD, where D is the
        dimension of the input data.

    Returns
    -------
    pred : ndarray
        Predictions from the model. N numbers where each number corresponds to
        a class.

    """

    # Scores for all class (N, 10)
    s_all = np.matmul(x, W) + b
    # Predections to use later
    pred = np.argmax(s_all, axis=1)

    return pred


def model_loss(W, b, x, y):
    """Loss function.

    Also computes the individual per-class losses to be used for gradient
    computation, as well as the predicted labels.

    Parameters
    ----------
    W : ndarray
        The weight parameters of the linear classifier. D x C, where C is the
        number of classes, and D is the dimenstion of input data.

    b : ndarray
        The bias parameters of the linear classifier. C, where C is the number
        of classes.

    x : ndarray
        Input data that we want to predic the labels of. NxD, where D is the
        dimension of the input data.

    y : ndarray
        Ground truth labels associated with each sample. N numbers where each
        number corresponds to a class.

    Returns
    -------
    loss : float
        The average loss coming from this model. In the lecture slides,
        represented as \sum_i L_i.

    loss_c : ndarray
        The individual losses per sample and class, or any other value that
        needs to be reused when computing gradients.

    pred : ndarray
        Predictions from the model. N numbers where each number corresponds to
        a class.

    """

    # TODO: Compute pred
    s_all = np.matmul(x, W)+b

    pred = np.argmax(s_all, axis =1)
    # TODO: Compute probs
    s_y = np.exp(s_all)[np.arange(len(y)),y]

    probs = np.exp(s_all)/np.mean(np.exp(s_all),axis =1, keepdims = True)

    # TODO: Compute loss
    loss = np.log(s_y) * -1

    loss = np.mean(loss)

    return loss, probs, pred


def model_grad(loss_c, x, y):
    """Gradient.

    Parameters
    ----------
    loss_c : ndarray
        The individual losses per sample and class, or any other value that
        needs to be reused when computing gradients.

    x : ndarray
        Input data that we want to predic the labels of. NxD, where D is the
        dimension of the input data.

    y : ndarray
        Ground truth labels associated with each sample. N numbers where each
        number corresponds to a class.

    Returns
    -------
    dW : ndarray
        Gradient associated with W. Should be the same shape (DxC) as W.

    db : ndarray
        Gradient associated with b. Should be the same shape (C) as b.

    """

    # loss_c : (N, C)
    # x : (N, D)
    # y : (N, )

    N, D = x.shape
    C = loss_c.shape[1]

    # TODO: Compute dW

    # TODO: Compute db

    return dW, db


