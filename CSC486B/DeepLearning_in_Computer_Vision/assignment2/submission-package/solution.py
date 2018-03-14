import matplotlib.pyplot as plt
import numpy as np

from config import get_config, print_usage
from utils.cifar10 import load_data
from utils.preprocess import normalize
from utils.regularizor import l2_grad, l2_loss




def compute_loss(W, b, x, y, config):
    """Computes the losses for each module."""

    # Lazy import of propper model
    if config.model_type == "linear_svm":
        from utils.linear_svm import model_loss
    elif config.model_type == "logistic_regression":
        from utils.logistic_regression import model_loss
    else:
        raise ValueError("Wrong model type {}".format(
            config.model_type))

    loss, loss_c, pred = model_loss(W, b, x, y)
    loss += config.reg_lambda * l2_loss(W)

    return loss, loss_c, pred


def compute_grad(W, x, y, loss_c, config):
    """Computes the gradient for each module."""

    # Lazy import of propper model
    if config.model_type == "linear_svm":
        from utils.linear_svm import model_grad
    elif config.model_type == "logistic_regression":
        from utils.logistic_regression import model_grad
    else:
        raise ValueError("Wrong model type {}".format(
            config.model_type))

    dW, db = model_grad(loss_c, x, y)
    dW += config.reg_lambda * l2_grad(W)

    return dW, db


def predict(W, b, x, config):
    """Predict function.

    Lazy imports the proper `model_parse` function and returns its
    results. behaves quite similarly to how `compute_loss` and `compute_grad`
    works.

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

    config : namespace
        Arguments and configurations parsed by `argparse`

    Returns
    -------
    pred : ndarray
        Predictions from the model. N numbers where each number corresponds to
        a class.

    """

    # TODODone: Lazy import of propper model
    # TODODone: use model_predict

    if config.model_type == "linear_svm":
        from utils.linear_svm import model_predict
    elif config.model_type == "logistic_regression":
        from utils.logistic_regression import model_predict
    else:
        raise ValueError("Wrong model type {}".format(
            config.model_type))



    return model_predict(W, b, x)


def train(x_tr, y_tr, x_va, y_va, config):
    """Training function.

    Parameters
    ----------
    x_tr : ndarray
        Training data.

    y_tr : ndarray
        Training labels.

    x_va : ndarray
        Validation data.

    y_va : ndarray
        Validation labels.

    config : namespace
        Arguments and configurations parsed by `argparse`

    Returns
    -------
    train_res : dictionary
        Training results stored in a dictionary file. It should contain W and b
        when best validation accuracy was achieved, as well as the average
        losses per epoch during training, and the average accuracy of each
        epoch to analyze how training went.
    """

    # ----------------------------------------
    # Preprocess data

    # Report data statistic
    print("Training data before: mean {}, std {}, min {}, max {}".format(
        x_tr.mean(), x_tr.std(), x_tr.min(), x_tr.max()
    ))

    # Normalize data using the normalize function. Note that we are remembering
    # the mean and the range of training data and applying that to the
    # validation/test data later on.
    x_tr_n, x_tr_mean, x_tr_range = normalize(x_tr)
    x_va_n, _, _ = normalize(x_va, x_tr_mean, x_tr_range)
    # Always a good idea to print some debug messages
    print("Training data after: mean {}, std {}, min {}, max {}".format(
        x_tr_n.mean(), x_tr_n.std(), x_tr_n.min(), x_tr_n.max()
    ))

    # ----------------------------------------
    # Initialize parameters of the classifier
    print("Initializing...")
    num_class = 10
    # TODODone: Initialize W to very small random values. e.g. random values between
    # -0.001 and 0.001   1000/10000000 0.00999 1/1000000

    W = np.random.uniform(-0.001,0.001,(x_tr[0].shape[0], num_class))

    # TODODone: Initialize b to zeros
    b = np.zeros(num_class)

    print("Testing...")
    # TODODone: Test on validation data
    y_pred = predict(W, b, x_va, config) 
    

    true_pred = (y_pred == y_va) #https://stackoverflow.com/questions/45418491/estimating-accuracy-with-x-y-mean-how-does-it-work

    
    acc = true_pred.mean()

    #acc number of correct prediction/number of predictions
    print("Initial Validation Accuracy: {}%".format(acc * 100))

    batch_size = config.batch_size
    num_epoch = config.num_epoch
    num_batch = len(x_tr_n) // batch_size
    loss_epoch = []
    tr_acc_epoch = []
    va_acc_epoch = []
    W_best = None
    b_best = None
    best_acc = 0
    # For each epoch
    for idx_epoch in range(num_epoch):
        # TODO: Create a random order to go through the data
        # TODO: For each training batch

        randomizer = np.arange(len(x_tr_n))
        np.random.shuffle(randomizer)
    
        x_tr_n      = x_tr_n[randomizer] # x features
        y_tr        = y_tr[randomizer] # label


        losses = np.zeros(num_batch)
        accs = np.zeros(num_batch)

        x_tr_b = np.reshape(x_tr_n, (num_batch, batch_size , -1))
        y_tr_b = np.reshape(y_tr, (num_batch, batch_size))

        for idx_batch in range(num_batch):

            # TODO: Construct batch

            x_b = x_tr_b[idx_batch]
            y_b = y_tr_b[idx_batch]


            # Get loss with compute_loss
            loss_cur, loss_c, pred_b = compute_loss(W, b, x_b, y_b, config)
            # Get gradient with compute_grad
            dW, db = compute_grad(W, x_b, y_b, loss_c, config) # implement? 
            # TODODone: Update parameters http://wiki.fast.ai/index.php/Gradient_Descent
            W = W - (config.learning_rate * dW) #w - alpha dw, alpha from config
            b = b - (config.learning_rate * db)#b - alpha db
            # TODODone: Record this batches result 
            losses[idx_batch] = loss_cur # single scaler
       


            batch_pred = predict(W, b, x_b, config)
            batch_accs = (batch_pred == y_b)

            accs[idx_batch] = batch_accs.mean() #single scaler

          #  print("Batch accuracy is: ", accs[idx_batch]*100)

        # Report average results within this epoch
        print("Epoch {} -- Train Loss: {}".format(
            idx_epoch, np.mean(losses)))
        print("Epoch {} -- Train Accuracy: {:.2f}%".format(
            idx_epoch, np.mean(accs) * 100))

        # TODODone: Test on validation data and report results
        final_pred = predict(W, b, x_va, config)
        pred_accs = (final_pred == y_va) 
        acc = pred_accs.mean()
        print("Epoch {} -- Validation Accuracy: {:.2f}%".format(
            idx_epoch, acc * 100))

        # TODODone: If best validation accuracy, update W_best, b_best, and best
        # accuracy. We will only return the best W and b
        if acc > best_acc:
            W_best      = W
            b_best      = b
            best_acc    = acc

        # TODODone: Record per epoch statistics
        loss_epoch += np.mean(losses)
        tr_acc_epoch += np.mean(accs)
        va_acc_epoch += best_acc

    # TODO: Pack results. Remeber to pack pre-processing related things here as
    # well
    train_res = {
        'best_acc': best_acc,
        'W_best': W_best,
        'b_best': b_best,
        'loss_epoch': loss_epoch,
        'tr_acc_epoch': tr_acc_epoch,
        'va_acc_epoch': va_acc_epoch


    }


    return train_res


def compute_acc(predict_array, truth_array):
    cnt = 0
    for idx in range(len(predict_array)):
        if predict_array[idx] ==truth_array[idx]:
            cnt+=1
    return cnt/len(truth_array)

def main(config):
    """The main function."""
    #initializition of some int
    avg_best_acc = 0
    avg_W_best = 0
    avg_b_best = 0
    result_acc = 0
    result_W = 0
    result_b = 0



    # Load cifar10 train data in my labtop
    config.data_dir = "/Users/CharlesLiu/Desktop/cifar-10-batches-py/"
    # ----------------------------------------
    # Load cifar10 train data
    print("Reading training data...")
    data_trva, y_trva = load_data(config.data_dir, "train")

    # ----------------------------------------
    # Load cifar10 test data
    print("Reading test data...")
    data_te, y_te = load_data(config.data_dir, "test")

    # ----------------------------------------
    # Extract features
    print("Extracting Features...")
    if config.feature_type == "hog":
        # HOG features
        from utils.features import extract_hog
        x_trva = extract_hog(data_trva) #x_trva is training data veatures
        x_te = extract_hog(data_te) #x_te is test data features
    elif config.feature_type == "h_histogram":
        # Hue Histogram features
        from utils.features import extract_h_histogram
        x_trva = extract_h_histogram(data_trva)
        x_te = extract_h_histogram(data_te)
    elif config.feature_type == "rgb":
        # raw RGB features
        x_trva = data_trva.astype(float).reshape(len(data_trva), -1)
        x_te = data_te.astype(float).reshape(len(data_te), -1)

    # ----------------------------------------
    # Create folds
    num_fold = 5
    

    # TODODone: Randomly shuffle data and labels. IMPORANT: make sure the data and
    # label is shuffled with the same random indices so that they don't get
    # mixed up!
    
    randomizer = np.arange(len(x_trva))
    np.random.shuffle(randomizer)
   
    data_trva   = data_trva[randomizer] # raw rgb data - not really using anymore, not needed to shuffle, unless showing image
    x_trva      = x_trva[randomizer] # x features
    y_trva      = y_trva[randomizer] # label

    

    # Reshape the data into 5x(N/5)xD, so that the first dimension is the fold
    x_trva = np.reshape(x_trva, (num_fold, len(x_trva) // num_fold, -1))
    y_trva = np.reshape(y_trva, (num_fold, len(y_trva) // num_fold))


    # Cross validation setup. If you set cross_validate as False, it will not
    # do all 5 folds, but report results only for the first fold. This is
    # useful when you want to debug.
    if config.cross_validate:
        va_fold_to_test = np.arange(num_fold)
    else:
        va_fold_to_test = np.arange(1)

    # ----------------------------------------
    # Cross validation loop
    train_res = []

    # x_tr is training data
    # x_va is validation data
    # y_tr is training label
    # y_va is validation label

    for idx_va_fold in va_fold_to_test: 
        # TODODone: Select train and validation. Notice that `idx_va_fold` will be
        # the fold that you use as validation set for this experiment
       
        #https://stackoverflow.com/questions/37370369/numpy-how-can-i-select-specific-indexes-in-an-np-array-for-k-fold-cross-validat
        x_tr = np.concatenate([x_trva[:idx_va_fold], x_trva[idx_va_fold+1:]])#x train data 4/5
        x_tr = np.reshape(x_tr, (x_tr.shape[0]*x_tr.shape[1], x_tr.shape[2]))
        y_tr = np.concatenate([y_trva[:idx_va_fold], y_trva[idx_va_fold+1:]]) #train data label 4/5
        y_tr = np.reshape(y_tr, (y_tr.shape[0]*y_tr.shape[1]))

        x_va = x_trva[idx_va_fold] # 1/5 validation data
        y_va = y_trva[idx_va_fold] # 1/5 validation label

        # ----------------------------------------
        # Train
        print("Training for fold {}...".format(idx_va_fold))
        # Run training
        cur_train_res = train(x_tr, y_tr, x_va, y_va, config)

        # Save results
        train_res += [cur_train_res]


    # TODO: Average results to see the average performance for this set of
    # hyper parameters on the validation set. This will be used to see how good
    # the design was. However, this should all be done *after* you are sure
    # your implementation is working. Do check how the training is going on by
    # looking at `loss_epoch` `tr_acc_epoch` and `va_acc_epoch`
    
    avg_loss_epoch = np.zeros(np.array(train_res[0]['loss_epoch']).shape)
    avg_tr_acc_epoch = np.zeros(np.array(train_res[0]['tr_acc_epoch']).shape)
    avg_va_acc_epoch = np.zeros(np.array(train_res[0]['va_acc_epoch']).shape)
    cnt = 0

    for fold in train_res:
        avg_best_acc += fold['best_acc']
        avg_W_best += fold['W_best']
        avg_b_best += fold['b_best']
        avg_loss_epoch += np.array(fold['loss_epoch'])
        avg_tr_acc_epoch += np.array(fold['tr_acc_epoch'])
        avg_va_acc_epoch += np.array(fold['va_acc_epoch'])
        cnt += 1

        if(fold['best_acc'] > result_acc):
            result_W = fold['W_best']
            result_b = fold['b_best']

    avg_best_acc /= cnt
    avg_W_best /= cnt
    avg_b_best /= cnt
    avg_loss_epoch /= cnt
    avg_tr_acc_epoch /= cnt
    avg_va_acc_epoch /= cnt

    # TODO: Find model with best validation accuracy and test it. Remeber you
    # don't want to use this result to make **any** decisions. This is purely
    # the number that you show other people for them to evaluate your model's
    # performance.
    print("Test data accuracy: ", compute_acc(predict(result_W, result_b, x_te, config), y_te) * 100, "%")

if __name__ == "__main__":

    # ----------------------------------------
    # Parse configuration
    config, unparsed = get_config()
    # If we have unparsed arguments, print usage and exit
    if len(unparsed) > 0:
        print_usage()
        exit(1)

    main(config)
