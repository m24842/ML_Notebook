import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical
import ssl
import warnings
ssl._create_default_https_context = ssl._create_unverified_context

warnings.filterwarnings('ignore')

np.random.seed(1234)

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

X_train = X_train.reshape(60000, 28*28)
X_test = X_test.reshape(10000, 28*28)

nb_classes = 10
y_train = to_categorical(y_train, nb_classes)
y_test = to_categorical(y_test, nb_classes)

# import of sigmoid and crossentropy
from scipy.special import expit
from sklearn.metrics import log_loss

def forward_pass(W1, W2, b1, b2, x):
    '''This is the forward pass. It is equal for any
    training algorithm. It's just one hidden layer
    with tanh activation function and sigmoid on the
    output layer'''
    # if the input is a batch, I have to tile as many
    # b1 and b2 as the batch size
    a1 = np.matmul(W1, x)+np.tile(b1, x.shape[1])
    h1 = np.tanh(a1)
    a2 = np.matmul(W2, h1)+np.tile(b2, x.shape[1])
    y_hat = expit(a2)
    return a1, h1, a2, y_hat

def backprop_backward_pass(e, h1, W2, a1, x):
    dW2 = -np.matmul(e, np.transpose(h1))
    da1 = np.matmul(np.transpose(W2), e)*(1-np.tanh(a1)**2)
    dW1 = -np.matmul(da1, np.transpose(x))
    db1 = -np.sum(da1, axis=1)
    db2 = -np.sum(e, axis=1)
    return dW1, dW2, db1[:,np.newaxis], db2[:,np.newaxis]

def dfa_backward_pass(e, h1, B1, a1, x):
    dW2 = -np.matmul(e, np.transpose(h1))
    da1 = np.matmul(B1, e)*(1-np.tanh(a1)**2)
    dW1 = -np.matmul(da1, np.transpose(x))
    db1 = -np.sum(da1, axis=1)
    db2 = -np.sum(e, axis=1)
    return dW1, dW2, db1[:,np.newaxis], db2[:,np.newaxis]

def average_angle(W2, B1, error, a1, a2):
    dh1 = np.mean(np.matmul(B1, error), axis=1)[:, np.newaxis] #forse non ci va la derivata
    c1 = np.mean(np.matmul(np.transpose(W2), error*(expit(a2)*(1-expit(a2)))), axis=1)[:, np.newaxis]
    dh1_norm = np.linalg.norm(dh1)
    c1_norm = np.linalg.norm(c1)
    inverse_dh1_norm = np.power(dh1_norm, -1)
    inverse_c1_norm = np.power(c1_norm, -1)
    
    # ALIGNMENT CRITERION AND ANGLE
    Lk = (np.matmul(np.transpose(dh1), c1)*inverse_dh1_norm)[0, 0]
    beta = np.arccos(np.clip(Lk*inverse_c1_norm, -1., 1.))*180/np.pi
    return Lk, beta

def train(x, y, n_epochs=10, lr=1e-3, batch_size=200, tol=1e-1):
    x = np.transpose(x)
    y = np.transpose(y)
    
    W1, W2 = np.random.randn(800, 784), np.random.randn(10, 800)
    b1, b2 = np.random.randn(800, 1), np.random.randn(10, 1)
    
    dataset_size = x.shape[1]
    n_batches = dataset_size//batch_size
    te_bp = []
    for i in range(n_epochs):
        perm = np.random.permutation(x.shape[1])
        x = x[:, perm]
        y = y[:, perm]
        loss = 0.
        train_error = 0.
        for j in range(n_batches):
            samples = x[:, j*batch_size:(j+1)*batch_size]
            targets = y[:, j*batch_size:(j+1)*batch_size]
            a1, h1, a2, y_hat = forward_pass(W1, W2, b1, b2, samples)
            error = y_hat - targets
            preds = np.argmax(y_hat, axis=0) 
            truth = np.argmax(targets, axis=0)
            train_error += np.sum(preds!=truth)
            loss_on_batch = log_loss(targets, y_hat)
            
            dW1, dW2, db1, db2 = backprop_backward_pass(error, h1, W2, a1, samples)
            W1 += lr*dW1
            W2 += lr*dW2
            b1 += lr*db1
            b2 += lr*db2
            loss += loss_on_batch
        training_error = 1.*train_error/x.shape[1]
        print('Loss at epoch', i+1, ':', loss/x.shape[1])
        print('Training error:', training_error)
        prev_training_error = 0 if i==0 else te_bp[-1]
        if np.abs(training_error-prev_training_error) <= tol:
            te_bp.append(training_error)
            break
        te_bp.append(training_error)
    return W1, W2, b1, b2, te_bp

W1, W2, b1, b2, te_bp = train(X_train, y_train, n_epochs=100, lr=1e-3, batch_size=200, tol=1e-6)

def dfa_train(x, y, n_epochs=10, lr=1e-3, batch_size=200, tol=1e-3):
    x = np.transpose(x)
    y = np.transpose(y)
    
    W1, W2 = np.random.randn(800, 784), np.random.randn(10, 800)
    b1, b2 = np.random.randn(800, 1), np.random.randn(10, 1)
    
    B1 = np.random.randn(800, 10)
    dataset_size = x.shape[1]
    n_batches = dataset_size//batch_size
    te_dfa = []
    angles = []
    for i in range(n_epochs):
        perm = np.random.permutation(x.shape[1])
        x = x[:, perm]
        y = y[:, perm]
        loss = 0.
        train_error = 0.
        for j in range(n_batches):
            samples = x[:, j*batch_size:(j+1)*batch_size]
            targets = y[:, j*batch_size:(j+1)*batch_size]
            a1, h1, a2, y_hat = forward_pass(W1, W2, b1, b2, samples)
            error = y_hat - targets
            preds = np.argmax(y_hat, axis=0) 
            truth = np.argmax(targets, axis=0)
            train_error += 1.*np.sum(preds!=truth)
            loss_on_batch = log_loss(targets, y_hat)
            
            dW1, dW2, db1, db2 = dfa_backward_pass(error, h1, B1, a1, samples)
            W1 += lr*dW1
            W2 += lr*dW2
            b1 += lr*db1
            b2 += lr*db2
            loss += loss_on_batch
            if j%100==0:
                angles.append(average_angle(W2, B1, error, a1, a2))
        training_error = 1.*train_error/x.shape[1]
        print('Loss at epoch', i+1, ':', loss/x.shape[1])
        print('Training error:', training_error)
        prev_training_error = 0 if i==0 else te_dfa[-1]
        if np.abs(training_error-prev_training_error) <= tol:
            te_dfa.append(training_error)
            break
        te_dfa.append(training_error)
    return W1, W2, b1, b2, te_dfa, angles

W1dfa, W2dfa, b1dfa, b2dfa, te_dfa, angles = dfa_train(X_train, y_train, n_epochs=100, lr=1e-3, batch_size=200, tol=1e-6)

def test(W1, W2, b1, b2, test_samples, test_targets):
    test_samples = np.transpose(test_samples)
    test_targets = np.transpose(test_targets)
    outs = forward_pass(W1, W2, b1, b2, test_samples)[-1]
    preds = np.argmax(outs, axis=0) 
    truth = np.argmax(test_targets, axis=0)
    test_error = 1.*np.sum(preds!=truth)/preds.shape[0]
    return test_error

print('BP:', test(W1, W2, b1, b2, X_test, y_test)*100, '%')
print('DFA:', test(W1dfa, W2dfa, b1dfa, b2dfa, X_test, y_test)*100, '%')

plt.plot(range(len(te_bp)), te_bp, label='BP training error')
plt.plot(range(len(te_dfa)), te_dfa, label='DFA training error')
plt.title('Learning rate 1e-4')
plt.xlabel('Epochs')
plt.ylabel('Training error %')
plt.legend(loc='best')
plt.show()

l, beta = zip(*angles)

plt.plot(range(len(beta)), beta, label='angle')
plt.legend(loc='best')
plt.xlabel('Epoch*3')
plt.ylabel('Angle')
plt.show()


