
# coding: utf-8

# In[23]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd  
from sklearn import svm, datasets
from sklearn.linear_model import LogisticRegression


# In[13]:


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
colnames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
irisdata = pd.read_csv(url, names=colnames) 


# In[14]:


X = irisdata.drop('Class', axis=1)  
y = irisdata['Class']  


# In[15]:


X.head()


# In[16]:


from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20,random_state =1234)  


# In[17]:


from sklearn.svm import SVC  
from sklearn.metrics import classification_report, confusion_matrix  
def polynomial_kernel():
    print("Polynomial Kernel")
    svclassifier = SVC(kernel='poly',degree=8)  
    svclassifier.fit(X_train, y_train) 
    y_pred = svclassifier.predict(X_test)  
    print(confusion_matrix(y_test,y_pred))  
    print(classification_report(y_test,y_pred))  

def gaussian_kernel():
    print("Gaussian Kernel")
    svclassifier = SVC(kernel='rbf')  
    svclassifier.fit(X_train, y_train) 
    y_pred = svclassifier.predict(X_test)  
    print(confusion_matrix(y_test,y_pred))  
    print(classification_report(y_test,y_pred))  

def sigmoid_kernel():
    print("Sigmoid Kernel")
    svclassifier = SVC(kernel='sigmoid')  
    svclassifier.fit(X_train, y_train) 
    y_pred = svclassifier.predict(X_test)  
    print(confusion_matrix(y_test,y_pred))  
    print(classification_report(y_test,y_pred))  

def test():
#     import_iris()
    polynomial_kernel()
    gaussian_kernel()
    sigmoid_kernel()

test()


# In[18]:


from sklearn import datasets
iris = datasets.load_iris()


# In[19]:


def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


# In[20]:


iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target

# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
C = 1.0  # SVM regularization parameter
models = (svm.SVC(kernel='linear',),
          svm.SVC(kernel='rbf',),
          svm.SVC(kernel='poly', degree=8))
models = (clf.fit(X, y) for clf in models)

# title for the plots
titles = ('linear kernel',
          'RBF kernel',
          'polynomial (degree 8) kernel')


fig, sub = plt.subplots(1, 3)
plt.subplots_adjust(wspace=0.5, hspace=0.4)

X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X0, X1)

for clf, title, ax in zip(models, titles, sub.flatten()):
    plot_contours(ax, clf, xx, yy,
                  cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Sepal length')
    ax.set_ylabel('Sepal width')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)
plt.show()


# In[21]:


X = iris.data[:, 2:]
y = iris.target
C = 1.0  # SVM regularization parameter
models = (svm.SVC(kernel='linear',),
          svm.SVC(kernel='rbf',),
          svm.SVC(kernel='poly', degree=8))
models = (clf.fit(X, y) for clf in models)

# title for the plots
# titles = ('SVC with linear kernel',
#           'SVC with RBF kernel',
#           'SVC with polynomial (degree 8) kernel')

# Set-up 2x2 grid for plotting.
fig, sub = plt.subplots(1, 3)
plt.subplots_adjust(wspace=0.4, hspace=0.4)

X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X0, X1)

for clf, title, ax in zip(models, titles, sub.flatten()):
    plot_contours(ax, clf, xx, yy,
                  cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Petal length')
    ax.set_ylabel('Petal width')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)
plt.show()


# In[24]:


X = iris.data[:, 2:]  # we only take the first two features.
Y = iris.target
logreg = LogisticRegression(multi_class="multinomial",solver="lbfgs", C=10, random_state=42)

# Create an instance of Logistic Regression Classifier and fit the data.
logreg.fit(X, Y)
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
h = .02  # step size in the mesh
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1, figsize=(4, 3))
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel('Petal length')
plt.ylabel('Petal width')

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())

plt.show()

