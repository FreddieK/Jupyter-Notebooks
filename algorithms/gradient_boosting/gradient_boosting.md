
## GBM
The workhorse of modern machine learning. After falling down the rabbit hole of understanding the construction of regressiontree (and creating my own implementation), I am now ready to plug it into a [gradient boosting machine](https://en.wikipedia.org/wiki/Gradient_boosting).

Below, my notes on the algorithm, but first some links to material that does a much better job of explaining it [here](http://explained.ai/gradient-boosting/index.html) and [here](http://www.chengli.io/tutorials/gradient_boosting.pdf).

### Pseudo algorithm
- Create an initial estimator (based on target variable mean) for data
- Based on error in initial prediction, recursively add a regression tree that tries to minimise the error of the previous prediction

### Intuition
The additive forward expansion of the residual is really an example of gradient descent, with each expansion signifying another step taken. 

Using squared loss, which is the most intuitive;

$$L(y, F(x)) = \frac{(y − F(x))^2}{2}$$
$$J = \sum{L(y_i, F(x_i))}$$

$$\frac{\partial J}{\partial F(x_i)} = F(x_i) - y_i$$

$$y_i - F(x_i) = - \frac{\partial J}{\partial F(x_i)}$$

Thus, we can reformulate;

$$F(x_i) := F(x_i) + h(x_i)$$
$$F(x_i) := F(x_i) + y - F(x_i)$$
$$F(x_i) := F(x_i) - 1\frac{\partial J}{\partial F(x_i)}$$
$$\theta_i := \theta_i - \lambda \frac{\partial J}{\partial \theta_i}$$

Where $lambda$ signifies the step size.

#### Using L1 (Absolute) Loss
Alternatives to the squared loss include Absolute loss and Huber loss which both are more robust to outliers in the data.

Since absolute loss derives to: 

$$- \frac{\partial J}{\partial F(x_i)} = sign(y_i - F(x_i))$$

Important to note is that the absolute loss is used when constructing the trees, but the predicted target value of the tree will be [the median of the examples](http://explained.ai/gradient-boosting/L1-loss.html#sec:1.1) in the leaf (compared to the mean for squared loss).

#### GBM for Classification
Train one model per class predicting 0, 1. I assume, you can add a softmax in the end to get something like class probabilities, where the sum of probabilities equals 1.


```python
# Verify right venv is used
import platform
assert platform.python_version() == '3.7.0'
```

## Regression target variable
So, let's get to it. To keep it simple, we'll predict a regression target variable and use L2 loss.

Since my custom tree model is not very optimized (profiling the performance and optimising might be a nice next challenge, including rewriting parts in Cython), the trees are a bit slow to build so I limit to ten trees and set a relatively high learning rate.


```python
from sklearn.datasets import load_boston
import pandas as pd
import numpy as np

data = load_boston()
X = data.data
y = data.target

X_df = pd.DataFrame(X, columns=data.feature_names)
```


```python
X_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CRIM</th>
      <th>ZN</th>
      <th>INDUS</th>
      <th>CHAS</th>
      <th>NOX</th>
      <th>RM</th>
      <th>AGE</th>
      <th>DIS</th>
      <th>RAD</th>
      <th>TAX</th>
      <th>PTRATIO</th>
      <th>B</th>
      <th>LSTAT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.00632</td>
      <td>18.0</td>
      <td>2.31</td>
      <td>0.0</td>
      <td>0.538</td>
      <td>6.575</td>
      <td>65.2</td>
      <td>4.0900</td>
      <td>1.0</td>
      <td>296.0</td>
      <td>15.3</td>
      <td>396.90</td>
      <td>4.98</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.02731</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0.0</td>
      <td>0.469</td>
      <td>6.421</td>
      <td>78.9</td>
      <td>4.9671</td>
      <td>2.0</td>
      <td>242.0</td>
      <td>17.8</td>
      <td>396.90</td>
      <td>9.14</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.02729</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0.0</td>
      <td>0.469</td>
      <td>7.185</td>
      <td>61.1</td>
      <td>4.9671</td>
      <td>2.0</td>
      <td>242.0</td>
      <td>17.8</td>
      <td>392.83</td>
      <td>4.03</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.03237</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0.0</td>
      <td>0.458</td>
      <td>6.998</td>
      <td>45.8</td>
      <td>6.0622</td>
      <td>3.0</td>
      <td>222.0</td>
      <td>18.7</td>
      <td>394.63</td>
      <td>2.94</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.06905</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0.0</td>
      <td>0.458</td>
      <td>7.147</td>
      <td>54.2</td>
      <td>6.0622</td>
      <td>3.0</td>
      <td>222.0</td>
      <td>18.7</td>
      <td>396.90</td>
      <td>5.33</td>
    </tr>
  </tbody>
</table>
</div>




```python
import sys
sys.path.insert(0, '/Users/freddie.karlbom/dev/algorithms')
from decisiontree.decisiontree import DecisionTree
# Importing my custom regression tree class, can be found at
# https://github.com/FreddieK/algorithms-in-python
```


```python
y_pred = np.mean(y) # initial model

learning_rate = 0.5
nr_trees = 10
```


```python
print(DecisionTree.score(y, y_pred)) # initial model mae

for i in range(nr_trees):
    dx = y - y_pred
    tree = DecisionTree()
    
    dx_df = pd.DataFrame(dx)
    dx_df.columns = ['y']

    tree.build_tree(X_df, dx_df)
    y_pred += learning_rate*np.array(tree.predict(X_df))
    
    mae = DecisionTree.score(y, y_pred)
    print(mae)
```

    6.647207423956008
    4.64223543227224
    3.5060203764688502
    3.0035297382705246
    2.7370645289421143
    2.5532973608407707
    2.3876131975762247
    2.27056874997753
    2.201764300656173
    2.139500731765435
    2.0867045115081146


## Takeaways
- Seeing the mean average error decrease consistently between iterations means the model keep improving with each new tree added
- When I decided to get more familiar with gradient boosting, I had no clue that I would end up most of the time reading up on decision trees
- Note to self: when refactoring this code into a class for my repository, I'll store all the trees in the model so that predictions can be made on a hold-out set post training as well...
