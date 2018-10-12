
# Decision Trees (CART)

With some time on my hand, I decided I wanted to dig deeper into some of the workhorse algorithms of modern machine learning, namely gradient boosting machines and random forest.

Starting with [GBM](https://en.wikipedia.org/wiki/Gradient_boosting) (and AdaBoost), the basic intuition behind the algorithm turned out to be almost embarassingly obvious (will return to that in another post), but the devil is really in the details.

What I had a hard time understanding was rather how a decision tree could be used to predict a continuous target variable, as well as for how the breakpoints would be set for continuous predictor variables.

Thus began some digging and experimenting.

## Pseudo algorithm
1. Starting with the complete dataset, identify the feature and threshold that has the highest prediction power and split the tree based on that.
2. Recursively keep growing the tree greedily until stopping criterias are met (depth, lack of improvement, threshold on min number of examples in the leaf etc.).
3. Prune tree, and other optimisations.

## Classification
When reading online, you find tons of examples of building a classification tree, either using [Gini impurity](https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity) or Information Gain. Based in information theory, the idea is to reduce the entropy (amount of information needed to describe the data) as much as possible with each split.

### Information Gain / Entropy
$$Entropy(S) = \sum_{i=1}^c{-p_i\log_2{p_i}}$$

$$Gain(S,A) = Entropy(S) - \sum_{v \in Values(A)}{\frac{|S_v|}{|S|}Entropy(S_v)}$$

Where $Values(A)$ is the set of possible values for the
attribute $A$ and $S_v$ is the subset of $S$ for which attribute
$A$ has value $v$.

Information Gain is the expected reduction in entropy caused by knowing the value of attribute A. The attribute which will mean the largest information gain will be chosen for the split.

### Gain Ratio
Information gain favours attributes with many possible values. An alternative is to calculate the gain ratio.

$$GainRatio(S,A) = \frac{Gain(S,A)}{SplitInfo(S,A}$$

$$SplitInfo(S,A) = -\sum_{i=1}^c{\frac{|S_i|}{|S|}\log_2{\frac{|S_i|}{|S|}}}$$

This can explode (or become undefined) if $S≈S_i$, in order to avoid this you can calculate the gain ratio only on attributes with higher than average gain.

### Handling continuous predictors
Dynamically create factor variables $A_c$ that is true if $A < c$.

1. Identify adjacent examples of $A$ with differing target values (if the target is the same, the break point won't make any sense).
2. Generate $c$ suggestion as the mean between the adjacent rows.
3. Repeat for all examples until you have a list of candidate breakpoints.
4. Identify which candidate suggestion that maximises information gain (and compare that to other attributes).

## Regression
Since the Gradient Boosting algorithm uses additive forward expansion to build an ensemble model out of multiple trees, each tree will need to output values that can be summed up.

Thus, we need regression trees that predicts a continuous value and not a class.

### Splitting points
Instead of information gain, we can compare the squared error in the generated sets based on different splits and choose the split that minimises the error. $\hat{y}$ in each leaf is the mean of the rows in that branch.

Using MSE (CART standard):

$$MSE = \frac{1}{n}\sum_{i=1}^n{(\hat{Y_i}-Y_i)^2}$$

Which for the branch after a split becomes:

$$MSE = \frac{1}{n}\sum_{k=1}^2{\sum_{i=1}^n{(\hat{Y_{ki}}-Y_{ki})^2}}$$

After comparison, the tree candidate with the lowest MSE is chosen.

I ended up using the summed squared error (SSE), but since the number of items it the same for all candidates, it's functionally equivalent.

### How the final leaf prediction is calculated
Like when comparing the different splits, the final leafs of the tree uses the mean of the rows for prediction.

An alternative is to select the median, which is used when using [a GBM with L1 loss to build the tree](http://explained.ai/gradient-boosting/L1-loss.html) (L1 instead of L2 to give outliers less of an impact on the trees).

## Post Construction

### Pruning
After tree is constructed, compare the predictions at each node with a validation set. If a tree after a certain point gives no improvement (or leads to worse result) going deeper, then remove the subtree and replace the node with a leaf node of the most common class.

#### Rule Post Pruning
Convert tree to rules, with one rule per branch. Then prune rules by removing preconditions if it leads to an improved estimated accuracy. Sort the pruned rules according to estimated accuracy to ensure the most accurate rules are considered before moving down to less accurate.

## Implementation
Finding the information about how to do the splitting for the regression tree was among the harder things, as most material I found only dealt with classification trees and just mentioned regression in passing.

Once I'd pieced together the information for how to create a split, it took me a few tries to get the recursion of the tree right (clearly out of practice implementing algorithms.

On the third try, I got something decent enough to serve as a base.


```python
import pandas as pd
import numpy as np

max_depth = 3
min_samples = 5

def find_split(set_):
    best_SSE = None
    best_split = None
    for index, row in set_.iterrows():
        SSE = 0
        branches = [set_[set_['x'] < row['x']], 
                    set_[~(set_['x'] < row['x'])]]
        for branch in branches:
            y_pred = branch['y'].mean()
            SSE += np.sum((y_pred - branch['y'])**2)
        if (best_SSE == None) or (SSE < best_SSE):
            best_SSE = SSE
            best_split = {
                'SSE' : SSE,
                'split_point' : row['x'],
                'left' : branches[0],
                'right' : branches[1]
            }
    return best_split


def iterate(node, set_, depth):
    if depth >= max_depth:
        # Return value
        node['value'] = set_['y'].mean()
        return
    if len(set_) <= min_samples:
        node['value'] = set_['y'].mean()
        return

    # Calculate best split and get groups
    split = find_split(set_)
    node['split_point'] = split['split_point']
    node['split_SSE'] = split['SSE']
    
    node['left'] = {'depth': depth}
    iterate(node['left'], split['left'], depth+1)
    
    node['right'] = {'depth': depth}
    iterate(node['right'], split['right'], depth+1)
    return node


def predict(node, row):
    if 'value' in node:
        return node['value']

    if row < node['split_point']:
        return predict(node['left'], row)
    else:
        return predict(node['right'], row)
```


```python
from sklearn.datasets import load_iris
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

set_ = pd.DataFrame(data={'x': df['sepal length (cm)'], 
                   'y': df['sepal width (cm)']})

iris = iterate({}, set_, 1)
```


```python
iris
```




    {'split_point': 5.5,
     'split_SSE': 25.411134222919934,
     'left': {'depth': 1,
      'split_point': 5.1,
      'split_SSE': 8.087187499999999,
      'left': {'depth': 2, 'value': 3.090625},
      'right': {'depth': 2, 'value': 3.5}},
     'right': {'depth': 1,
      'split_point': 6.7,
      'split_SSE': 14.528428571428572,
      'left': {'depth': 2, 'value': 2.9014285714285712},
      'right': {'depth': 2, 'value': 3.0928571428571425}}}




```python
print(predict(iris, 5))
```

    3.090625


## Takeaways
- The basic algorithm is surprisingly simple once you wrap your head around it
- It becomes clear from an information theory perspective that Machine Learning basically is the same as data compression. From a high entropy dataset, we produce a more condensed model that with some loss is able to recreate the initial set.

## Next Steps
- Implement support for handling multiple features (including categorical)
- Support taking DF for predictions, to properly assess performance
- Write GBM that can use the trees to make more powerful predictions
