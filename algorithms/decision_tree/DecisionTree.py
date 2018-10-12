import pandas as pd
import numpy as np


class DecisionTree:

    def __init__(self, max_depth=3, min_samples=5):
        self.max_depth = max_depth
        self.min_samples = min_samples

    @staticmethod
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
            if (best_SSE is None) or (SSE < best_SSE):
                best_SSE = SSE
                best_split = {
                    'SSE': SSE,
                    'split_point': row['x'],
                    'left': branches[0],
                    'right': branches[1]
                }
        return best_split

    def iterate(self, set_, node={}, depth=1):
        if depth >= self.max_depth:
            # Return value
            node['value'] = set_['y'].mean()
            return
        if len(set_) <= self.min_samples:
            node['value'] = set_['y'].mean()
            return

        # Calculate best split and get groups
        split = self.find_split(set_)
        node['split_point'] = split['split_point']
        node['split_SSE'] = split['SSE']

        node['left'] = {'depth': depth}
        self.iterate(split['left'], node['left'], depth+1)

        node['right'] = {'depth': depth}
        self.iterate(split['right'], node['right'], depth+1)
        return node

    def predict(self, node, row):
        if 'value' in node:
            return node['value']

        if row < node['split_point']:
            return self.predict(node['left'], row)
        else:
            return self.predict(node['right'], row)


if __name__ == '__main__':
    from sklearn.datasets import load_iris

    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)

    set_ = pd.DataFrame(data={'x': df['sepal length (cm)'],
                              'y': df['sepal width (cm)']})

    tree = DecisionTree()
    iris = tree.iterate(set_)

    print(iris)
    #iris = iterate({}, set_, 1)

# ## Next Steps
# - Implement support for handling multiple features (including categorical)
# - Support taking DF for predictions, to properly assess performance
# - Write GBM that can use the trees to make more powerful predictions
