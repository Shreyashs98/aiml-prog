import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.mixture import GaussianMixture

iris = load_iris()
df = pd.DataFrame(iris['data'], columns=iris['feature_names'])
df['target'] = iris['target']
X = df.iloc[:, :-1]
y = df['target']

scaler = preprocessing.StandardScaler()
X_scaled = scaler.fit_transform(X)  
print(X_scaled)
plt.figure(figsize=(14, 7))
colormap = np.array(['red', 'green', 'blue'])

def plot_data(ax, title, data, y_pred):
    ax.scatter(data[:, 2], data[:, 3], c=colormap[y_pred], s=40)
    ax.set_title(title)

kmeans = KMeans(n_clusters=3, random_state=0).fit(X_scaled)
plot_data(plt.subplot(1, 3, 1), 'Real', X_scaled, y)
plot_data(plt.subplot(1, 3, 2), 'KMeans', X_scaled, kmeans.labels_)

gmm = GaussianMixture(n_components=3, max_iter=200).fit(X_scaled)
plot_data(plt.subplot(1, 3, 3), 'GMM Classification', X_scaled, gmm.predict(X_scaled))

plt.show()
