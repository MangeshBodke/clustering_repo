import importlib
from tkinter import OFF
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import figure
import mpld3
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from flask import Flask, render_template


drop_Columns = ['Medical_Keyword_8', 'Medical_Keyword_48', 'Medical_Keyword_23', 'Medical_Keyword_30', 'Medical_Keyword_37', 'Medical_Keyword_36', 'Medical_Keyword_21', 'Medical_Keyword_32', 'Medical_Keyword_24', 'Medical_Keyword_22', 'Medical_Keyword_1', 'Medical_Keyword_4', 'Medical_Keyword_46', 'Medical_Keyword_45', 'Medical_Keyword_33', 'Id', 'Medical_Keyword_15', 'Product_Info_2', 'Medical_Keyword_11', 'Medical_Keyword_43', 'Medical_Keyword_27', 'Medical_Keyword_47', 'Medical_Keyword_19', 'Medical_Keyword_29', 'Medical_Keyword_40', 'Medical_Keyword_38', 'Medical_History_24', 'Medical_Keyword_28', 'Medical_Keyword_14', 'Medical_Keyword_39', 'Medical_Keyword_7', 'Medical_Keyword_10', 'Medical_Keyword_20', 'Medical_Keyword_25', 'Medical_Keyword_12', 'BMI', 'Medical_Keyword_5', 'Medical_Keyword_9', 'Medical_Keyword_16', 'Medical_History_15', 'Medical_Keyword_31', 'Medical_Keyword_35', 'Medical_History_10', 'Medical_Keyword_6', 'Medical_Keyword_13', 'Medical_History_32', 'Medical_Keyword_26', 'Medical_Keyword_2', 'Medical_Keyword_34', 'Medical_Keyword_44', 'Medical_Keyword_42', 'Medical_Keyword_17', 'Medical_Keyword_3', 'Medical_Keyword_41', 'Medical_Keyword_18']

ins_data= pd.read_csv('test.csv')

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
ins_data['Product_Info_2']= le.fit_transform(ins_data['Product_Info_2'])

from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
ins_data_scaled = make_pipeline(SimpleImputer(strategy='mean'), StandardScaler()).fit_transform(ins_data)
ins_data = pd.DataFrame(ins_data_scaled, columns=ins_data.columns)
# print(ins_data.describe())


pca_model = pickle.load(open('pca_model.pkl', 'rb'))

pca_2_result = pca_model.fit_transform(ins_data)

# print(pca_2_result)
data_pca = pd.DataFrame(abs(pca_model.components_), columns=ins_data.columns, index=['PC_1', 'PC_2'])

# print(data_pca)
# print('\n As per PC_1:\n', (data_pca[data_pca > 0.3].iloc[0]).dropna())
# print('\n As per PC_2 \n', (data_pca[data_pca > 0.3]).iloc[1].dropna())

# print(pca_2_result, pca_model)

parameters = [2, 3, 4, 5, 10, 15]
param_grid = ParameterGrid({'n_clusters': parameters})
best_score = -1

kmeans_model = pickle.load(open('kmeans_model.pkl', 'rb'))

silhouette_scores = []
from sklearn.metrics import silhouette_score
for p in param_grid:
    kmeans_model.set_params(**p) # set current hyper parameter
    kmeans_model.fit(ins_data) # fit model on insurance dataset, this will find clusters based on parameter p
    ss = silhouette_score(ins_data, kmeans_model.labels_)  # calculate silhouette_score
    silhouette_scores += [ss]  # store all the scores
    print('Parameter:', p , 'Score:', ss)
    # check p which has the best score
    if ss > best_score:
        best_score = ss
        best_grid = p

kmeans_cluster_model = pickle.load(open('kmeans_cluster_model.pkl', 'rb'))
optimum_num_clusters = best_grid['n_clusters']
# kmeans = KMeans(n_clusters=optimum_num_clusters)
kmeans_cluster_model.fit(ins_data)
centroids = kmeans_cluster_model.cluster_centers_
centroids_pca = pca_model.transform(centroids)
# print(centroids_pca)

app = Flask(__name__)
@app.route('/')
@app.route('/clustering')
def clustering():
    x = pca_2_result[:, 0]
    y = pca_2_result[:, 1]
    fig , ax = plt.subplots(figsize=(12,6))
    plt.scatter(x, y, c=kmeans_cluster_model.labels_, alpha=0.5, s=200) # plot different colors per cluster
    plt.title('Insurance Clusters')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.scatter(pca_model.transform(centroids)[:,0], centroids_pca[:,1], marker='x', s=200, linewidths=1.5, color='red', edgecolors='black', lw=1.5)
    mpld3.show()
    # fig.savefig('clustering_image\image\clustering.png')
    mpld3.save_html(fig, 'clustering.html')
    return mpld3.fig_to_html(fig)


if __name__ == "__main__":
    import random, threading, webbrowser
    port = 5000 + random.randint(0, 999)
    url = "http://127.0.0.1:{0}".format(port)
    threading.Timer(1.25, lambda: webbrowser.open(url) ).start()
    app.run(host='0.0.0.0', port=8080, debug=True, use_reloader=False)
