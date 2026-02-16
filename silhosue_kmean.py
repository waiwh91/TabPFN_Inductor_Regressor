from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import pandas as pd

def silhouse_kmean(outputs):

    scaler = StandardScaler()
    outputs_scaled = scaler.fit_transform(outputs)

    # 轮廓系数法选择簇数
    sil_scores = []
    k_values = range(2, 30)
    for k in k_values:
        labels = KMeans(n_clusters=k, random_state=42).fit_predict(outputs_scaled)
        sil_scores.append(silhouette_score(outputs_scaled, labels))

    # plt.plot(k_values, sil_scores, marker='o')
    # plt.xlabel("Number of clusters")
    # plt.ylabel("Silhouette Score")
    # plt.show()

    best_k = k_values[sil_scores.index(max(sil_scores))]
    print("Best cluster number:", best_k)

    # 使用最佳簇数进行最终聚类
    kmeans = KMeans(n_clusters=best_k, random_state=42).fit(outputs_scaled)
    cluster_centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    cluster_centers = scaler.inverse_transform(cluster_centers)

    print("Cluster centers:", cluster_centers)
    print("Cluster labels:", labels)

    return cluster_centers


    output_df = pd.DataFrame(
        {"tCu": cluster_centers[:, 0], "wCu": cluster_centers[:, 1], "tLam": cluster_centers[:, 2], "nLam": cluster_centers[:, 3],
         "aln": cluster_centers[:, 4], "tsu": cluster_centers[:, 5]})

    output_df.to_csv("kmean_designs.csv", index=False)


if __name__ == "__main__":
    df = pd.read_csv("designs.csv").to_numpy()
    results = df[:,:7]

    silhouse_kmean(results)