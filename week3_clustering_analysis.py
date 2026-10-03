"""Week 3: Unsupervised learning and clustering on public Iris dataset."""
import os, pandas as pd, matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score
OUT="clustering_outputs"; os.makedirs(OUT,exist_ok=True)
iris=load_iris(as_frame=True); X=iris.data.copy()
X.columns=["sepal_length_cm","sepal_width_cm","petal_length_cm","petal_width_cm"]
reference=iris.target
Xs=StandardScaler().fit_transform(X)
ks=range(2,9); inertia=[]; sils=[]
for k in ks:
    m=KMeans(n_clusters=k,random_state=42,n_init=20); y=m.fit_predict(Xs)
    inertia.append(m.inertia_); sils.append(silhouette_score(Xs,y))
for vals,title,ylabel,file in [(inertia,"K-Means Elbow Method","Inertia","elbow.png"),(sils,"Silhouette Score by k","Silhouette score","silhouette.png")]:
    plt.figure(figsize=(7,4)); plt.plot(list(ks),vals,marker="o"); plt.title(title); plt.xlabel("Number of clusters (k)"); plt.ylabel(ylabel); plt.grid(alpha=.25); plt.tight_layout(); plt.savefig(os.path.join(OUT,file),dpi=150); plt.close()
model=KMeans(n_clusters=3,random_state=42,n_init=20); labels=model.fit_predict(Xs)
print("Silhouette:",silhouette_score(Xs,labels))
print("Davies-Bouldin:",davies_bouldin_score(Xs,labels))
print("Calinski-Harabasz:",calinski_harabasz_score(Xs,labels))
print("ARI (post-hoc only):",adjusted_rand_score(reference,labels))
results=X.copy(); results["cluster"]=labels
print("Cluster means (original units):\\n",results.groupby("cluster")[X.columns].mean().round(2))
results.to_csv(os.path.join(OUT,"iris_clustered_results.csv"),index=False)
pca=PCA(n_components=2); Z=pca.fit_transform(Xs)
plt.figure(figsize=(7,5))
for c in range(3):
    mask=labels==c; plt.scatter(Z[mask,0],Z[mask,1],label=f"Cluster {c}",alpha=.8)
plt.title(f"PCA cluster projection ({pca.explained_variance_ratio_.sum()*100:.1f}% variance)"); plt.xlabel("PC1"); plt.ylabel("PC2"); plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(OUT,"pca_clusters.png"),dpi=150); plt.close()
agg=AgglomerativeClustering(n_clusters=3,linkage="ward").fit_predict(Xs)
print("Agglomerative silhouette:",silhouette_score(Xs,agg))
print("All outputs saved to",OUT)
