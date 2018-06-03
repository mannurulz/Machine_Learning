import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, centroid
import sys
#from blaze.expr.collections import sample
#from matplotlib.pyplot import axis

#File_Name=sys.argv[1]
#sample_file = np.loadtxt("/Users/mannu/Desktop/INF553/"+File_Name,delimiter=',')

sample_file = np.loadtxt("/Users/mannu/Desktop/INF553/sample_data.txt",delimiter=',')

#print(sample_file)
merging = linkage(sample_file, method='complete')
#print(merging)
dendrogram(merging,leaf_rotation=90,leaf_font_size=6)
plt.show()

#from sklearn.cluster import KMeans
#model = KMeans(n_clusters=4)
#X = model.fit(sample_file)
#labels = X.predict(sample_file)
#centroids = X.cluster_centers_
#print(type(sample_file))
#print(type(labels))
#print(sample_file.shape)
#print(labels.shape)
#Output_File=np.column_stack((sample_file, labels))
#print(Output_File)
#print("\n".join(str(x) for x in Output_File))
#print(*Output_File, sep =',')
#print(Output_File[0]+","+Output_File[1]+","+Output_File[2])


#print(np.append(sample_file, labels, axis=0))

#print(sample_file)
#print(np.append(sample_file, labels))
#print(np.unique(labels))
#print(centroids)

#label = fcluster(merging, 2, criterion='distance')
#print(label)
