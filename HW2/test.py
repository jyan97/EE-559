from scipy.spatial.distance import cdist


a = [1,2]
b = [2,3]
c = cdist(a,b,metric='euclidean')
print(c)