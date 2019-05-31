import glob
import numpy as np
import csv
from sklearn.cluster import KMeans

fileNames = glob.glob('2019Proj2_train/*.txt')
# print(fileNames)


def fileReader(file,num_clusters = 50):
    '''
    This will read the file, and output a discretized value, based on Kmeans clustering
    
    input:
    file: Filename of training data used
    num_clusters: number of discrete values based on Kmeans clustering
    
    output:
    kmean.labels: [0,num_clusters) - exclusive of num_clusters number!
    '''
    a = []
    # Creating a list, appending to list, and then making a numpy array
    
    
    with open(file, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            a.append(row)
    tmp = np.array(a)
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(tmp)
    return kmeans.labels_

if __name__ == '__main__':

# # This should be trained on all of the data. After doing this I need to make the silos of data

    num_clusters = 50 # number of discrete values

    beat_three = fileNames[0:5]; beat3_data = np.array([fileReader(beat3) for beat3 in beat_three])
    beat_four = fileNames[5:10]; beat4_data = np.array([fileReader(beat4) for beat4 in beat_four])
    circle = fileNames[10:15]; circle_data = np.array([fileReader(circ) for circ in circle])
    eight = fileNames[15:20]; eight_data = np.array([fileReader(e) for e in eight])
    inf = fileNames[20:25]; inf_data = np.array([fileReader(i) for i in inf])
    wave = fileNames[25:30]; wave_data = np.array([fileReader(wv) for wv in wave])

    # print(wave_data[:10], "\nlast couple: \n", wave_data[-10:])

    print(inf_data[0][:10])