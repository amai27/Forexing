#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      LiaAmai
#
# Created:     14/12/2015
# Copyright:   (c) LiaAmai 2015
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import csv
import numpy as np
import random as rand

def baca_file(filename):

    data = []
    with open(filename, 'rb') as csvfile:
        for rows in csv.reader(csvfile):
            fixed_data = map(float, rows[2:6])
            data.append(fixed_data);

    return data;


def centroid_awal(data, k):
    """
    mengambil secara acak data sebanyak jlh kluster(k)
    """
    return np.array(rand.sample(data, k))

def jarak(v1, v2):
    """
    menghitung jarak dengan euclidean distance
    v1 dan v2 adalah data vektornya
    """
    return np.sqrt(sum(np.power(v1 - v2, 2)))

def pembagian_kelas(data, centroids):
    """
    membagi kelas, yg memiliki jarak terkecil akan masuk ke k kluster
    """
    k, dim = centroids.shape
    clusters = list()
    clusters = []
    for i in xrange(k):
        clusters.append([])
    for elemen in data:
        min_dist = 1000
        min_index = 0  # index terdekat centroid
        for index, centroid in enumerate(centroids):
            this_dist = jarak(elemen, centroid)
            #jika jarak sekarang lebih kecil dari min_dist maka min_dist = this_dist
            if this_dist < min_dist:
                min_dist = this_dist
                min_index = index
        clusters[min_index].append(elemen)
    return np.array(clusters)

#menghitung nilai centroid
def mean(vector):
    """
    menghitung nilai rata-rata anggota setiap kluster
    yang hasilnya akan jadi centroid baru di iterasi selanjutnya
    """
    return np.mean(vector, axis=0)

#mengganti centroid
def ubah_centroid(clusters):
    """
    mengganti centroid lama dengan centroid baru
    """
    centroids = []
    for cluster in clusters:
        centroids.append(mean(cluster))
    return np.array(centroids)

#untuk mengecek apakah centroid berubah atau tidak
def centroid_baru(last_centroids, centroids):
    """
    jika centroid berubah
    """
    return (np.sort(last_centroids) == np.sort(centroids)).all()

#kmean : jika centroid sudah sama, maka last centroid adalah centroid terakhir
def kmeans(data, k):
    last_centroids = centroid_awal(data, k)
    #print "\ncentroid awal : \n"
    #print last_centroids
    clusters = pembagian_kelas(data, last_centroids)
    centroids = ubah_centroid(clusters)
    while not centroid_baru(last_centroids, centroids):
        clusters = pembagian_kelas(data, centroids)
        last_centroids = centroids
        centroids = ubah_centroid(clusters)

    #print "\npembagian kelas: \n"
    #print clusters
    print "centroid: \n",centroids
    return centroids

def main():
    direktori = str(raw_input("Masukkan nama data dengan direktorinya: "))
    num_input_layer = int(input("masukan jumlah layer input (input nodes): " ))
    num_hidden_layer = int(input("masukan jumlah layer tersembunyi (hidden nodes): " ))
    num_output_layer = int(input("masukan jumlah layer output (output nodes): " ))

    #jlh kluster
    data = baca_file(direktori)
    k = num_hidden_layer
    kmeans(data,k)


if __name__ == '__main__':
    main()


