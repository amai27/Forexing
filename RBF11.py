#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      LiaAmai
#
# Created:     18/12/2015
# Copyright:   (c) LiaAmai 2015
# Licence:     <your licence>
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      LiaAmai
#
# Created:     18/12/2015
# Copyright:   (c) LiaAmai 2015
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import random
import math
from kmeans import kmeans,baca_file
import numpy as np
np.seterr(divide='ignore', invalid='ignore')

class RadialNet():
    def __init__(self,JumlahInput,JumlahHidden,JumlahOutput):
        self.JumlahInput = JumlahInput
        self.JumlahHidden = JumlahHidden
        self.JumlahOutput = JumlahOutput
        self.inputs = [0 for i in range(JumlahInput-1)]
        self.centroids = [0 for i in range(JumlahHidden-1) for j in range(JumlahInput-1)]
        self.widths = [1 for i in range(JumlahHidden)]
        self.ihBobot = [0 for i in range(JumlahInput) for j in range(JumlahHidden)]
        self.hBias = [0 for i in range(JumlahHidden-1)]
        self.hoBobot = self.makematrix(JumlahHidden, JumlahOutput)
        self.oBias = [0 for i in range(JumlahOutput)]
        self.hOutputs = [0 for i in range(JumlahHidden-1)]
        self.outputs = [0 for i in range(JumlahOutput-1)]

    def makematrix(self, rows, cols):
        result = [[0 for j in range(cols)] for i in range(rows)]
        return result

    def RBF(self, data, jumlahIterasi):
        print "Mencari data centroid sebanyak",self.JumlahHidden,"buah\n"
        centroid = self.HitungCentroid(data,k)

        print "Mencari nilai width untuk masing-masing Hidden Nodes\n"
        width = self.HitungWidth(centroid)
        print width,"\n"

        jumlahBobot = (self.JumlahHidden * self.JumlahInput) + self.JumlahOutput
        print "Menentukan nilai bobot dan nilai bias sebanyak",jumlahBobot,"buah\n"
        bobot = self.inisialisasiBobot()
        print "\n",bobot


    def HitungCentroid(self, data,k):
        self.centroids = kmeans(data,k)
        return self.centroids

    def HitungWidth(self,centroids):
        jumlahJarak = 0.0
        ct = 0
        for i in range(len(centroids)-2):
            j=i+1
            for j in range(len(centroids)-1):
                jarak = self.jarakEuclidean(centroids[i], centroids[j], len(centroids[i]))
                jumlahJarak += jarak
                ct += 1

        ratarataJarak = jumlahJarak/ct
        width = ratarataJarak
        #print "Width rata-rata yang akan digunakan adalah: ",width
        for i in range(len(self.widths)-1):
            self.widths[i] = width
        return self.widths

    def getBobot(self):
        numWts = (self.JumlahHidden * self.JumlahOutput) + self.JumlahOutput
        result = [0 for i in range(numWts)]
        k = 0
        for i in range(self.JumlahHidden-1):
          for j in range(self.JumlahOutput-1):
            result[k] = self.hoBobot[i][j]
            k += 1
        for i in range(self.JumlahOutput-1):
          result[k] = self.oBias[i]
          k += 1
        return result

    def inisialisasiBobot(self):
        numWts = (self.JumlahHidden * self.JumlahOutput) + self.JumlahOutput
        wts = [0 for i in range(numWts)]
        lo = -0.01
        hi = 0.01
        for i in range(len(wts)):
          wts[i] = (hi - lo) * random.uniform(-1.0,5.0) + lo
        self.setBobot(wts)
        result = self.getBobot()
        for i in range(self.JumlahHidden-1):
            for j in range(self.JumlahOutput):
                self.hoBobot[i][j] = result[i]
                self.oBias[j] = result[i]
        return self.hoBobot, self.oBias


    def setBobot(self, bobot):
        k = 0
        for i in range(self.JumlahHidden-1):
            for j in range(self.JumlahOutput-1):
                self.hoBobot[i][j] = bobot[k]
                k += 1
        for i in range(self.JumlahOutput-1):
          self.oBias[i] = bobot[i]
          k += 1

    def HitungNilaiOutput(self,nilaiInput):
        for i in range(len(nilaiInput)):
            self.inputs[i] = nilaiInput[i]

        #nilai output pada hidden node
        for j in range(self.JumlahHidden-1):
            d = self.jarakEuclidean(self.inputs, self.centroids[j], len(self.inputs))
            r = float(-1.0 * (d*d)/(2*self.widths[j] * self.widths[j]))
            g = math.exp(r)
            self.hOutputs[j] = g

        #hitung nilai sementara dgn cara perkalian antara nilai output pd hidden node dgn matriks bobot pd hidden node
        nilaiOutputSementara = [0 for i in range(self.JumlahOutput - 1)]
        for k in range(self.JumlahOutput - 1):
            for j in range(self.JumlahHidden - 1):
                nilaiOutputSementara[k] = float(nilaiOutputSementara[k]) + (self.hOutputs[j] * self.hoBobot[j][k])

        #tambahkan nilai bias pada nilai sementara
        for k in range(self.JumlahOutput - 1):
            nilaiOutputSementara[k] = self.oBias[k]

        #hitung nilai akhir dgn menggunakan metode softmax
        #1.cari nilai terbesar dari output sementara
        maks = nilaiOutputSementara[0]
        for i in range(len(nilaiOutputSementara)-1):
            if nilaiOutputSementara[i] > maks:
                maks = nilaiOutputSementara

        #2.Tentukan faktor skala, yaitu junlah dari konstanta e dipangkatkan (setiap nilai dikurangi nilai terbesar)
        skala = 0.0
        for i in range(len(nilaiOutputSementara)-1):
            skala += math.exp(nilaiOutputSementara[i] - maks)

        """
        3. Nilai akhir adalah nilai output sementara dibagi dengan faktor skala
        Sehingga jumlah semua nilai pada nilai output adalah 1
        """
        nilaiOutputAkhir = [0 for i in range(len(nilaiOutputSementara)-1)]
        for i in range(len(nilaiOutputSementara)-1):
            nilaiOutputAkhir[i] = math.exp(nilaiOutputSementara[i] - maks)/skala

        for i in range(len(nilaiOutputAkhir)):
            self.outputs[i] = nilaiOutputAkhir[i]

        nilaiAkhir = [0 for i in range(self.JumlahOutput - 1)]
        for i in range(len(self.outputs)):
            nilaiAkhir[i] = nilaiOutputAkhir[i]
            return nilaiAkhir

    def MeanSquaredError(self,data,bobot):
        self.setBobot(bobot)
        nilaiInput = [0 for i in range(self.JumlahInput)]
        nilaiOutput = [0 for i in range(self.JumlahOutput)]
        hasil = 0.0
        for i in range(len(data)-1):

            for j in range(self.JumlahInput):
                nilaiInput[j] = data[i][j]

            for j in range(self.JumlahOutput):
                nilaiOutput[j] = data[i][j+self.JumlahInput]

            dataKolomHasil = self.HitungNilaiOutput(nilaiInput)
            for j in range(len(dataKolomHasil) - 1):
                hasil +=((dataKolomHasil[j] - nilaiOutput[j]) * (dataKolomHasil[j] - nilaiOutput[j]))
        return hasil/len(data)

    def jarakEuclidean(self, dataPertama, dataKedua, JumlahData):
        jumlah = 0.0
        for i in range(JumlahData - 1):
            delta = (dataPertama[i] - dataKedua[i]) * (dataPertama[i] - dataKedua[i])
            jumlah += delta
        return math.sqrt(jumlah)

class Partikel():
    def __init__(self, posisi, nilaiKesalahan, kecepatan, posisiTerbaik, nilaiKesalahanTerendah):
        self.posisi = [0 for i in range(len(posisi)-1)]
        posisi = self.posisi
        self.nilaiKesalahan = nilaiKesalahan
        self.kecepatan = [0 for i in range(len(kecepatan)-1)]
        kecepatan = self.kecepatan
        self.posisiTerbaik = [0 for i in range(len(posisiTerbaik)-1)]
        posisiTerbaik = self.posisiTerbaik
        self.nilaiKesalahanTerendah = nilaiKesalahanTerendah



direktori_trainData = str(raw_input("Masukkan nama data latih dengan direktorinya: "))
#direktori_testData = str(raw_input("Masukkan nama data test dengan direktorinya: "))
num_input_layer = int(input("masukan jumlah layer input (input nodes): " ))
num_hidden_layer = int(input("masukan jumlah layer tersembunyi (hidden nodes): " ))
num_output_layer = int(input("masukan jumlah layer output (output nodes): " ))

#jlh kluster
k = num_hidden_layer
#ambil data awal
trainData = baca_file(direktori_trainData)
#testData = baca_file(direktori_testData)


print "\nCreating a ",num_input_layer,"-input, ",num_hidden_layer,"-hidden, ",num_output_layer,"-output neural network\n"
nn = RadialNet(num_input_layer, num_hidden_layer, num_output_layer)
maxEpochs = 100
print "Setting maxEpochs = " + str(maxEpochs)

print "Hitung Nilai Output\n"
nn.HitungNilaiOutput(trainData)
#print "\nBeginning training using random guesses"
#nn.RBF(trainData, maxEpochs)















