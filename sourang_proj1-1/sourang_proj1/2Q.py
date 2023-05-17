import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def covariance(A,B):
    A_mean=np.mean(A)
    B_mean=np.mean(B)

    return np.sum((A-A_mean)*(B-B_mean))/(len(A))
    
def variance(C):
    C_mean=C.mean()
    return np.sum((C-C_mean)**2)/(len(C))

load=pd.read_csv("pc1.csv", header = None)

X_cord=load[0]
Y_cord=load[1]
Z_cord=load[2]

Matrix=[[variance(X_cord), covariance(X_cord,Y_cord), covariance(X_cord,Z_cord)],
        [covariance(X_cord,Y_cord), variance(Y_cord), covariance(Y_cord,Z_cord)],
        [covariance(X_cord,Z_cord), covariance(Y_cord,Z_cord), variance(Z_cord)]]

eig_val, eig_vec =np.linalg.eig(Matrix)

if min(eig_val)==eig_val[0]:
    eigen_vec_f = eig_vec[0]
    print(eigen_vec_f)







    

