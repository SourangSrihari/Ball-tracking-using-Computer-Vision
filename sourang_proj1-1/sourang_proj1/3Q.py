import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import random

def covariance(A,B):
    A_mean=np.mean(A)
    B_mean=np.mean(B)

    return np.sum((A-A_mean)*(B-B_mean))/(len(A))

def variance(C):
    C_mean=C.mean()
    return np.sum((C-C_mean)**2)/(len(C))


def least_square(x,y,z):
    A_t=np.vstack((x,z,np.ones(len(x))))
    y=y[np.newaxis,:]
    y_t=y.T
    A_t_A=np.matmul(A_t,A_t.T)
    A_t_b=np.matmul(A_t,y_t)
    best_fit_coefficients=np.matmul(np.linalg.inv(A_t_A),A_t_b)
    return(best_fit_coefficients)

def total_least_square(x, y, z):
    x_mean=np.sum(x)/len(x)
    y_mean=np.sum(y)/len(y)
    z_mean=np.sum(z)/len(z)

    Mat=[[variance(X_cord), covariance(X_cord,Y_cord), covariance(X_cord,Z_cord)],
        [covariance(X_cord,Y_cord), variance(Y_cord), covariance(Y_cord,Z_cord)],
        [covariance(X_cord,Z_cord), covariance(Y_cord,Z_cord), variance(Z_cord)]]
    
    eig_val, eig_vec =np.linalg.eig(Mat)
    if min(eig_val)==eig_val[0]:
        d=eig_vec[0][0]*x_mean+eig_vec[1][0]*y_mean+eig_vec[2][0]*z_mean
        return d,eig_vec[:,0]

    if min(eig_val)==eig_val[1]:
        d=eig_vec[0][1]*x_mean+eig_vec[1][1]*y_mean+eig_vec[2][1]*z_mean
        return d,eig_vec[:,1]

    if min(eig_val)==eig_val[2]:
        d=eig_vec[0][2]*x_mean+eig_vec[1][2]*y_mean+eig_vec[2][2]*z_mean
        return d,eig_vec[:,2]
    
def ransac(u,v,w,x,y,z):
    rand_list=np.empty(v,dtype="int")
    counter_max=0
    final_vector=np.array([])
    final_d=0
    for j in range(u):
        counter=0
        for i in range(v):
            rand_list[i]=random.randint(0,len(x)-1)
        x_list=np.zeros(v)
        y_list=np.zeros(v)
        z_list=np.zeros(v)
        count=0
        for i in rand_list:
            x_list[count]=x[i]
            y_list[count]=y[i]
            z_list[count]=z[i]
            count=count+1
        points_mat=np.vstack((x_list,y_list,z_list))
        vec_1=points_mat[:,1]-points_mat[:,0]
        vec_2=points_mat[:,2]-points_mat[:,0]
        normal=np.cross(vec_1,vec_2)
        d=np.dot(normal,points_mat[:,0])
        for i in range(len(x)):
            point=np.array([x[i],y[i],z[i]])
            error=abs((np.dot(point,normal)-d)/math.sqrt(np.dot(normal,normal)))
            if error<=w:
                counter=counter+1
        if counter>counter_max:
            counter_max=counter
            final_vector=normal
            final_d=d
    return final_vector,d
    

def function_y(k,x,l,z,m):
    return k*x+l*z+m

def function_y_total(k,x,l,m,z,n):
    return -k/l*x-m/l*z+n/l


#FOR DATASET 1
load1=pd.read_csv("pc1.csv", header = None)

X_cord=load1[0]
Y_cord=load1[1]
Z_cord=load1[2]

coeff_1= least_square(X_cord, Y_cord, Z_cord)
X_cord_mesh, Z_cord_mesh=np.meshgrid(X_cord,Z_cord)

Y_cord_mesh=function_y(coeff_1[0],X_cord_mesh,coeff_1[1],Z_cord_mesh,coeff_1[2])

d_1,coeff_1_TLS=total_least_square(X_cord,Y_cord,Z_cord)

y_1_mesh_total=function_y_total(coeff_1_TLS[0],X_cord_mesh,coeff_1_TLS[1],coeff_1_TLS[2],Z_cord_mesh,d_1)

coeff_1_ran,d_1_ran=ransac(40,3,0.15,X_cord,Y_cord,Z_cord)
y_1_mesh_ransac=function_y_total(coeff_1_ran[0],X_cord_mesh,coeff_1_ran[1],coeff_1_ran[2],Z_cord_mesh,d_1_ran)



fig=plt.figure()
plot1=fig.add_subplot(231,projection="3d")
plot1.plot_surface(X_cord_mesh,Y_cord_mesh,Z_cord_mesh, color="green",label='LEAST SQUARE ESTIMATION')
plot1.set_title("LST SQR(pc1)")

plot1_total=fig.add_subplot(232,projection="3d")
plot1.scatter3D(X_cord,Y_cord,Z_cord,label='raw data')
plot1_total.plot_surface(X_cord_mesh,y_1_mesh_total,Z_cord_mesh,color="red",label='TOTAL LEAST SQUARE ESTIMATION')
plot1_total.scatter3D(X_cord,Y_cord,Z_cord,label='raw data')
plot1_total.set_title("TOTAL LST SQR(pc1)")

plot1_ransac=fig.add_subplot(233,projection="3d")
plot1_ransac.plot_surface(X_cord_mesh,y_1_mesh_ransac,Z_cord_mesh,color="blue",label='RANSAC ESTIMATION')
plot1_ransac.scatter3D(X_cord,Y_cord,Z_cord,label='raw data')
plot1_ransac.set_title("RANSAC(pc1)")



#FOR DATASET 2
load2=pd.read_csv("pc2.csv", header = None)

X_cord2=load2[0]
Y_cord2=load2[1]
Z_cord2=load2[2]

coeff_2= least_square(X_cord2, Y_cord2, Z_cord2)
X_cord_mesh2, Z_cord_mesh2=np.meshgrid(X_cord2,Z_cord2)

Y_cord_mesh2=function_y(coeff_2[0],X_cord_mesh2,coeff_2[1],Z_cord_mesh2,coeff_2[2])

d_2,coeff_2_TLS=total_least_square(X_cord2,Y_cord2,Z_cord2)

y_2_mesh_total=function_y_total(coeff_2_TLS[0],X_cord_mesh2,coeff_2_TLS[1],coeff_2_TLS[2],Z_cord_mesh2,d_2)

coeff_2_ran,d_2_ran=ransac(40,3,0.15,X_cord2,Y_cord2,Z_cord2)
y_2_mesh_ran=function_y_total(coeff_2_ran[0],X_cord_mesh2,coeff_2_ran[1],coeff_2_ran[2],Z_cord_mesh2,d_2_ran)

fig=plt.figure()
plot2=fig.add_subplot(231,projection="3d")
plot2.plot_surface(X_cord_mesh2,Y_cord_mesh2,Z_cord_mesh2, color="green",label='LEAST SQUARE ESTIMATION')
plot2.set_title("LST SQR(pc2)")
plot2_total=fig.add_subplot(232,projection="3d")
plot2.scatter3D(X_cord2,Y_cord2,Z_cord2,label='raw data')
plot2_total.plot_surface(X_cord_mesh2,y_2_mesh_total,Z_cord_mesh2,color="red",label='TOTAL LEAST SQUARE ESTIMATION')
plot2_total.scatter3D(X_cord2,Y_cord2,Z_cord2,label='raw data')
plot2_total.set_title("TOTAL LST SQR(pc2)")
plot2_ransac=fig.add_subplot(233,projection="3d")
plot2_ransac.plot_surface(X_cord_mesh2,y_2_mesh_ran,Z_cord_mesh2,color="blue",label='RANSAC ESTIMATION')
plot2_ransac.scatter3D(X_cord2,Y_cord2,Z_cord2,label='raw data')
plot2_ransac.set_title("RANSAC(pc2)")
plt.show()
