import cv2
import numpy as np
import matplotlib.pyplot as plt

def LeastSquare(X, X2, Y):
    oness=np.ones(len(X))
    matrix_X = np.transpose(np.matrix([X2,X,oness]))
    matrix_Y = np.transpose(np.matrix([Y]))
    XtransX = np.transpose(matrix_X) @ (matrix_X)
    XtransY = np.transpose(matrix_X) @ (matrix_Y)
    coeff_B = np.linalg.inv(XtransX) @ XtransY
    coeff_B1= float(coeff_B[0])
    coeff_B2= float(coeff_B[1])
    coeff_B3= float(coeff_B[2])
    for i in X:
        new_Y.append(coeff_B1*(i*i)+coeff_B2*i+coeff_B3)
    return coeff_B1,coeff_B2,coeff_B3, new_Y

cap = cv2.VideoCapture('assets/ball.mov')

X=[]
X2=[]
Y=[]
new_Y=[]

while(1):
    img, frame = cap.read()
    if not img:
        break
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 175, 120])
    upper_red = np.array([255, 255, 255])


    mask = cv2.inRange(hsv, lower_red, upper_red)
    result = cv2.bitwise_and(frame, frame, mask = mask)


    x_axis, y_axis= np.where(mask!=0)
    x = list(y_axis)
    y = list(x_axis)
    if len(x) != 0 and len(y) != 0:
        mean_x= sum(x)/len(x)
        mean_y= sum(y)/len(y)
        X.append(mean_x)
        X2.append((mean_x)**2)
        Y.append(mean_y)
        
    cv2.imshow('frame', mask)
    cv2.waitKey(2)
    


a,b,c, Y_new=LeastSquare(X, X2, Y)

X_found = np.roots([a,b,c-(Y_new[0]+300)])
for x in X_found:
    if x>0:
        x_final=x
        print(x_final)
                
plt.scatter(X, Y, s = 1)
plt.plot(X,Y_new)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('BALL TRAJECTORY')
plt.show()
cv2.destroyAllWindows()
cap.release()