# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 18:28:58 2021

@author: AlDamen
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import trapz
from scipy.optimize import curve_fit
from matplotlib import gridspec
from scipy.stats import kurtosis
from scipy.stats import skew
from scipy import optimize
from scipy.signal import savgol_filter
from scipy.special import wofz
import pylab
from lmfit import Model
from lmfit.models import ExponentialModel, GaussianModel

# Smooth and reduce noise function     
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def G(x, alpha):
    """ Return Gaussian line shape at x with HWHM alpha """
    return np.sqrt(np.log(2) / np.pi) / alpha\
                             * np.exp(-(x / alpha)**2 * np.log(2))

def L(x, gamma):
    """ Return Lorentzian line shape at x with HWHM gamma """
    return gamma / np.pi / (x**2 + gamma**2)

def V(x, alpha, gamma):
    """
    Return the Voigt line shape at x with Lorentzian component HWHM gamma
    and Gaussian component HWHM alpha.

    """
    sigma = alpha / np.sqrt(2 * np.log(2))
    return np.real(wofz((x + 1j*gamma)/sigma/np.sqrt(2))) / sigma\
         /np.sqrt(2*np.pi)
        

f1 = np.loadtxt('Fe6Gd.txt')
# f1 = np.genfromtxt('t.csv', delimiter=',')
Ln = 'Gd'
# the function
x = f1[:, 0]
y = f1[:, 1]
# smooth
# s = [3,1,1]


y = smooth(y,7)
########################################
fig, ax1 = plt.subplots() 
ax2 = ax1.twinx()

ax1.plot(x, y,'k-', fillstyle='none')
ax1.set_xlim([400,4000])
ax1.set_ylim([0,100]) 
ax2.set_ylim([-1,1])  
ax1.invert_xaxis()
quit()
# the 1st derivative
dy = np.diff(y, 1)
dx = np.diff(x, 1)
yfirst = dy/dx
xfirst = 0.5*(x[:-1]+x[1:])
# yfirst = savgol_filter(yfirst, 5, 2)
# ax2.plot(xfirst,yfirst,'k--')

# the 2nd derivative+ smooth
dyy = np.diff(yfirst, 1)
dxx = np.diff(xfirst, 1)
ysecond = dyy/dxx
xsecond = 0.5*(xfirst[:-1]+xfirst[1:])
# #wsecond = savgol_filter(ysecond, 21, 2)
# ax2.plot(xsecond,ysecond,'r-')
#XXXXXXXXXXXXXXXXXXXXX first derivative  XXXXXXXXXXXXXXXXXXXXXXXXXX

min_dx1, min_dy1, min_y1  = [], [], []
for i in range(len(dy)-1):
    if (yfirst[i] < 0 and yfirst[i+1] > 0):
        min_dx1.append((xfirst[i]+xfirst[i+1])/2)

min_dx2, min_dy2, min_y2  = [], [], []
for i1 in range(len(x)):
    for i2 in range(len(min_dx1)):
        if (np.abs(x[i1]-min_dx1[i2]) < 0.1):
            min_dx2.append(x[i1])
            min_dy2.append(y[i1])                

max_dx1, max_dy1, max_y1  = [], [], []
for i in range(len(dy)-1):
    if (yfirst[i] > 0 and yfirst[i+1] < 0):
        max_dx1.append((xfirst[i]+xfirst[i+1])/2)

max_dx2, max_dy2, max_y2  = [], [], []
for i1 in range(len(x)):
    for i2 in range(len(max_dx1)):
        if (np.abs(x[i1]-max_dx1[i2]) < 0.1):
            max_dx2.append(x[i1])
            max_dy2.append(y[i1])   

# ax1.plot(max_dx2,max_dy2,'mx')
##############################################
max_dxx1, max_dyy1, max_y1  = [], [], []
for i in range(1,len(dyy)-1):
    if (ysecond[i-1] < ysecond[i]) and (ysecond[i] > ysecond[i+1]) :
        # if (ysecond[i]>0.03):    # noise
            # if (xsecond[i]-xsecond[i-1])>10: # minimum detection dx
                max_dxx1.append(xsecond[i])
                max_dyy1.append(ysecond[i])
# ax2.plot(max_dxx1,max_dyy1,'kx')   

max_dxx2, max_dyy2, max_y2  = [], [], []
for i1 in range(len(x)):
    for i2 in range(len(max_dxx1)):
        if (np.abs(x[i1]-max_dxx1[i2]) < 0.1):
            max_dxx2.append(x[i1])
            max_dyy2.append(y[i1])  

min_dxx1, min_dyy1, min_yy1  = [], [], []
for i in range(len(dyy)-1):
    if (ysecond[i-1] > ysecond[i]) and (ysecond[i] < ysecond[i+1]) :
        # if (ysecond[i]<-0.03):    # noise
            # if (xsecond[i]-xsecond[i-1])>10: # minimum detection dx
                min_dxx1.append(xsecond[i])
                min_dyy1.append(ysecond[i])
# ax2.plot(min_dxx1,min_dyy1,'k+')


min_dxx2, min_dyy2, min_yy2  = [], [], []
for i1 in range(len(x)):
    for i2 in range(len(min_dxx1)):
        if (np.abs(x[i1]-min_dxx1[i2]) < 0.1):
            min_dxx2.append(x[i1])
            min_dyy2.append(y[i1])                


nmin_dxx1, nmin_dyy1, nmax_dxx1, nmax_dyy1 = [],[],[],[]
for imax in range (len(max_dyy1)):
    for imin in range(1,len(min_dyy1)):
        if (max_dxx1[imax]-min_dxx1[imin] < 0) and (max_dxx1[imax]-min_dxx1[imin-1] > 0):
            if (min_dyy1[imin]<-0.00):
                nmin_dxx1.append(min_dxx1[imin])
                nmin_dyy1.append(min_dyy1[imin])
        if (max_dxx1[imax]-min_dxx1[imin] < 0) and (max_dxx1[imax]-min_dxx1[imin-1] < 0):
            if (max_dyy1[imax]>+0.00):
                nmax_dxx1.append(max_dxx1[imax])
                nmax_dyy1.append(max_dyy1[imax])

#test area 
nmin_dxx1, nmin_dyy1, nmax_dxx1, nmax_dyy1 = [],[],[],[]
for imax in range (1,len(max_dyy1)-1):
    for imin in range(1,len(min_dyy1)-1):
        if (max_dxx1[imax]-min_dxx1[imin-1] >  0) and (max_dxx1[imax]-min_dxx1[imin+1] < 0):
            if (min_dyy1[imin]<-0.0):
                nmin_dxx1.append(min_dxx1[imin])
                nmin_dyy1.append(min_dyy1[imin])
        if (max_dxx1[imax]-min_dxx1[imin] < 0) and (max_dxx1[imax+1]-min_dxx1[imin] > 0):
            if (max_dyy1[imax]>+0.0):
                nmax_dxx1.append(max_dxx1[imax])
                nmax_dyy1.append(max_dyy1[imax])

# ax2.plot(nmax_dxx1[5],nmax_dyy1[5],'ko')          
 



# ax2.plot(nmin_dxx1,nmin_dyy1,'go')          
# ax1.plot(min_dx2,min_dy2,'r+')   

nmin_p1_dxx1, nmin_p1_dyy1, nmin_p2_dxx1, nmin_p2_dyy1, nmin1_dxx1, nmin1_dyy1 = [], [], [], [],[], []
for i in range(1,len(nmin_dxx1)):
    for j in range(len(min_dx2)):
        if(min_dx2[j] > nmin_dxx1[i-1]) and (min_dx2[j] < nmin_dxx1[i])  : #2 is the dx
            nmin_p1_dxx1.append(nmin_dxx1[i-1])
            nmin_p2_dxx1.append(nmin_dxx1[i ])
            nmin_p1_dyy1.append(nmin_dyy1[i-1])
            nmin_p2_dyy1.append(nmin_dyy1[i ])
            nmin1_dxx1.append(min_dx2[j])
            nmin1_dyy1.append(min_dy2[j])   
            
# ax2.plot(nmin_p1_dxx1[9],nmin_p1_dyy1[9],'ko')
# ax2.plot(nmin_p2_dxx1[9],nmin_p2_dyy1[9],'go')
# ax1.plot(nmin1_dxx1[9],nmin1_dyy1[9],'bo')
          


# ax2.plot(nmin_p1_dxx1[9],nmin_p1_dyy1[9],'ko')
# ax2.plot(nmin_p2_dxx1[9],nmin_p2_dyy1[9],'go')

pdxx1, pdyy1, pdxx2, pdyy2 = [],[],[],[]
for i1 in range(len(x)):
    for i2 in range(len(nmin_p1_dxx1)):
        if (np.abs(x[i1]-nmin_p1_dxx1[i2]) < 0.1):
            pdxx1.append(x[i1])
            pdyy1.append(y[i1])   
for i1 in range(len(x)):
    for i2 in range(len(nmin_p2_dxx1)):
        if (np.abs(x[i1]-nmin_p2_dxx1[i2]) < 0.1):
            pdxx2.append(x[i1])
            pdyy2.append(y[i1])   

# for i in range(len(pdxx2)):
        
#     ax1.plot(pdxx1[i],pdyy1[i],'bo')
#     ax1.plot(pdxx2[i],pdyy2[i],'go')
#     ax1.plot(nmin1_dxx1[i],nmin1_dyy1[i],'ko')
 


B = []
Band = []
g = open("Bands1.txt", "w")
# for i in   range(len(min_y1))[::-1]:
for i in range(len(min_dy2)):
    B.clear()
    # print(rdx2[i],ry2[i])
    if ((min_dy2[i]) >= 0) and ((min_dy2[i]) <= 35):
        #print(i,round(min_dx1[i],2),round(min_y1[i],2),"strong")
        print(str(round(min_dx2[i]))+"(s)"+","+" ",end="", flush=True),
        B.append(["Band: ", round(min_dx2[i],2),round(min_dy2[i],2), "strong"])
        Band.append(i)
        Band.append(round(min_dx2[i],2))
        Band.append(round(min_dy2[i],2))
        Band.append("s")
        g.write(str(B) + '\n')
    if(min_dy2[i] < 75) and (min_dy2[i] > 35):
        #print(i,round(min_dx1[i],2),round(min_y1[i],2),"medium")
        print(str(round(min_dx2[i]))+"(m)"+","+" ",end="", flush=True),
        B.append(["Band: ", round(min_dx2[i],2),round(min_dy2[i],2), "medium"])
        Band.append(i)
        Band.append(round(min_dx2[i],2))
        Band.append(round(min_dy2[i],2))
        Band.append("m")
        g.write(str(B) + '\n')
        
    if(min_dy2[i] >= 75) and (min_dy2[i] <= 95):
        #print(i,round(min_dx1[i],2),round(min_y1[i],2),"strong")
        print(str(round(min_dx2[i]))+"(w)"+","+" ",end="", flush=True),
        B.append(["Band: ", round(min_dx2[i],2),round(min_dy2[i],2), "weak"])
        Band.append(i)
        Band.append(round(min_dx2[i],2))
        Band.append(round(min_dy2[i],2))
        Band.append("w")
        g.write(str(B) + '\n')
print("")
g.close()





# for i in range(15,30):
#     print(i, " " , min_dx1[i])
    
# for i in range(15,30):
#     print(i, max_dx1[i])



#take full curve from max to max
# b = 2
# kr  = 100
# for i in range(len(max_dx2)-1):
#     point.append(i)
#     point.append(0)
#     point.append(min_dx2[i])
#     pointx.append(min_dx2[i])
#     for k in range(1,kr):
#         if(min_dx2[i] - b*k > (max_dx2[i] + b)):
#             imin.append(i)
#             kmin.append(k)
#             point.append(i)
#             point.append(k)
#             point.append(min_dx2[i] - b*k) 
#             pointx.append(min_dx2[i] - b*k)
#     for k in range(1,kr):
#         if(min_dx2[i] + b*k < (b + max_dx2[i+1])):
#             imax.append(i)
#             kmax.append(k)
#             point.append(i)
#             point.append(k)
#             point.append(min_dx2[i] + b*k)
#             pointx.append(min_dx2[i] + b*k)

pointx = []
pointy = []
kmin,kmax = [],[]
imin,imax = [],[]
point = []
b = 2
kr  = 200
dx = 0
#take curve to mid point beteen max(i -> i+1) and min
for i in range(len(pdyy2)):
    imax.append(i)
    kmax.append(0)
    point.append(i)
    point.append(0)
    point.append(min_dx2[i])
    pointx.append(min_dx2[i])
    # half range distance 
    # S0 = (1/2)*((max_dx2[i]+min_dx2[i]))
    # S1 = (1/2)*((max_dx2[i+1]+min_dx2[i]))
        
    #full range peak max to max
    # S0 = min(((max_dx2[i],min_dx2[i])))
    # S1 = max(((max_dx2[i+1],min_dx2[i])))
     #or
    # S0 = min(max_dx2[i-1], max_dx2[i])
    # S1 = min(max_dx2[i], max_dx2[i+1])
    
    S0 = pdxx1[i]
    S1 = pdxx2[i]
     
    for k in range(kr):
        if(nmin1_dxx1[i] - b*(k)) > S0   :
            imin.append(i)
            kmin.append(k)
            point.append(i)
            point.append(k)
            point.append(nmin1_dxx1[i] - b*k) 
            pointx.append(nmin1_dxx1[i] - b*k)
    for k in range(kr):
        if(nmin1_dxx1[i] + b*(k)) < S1  :
            imax.append(i)
            kmax.append(k)
            point.append(i)
            point.append(k)
            point.append(nmin1_dxx1[i] + b*k)
            pointx.append(nmin1_dxx1[i] + b*k)

pointx1,pointy1,pointi,pointk  = [],[],[],[]        
for i in range(len(pointx)):
    for j in range(len(x)):
        if np.abs(pointx[i] - x[j]) < 1:
           pointi.append(point[3*i])
           pointk.append(point[3*i+1])
           pointx1.append(x[j])
           pointy1.append(y[j]) 

# for i in range(len(pointx1)):
#       ax1.plot(pointx1[i],pointy1[i],'o')

P = []
g = open("out.txt", "w")
for i in range(len(pointi)):
    for j in range(max(pointi)):
        P.clear()
        if(pointi[i]==j):
           P.append("  ")
           P.append(pointi[i])
           P.append(pointx1[i])
           P.append(pointy1[i])
           P.append("  ")       
           g.write(str(P) + '\n')
       
g.close()

txt = np.genfromtxt("out.txt", delimiter=',')  
out = sum(1 for line in open("out.txt")) 


oz = txt[: ,1]
ox = txt[:, 2]
oy = txt[:, 3]



L = []
s=[]
# for j in range(int(np.max(oz))):
for j in range(int(np.max(oz))):
    s1 = 0
    g = open("out{}".format(j), "w")
    for i in range(out):
        if(oz[i]==j):               
            L.clear()
            L.append("E ")
            L.append(ox[i])
            L.append(oy[i])
            L.append("S ")
            g.write(str(L) + '\n')
            s1 = s1 + 1
    s.append(s1-1)
g.close()





# get max(Y), and its X
x_curve_max, y_curve_max = [],[]
for j in range(int(np.max(oz))):
    txt1 = np.genfromtxt("out{}".format(j), delimiter=',')         
    if (s[j]>6):
        # ax1.plot(txt1[:,1], txt1[:,2],'.')
        txt1 = txt1[txt1[:,1].argsort()] 
        x_curve = txt1[:,1]
        y_curve = 100-txt1[:,2]   
        # connect the max Y with its X      
        for i in range(len(x_curve)):
            if (max(y_curve) == y_curve[i]):
                y_curve_max.append(y_curve[i])
                x_curve_max.append(x_curve[i])
                
# print("start fit")               
# for j in range(int(np.max(oz))):
#     txt1 = np.genfromtxt("out{}".format(j), delimiter=',')  
#     txt1 = txt1[txt1[:,1].argsort()] 
#     x_curve = txt1[:,1]
#     y_curve = 100-txt1[:,2]     
#     if (s[j]>6):
#         x = x_curve
#         y = y_curve
        
#         exp_mod = ExponentialModel(prefix='exp_')
#         pars = exp_mod.guess(y, x=x)
        
#         gauss1 = GaussianModel(prefix='g1_')
#         pars.update(gauss1.make_params())
        
#         pars['g1_center'].set(value=((max(x_curve)+min(x_curve))/2), min=min(x_curve), max=max(x_curve))
#         pars['g1_sigma'].set(value=10, min=5)
#         pars['g1_amplitude'].set(value=max(y_curve), min=0)
        
#         gauss2 = GaussianModel(prefix='g2_')
#         pars.update(gauss2.make_params())
        
#         pars['g2_center'].set(value=((max(x_curve)+min(x_curve))/2)+10, min=min(x_curve)-10, max=max(x_curve)+10)
#         pars['g2_sigma'].set(value=10, min=5)
#         pars['g2_amplitude'].set(value=max(y_curve), min=0)
        
#         mod = gauss1 + gauss2 + exp_mod
        
#         init = mod.eval(pars, x=x)
#         out = mod.fit(y, pars, x=x)
        
#         print(out.fit_report(min_correl=0.5))
        
#         fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
#         axes[0].plot(x, y, 'b')
#         axes[0].plot(x, init, 'k--', label='initial fit')
#         axes[0].plot(x, out.best_fit, 'r-', label='best fit')
#         axes[0].legend(loc='best')
        
#         comps = out.eval_components(x=x)
#         axes[1].plot(x, y, 'b')
#         axes[1].plot(x, comps['g1_'], 'g--', label='Gaussian component 1')
#         axes[1].plot(x, comps['g2_'], 'm--', label='Gaussian component 2')
#         axes[1].plot(x, comps['exp_'], 'k--', label='Exponential component')
#         axes[1].legend(loc='best')

# plt.show()
# quit()
                
                
for j in range(int(np.max(oz))):
    txt1 = np.genfromtxt("out{}".format(j), delimiter=',')  
    txt1 = txt1[txt1[:,1].argsort()] 
    x_curve = txt1[:,1]
    y_curve = 100-txt1[:,2]     
    if (s[j]>3):
        ax1.plot(txt1[:,1], txt1[:,2],'-o',fillstyle='none')
        # Compute the area using the composite trapezoidal rule.
        area = trapz(y_curve, dx=1)
        # print("area =", area/((4000-400)))
         
         #Gaussian function
        def gauss_function(x, a, x0, sigma):
            return a*np.exp(-(x-x0)**2/float((2*sigma**2)))



        x = x_curve
        y = y_curve #density values at each height
        
        
        amp = max(y)
        center = x[np.argmax(y)]
        width = 100 #eye-balled estimate
        #p0 = amp, width, center
        popt, pcov = curve_fit(gauss_function, x, y, p0 = [amp, width, center],maxfev = 10000)
        
        # please assign the highest y with its not use mean,then iterate with max_x min_x
        # a = (max(x_curve)+min(x_curve))/2
        # b = (max(y_curve)+min(y_curve))/2
        # c = (max(x_curve)+min(x_curve))/2 - 5 
        
        ax1.plot(x_curve, 100-gauss_function(x_curve, *popt),'r-')

        print("________________________________________")
        print("index", j) 
        y_curve = 100- y_curve
        print("width (X2-X1)=", round(np.max(x_curve)-np.min(x_curve),2), "height (Ymax)=", round(100-max(y_curve),2))
        print("area=",   round(area,2))
        print("Check:", round(max(y_curve)*(np.max(x_curve)-np.min(x_curve))/2,2))
        print("mean : ", round(np.mean(x_curve),2))
        print("skew : ", round(skew(100-y_curve),2))
        print("kurt : ", round(kurtosis(100-y_curve),2))



############# FIGURE SPECIFICATIONs #################
# ax1.set_xlim([400,4000])
# ax1.set_ylim([0,100]) 
# ax1.set_xlim([600,800])
# ax2.set_ylim([-5,5])
# ax1.invert_xaxis()
ax1.legend(loc ='lower left', frameon=False, prop={'size':8 })
ax1.set_xlabel('wavenummber ($cm^{-1}$)', color='k')
ax1.set_ylabel('Transmittance (%)')

plt.savefig("IR.png", format="png", dpi=200)

