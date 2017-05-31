#From Numerical Recipes in C

SAFETY=0.9
PGROW = -0.2
PSHRINK = -0.25
ERRCON = 1.89e-4



a2=0.2;a3=0.3;a4=0.6;a5=1.0;a6=0.875;b21=0.2;
b31=3.0/40.0;b32=9.0/40.0;b41=0.3;b42 = -0.9;b43=1.2;
b51 = -11.0/54.0; b52=2.5;b53 = -70.0/27.0;b54=35.0/27.0;
b61=1631.0/55296.0;b62=175.0/512.0;b63=575.0/13824.0;
b64=44275.0/110592.0;b65=253.0/4096.0;c1=37.0/378.0;
c3=250.0/621.0;c4=125.0/594.0;c6=512.0/1771.0;
dc5 = -277.00/14336.0;



dc1=c1-2825.0/27648.0;dc3=c3-18575.0/48384.0;
dc4=c4-13525.0/55296.0;dc6=c6-0.25;



#Adaptive stepsize monitoring
def rkqs(y, dydx, n, x, htry, eps, yscal, derivs):
    yerr = [0 for x in range(n)]
    ytemp = [0 for x in range(n)]
    h=htry
    while True:
        ytemp,yerr=rkck(y,dydx,n,x,h,derivs)
        errarr= [yerr[i]/yscal[i] for i in range(n)]
        errmax = max(errarr)
        errmax /= eps
        if errmax <= 1.0: #Step was succesful, time for next step!
            break

        htemp=SAFETY*h*pow(errmax,PSHRINK)#reducing step size, try again
        if h >= 0.0:
            h=max(htemp,0.1*h)
        else:
            h=min(htemp,0.1*h)

        xnew = x*+h
        if xnew == x:
            print("Stepsize underflow in rkqs")

    if errmax > ERRCON:
        hnext = SAFETY*h*pow(errmax,PGROW)
    else:
        hnext = 5.0*h
    hdid =h
    x+=hdid

    return x,ytemp,hdid,hnext

def rkqs1d(y, dydx, x, htry, eps, derivs):
    h=htry
    while True:
        ytemp,yerr=rkck1d(y,dydx,x,h,derivs)
        errmax= abs(yerr/ytemp)
        errmax /= eps
        if errmax <= 1.0: #Step was succesful, time for next step!
            break

        htemp=SAFETY*h*pow(errmax,PSHRINK)#reducing step size, try again
        if h >= 0.0:
            h=max(htemp,0.1*h)
        else:
            h=min(htemp,0.1*h)

        xnew = x*+h
        if xnew == x:
            print("Stepsize underflow in rkqs")

    if errmax > ERRCON:
        hnext = SAFETY*h*pow(errmax,PGROW)
    else:
        hnext = 5.0*h
    hdid =h
    x+=hdid

    return x,ytemp,hdid,hnext



#Cash-Karp Runge-Kutta step
def rkck(y, dydx, n, x, h, derivs):
    ak2=[0 for x in range(n)]
    ak3=[0 for x in range(n)]
    ak4=[0 for x in range(n)]
    ak5=[0 for x in range(n)]
    ak6=[0 for x in range(n)]
    ytemp=[0 for x in range(n)]

    for i in range(n):
        ytemp[i]=y[i]+b21*h*dydx[i]#First step
    ak2=derivs(x+a2*h,ytemp)#Second Step

    for i in range(n):
        ytemp[i]=y[i]+h*(b31*dydx[i]+b32*ak2[i])
    ak3=derivs(x+a3*h,ytemp);#Third Step

    for i in range(n):
        ytemp[i]=y[i]+h*(b41*dydx[i]+b42*ak2[i]+b43*ak3[i])
    ak4=derivs(x+a4*h,ytemp);#Fourth step

    for i in range(n):
        ytemp[i]=y[i]+h*(b51*dydx[i]+b52*ak2[i]+b53*ak3[i]+b54*ak4[i]);
    ak5=derivs(x+a5*h,ytemp);#Fifth step

    for i in range(n):
        ytemp[i]=y[i]+h*(b61*dydx[i]+b62*ak2[i]+b63*ak3[i]+b64*ak4[i]+b65*ak5[i]);
    ak6=derivs(x+a6*h,ytemp);#Sixth step.


    for i in range(n):
        yout[i]=y[i]+h*(c1*dydx[i]+c3*ak3[i]+c4*ak4[i]+c6*ak6[i]);

    for i in range(n):
        yerr[i]=h*(dc1*dydx[i]+dc3*ak3[i]+dc4*ak4[i]+dc5*ak5[i]+dc6*ak6[i]);

    return yout,yerr
#Cash-Karp Runge-Kutta step
def rkck1d(y, dydx, x, h, derivs):
    ytemp=y+b21*h*dydx#First step

    ak2=derivs(x+a2*h,ytemp)#Second Step

    ytemp=y+h*(b31*dydx+b32*ak2)

    ak3=derivs(x+a3*h,ytemp);#Third Step

    ytemp=y+h*(b41*dydx+b42*ak2+b43*ak3)

    ak4=derivs(x+a4*h,ytemp);#Fourth step

    ytemp=y+h*(b51*dydx+b52*ak2+b53*ak3+b54*ak4);

    ak5=derivs(x+a5*h,ytemp);#Fifth step

    ytemp=y+h*(b61*dydx+b62*ak2+b63*ak3+b64*ak4+b65*ak5);

    ak6=derivs(x+a6*h,ytemp);#Sixth step.


    yout=y+h*(c1*dydx+c3*ak3+c4*ak4+c6*ak6);

    yerr=h*(dc1*dydx+dc3*ak3+dc4*ak4+dc5*ak5+dc6*ak6);

    return yout,yerr





