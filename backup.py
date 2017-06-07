#First test evolution function. Oddly, performs only a little worse than Runge-Kutta.
def Euler_Step(x,xres,dydx,Y):
    return dydx(x,xres,Y)*xres,xres

def Ystep_euler(g,mx,sigmav,xstep=1e-2,Deltax=1e-4):
    Yeqset = lambda x: Yeq(mx,mx/x,g,0,0)
    neqset = lambda x: neq(mx,mx/x,g,0,0)

    #Find a point shortly before freezeout
    xstart=brentq(DeltaCond,1,100,args=(mx,Deltax,g,sigmav,Delta_Y_Condition,))
    Y = Yeqset(xstart)
    xi=xstart
    xmax=xstart+20
    dydx = lambda x, xstep, Y: -Yeqset(x+xstep)/x*neqset(x+xstep)*sigmav(x+xstep)        /Hub(mx/(x+xstep))*((Y/Yeqset(x+xstep))**2-1)
    while True:
        if Y>2.5*Yeqset(xi) or xi>xmax:
            break
        deltay,xstep = Euler_Step(xi,xstep,dydx,Y)
        #print(xi,Y,Yeqset(xi),deltay)
        Y+=deltay
        xi+=xstep

    Yinf_val,Yinf_error = quad(Yevolution_integrand,xi,1000,epsabs=1e-300,epsrel=1e-4,limit=400,args=(mx,sigmav,))
    if Yinf_val < 100*Yinf_error:
        print("Error in Ystep integration")
        print(Yinf_val,Yinf_error)
        print(xi,mx)
    Yinf = 1.0/(1.0/(2.5*Yeqset(xi))+Yinf_val)
    return Yinf,xi+xstep



def Ysearch_euler(g,alpha_D,mv,mx,tol=1e-3,xstep=1e-2,Deltax=1e-4):
    kappa = math.sqrt(relic_density_sigma/sigmav(mx/20.0,alpha_D,1.0,mv,mx)/conversion)
    #print(kappa)
    while True:
        sig = lambda x: sigmav(mx/x,alpha_D,kappa,mv,mx)
        Y,xf = Ystep_euler(g,mx,sig,xstep,Deltax)
        Omega = Omega_from_Y(Y,mx)
        if abs(OmegaCDM-Omega)<tol:
            break
        #print(kappa,Omega)
        kappa = math.sqrt(kappa**2*Omega/OmegaCDM)
    return kappa,Omega,xf,Y


