#Checks that a string is actually a number.
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False 

#Generate a string from a mass. Only set up for MeV and GeV right now.
def masstext(m):
    mstr=""
    if m>=gev:
        if m==1.0 or m==2.0:
            mstr=mstr+str(int(m))
        else:
            mstr=mstr+str(m)
        mstr=mstr+"gev"
    else:
        if m*1000%1 == 0.0:
            mstr=mstr+str(int(m*1000))+"mev"
        else:
            mstr=mstr+str(m*1000)+"mev"
    return mstr