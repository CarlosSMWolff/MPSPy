from mpmath import hyp3f2,fac,rf

def QHahn(k,n,N):
    return float(hyp3f2(-k,k+2,-n,2,-N,1))

def rhon2(n,N):
    return float(((-1)**n * rf(n+2,N+1)*rf(1,n)*fac(n))/((2*n+2)*fac(1+n)*rf(-N,n)*fac(N)))

def An(n,N):
    return ((n+2)**2*(N-n))/((2*n+2)*(2*n+3))
    
def Cn(n,N):
    return (n*(n+N+2)*n)/((2*n+1)*(2*n+2))

