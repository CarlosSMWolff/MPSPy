import numpy as np
from numpy import transpose, dot, reshape, sqrt,conjugate
import time
from mpmath import hyp3f2,fac,rf
from scipy.linalg import expm
from QHahn import *
import scipy
from scipy.sparse import csr_matrix, csc_matrix

def kron(A,B):
    return csc_matrix(scipy.sparse.kron(A,B))

def FormTheta(lambdaLeft,lambdaCent,lambdaRight,gammaLeft,gammaRight,chi,d1,d2):
    theta = np.zeros([chi,d1,d2,chi],dtype=complex)

    for alpha1 in range(chi):
        for alpha2 in range(chi):
            for gamma in range(chi):
                theta[alpha1,:,:,alpha2] += lambdaLeft[alpha1]*lambdaCent[gamma]*lambdaRight[alpha2]*\
                dot(transpose(gammaLeft[alpha1,:,gamma][np.newaxis]),gammaRight[gamma,:,alpha2][np.newaxis])
    return theta
    
    
def ThetaNewOld(operator,theta):
    dims = theta.shape
    chi = dims[0]
    d1 = dims[1]
    d2 = dims[2]
    thetaNew = np.zeros([chi,d1,d2,chi],dtype=complex)
    for alpha1 in range(chi):
        for i in range(d1):
            for j in range(d2):
                for alpha2 in range(chi):
                    for ii in range(d1):
                        for jj in range(d2):
                            thetaNew[alpha1,i,j,alpha2]+= operator[i,j,ii,jj]*theta[alpha1,ii,jj,alpha2]
    return thetaNew
    
    
def ThetaNew(operatorRS,theta):
    dims = theta.shape
    chi = dims[0]
    d1 = dims[1]
    d2 = dims[2]
    thetaNew = np.zeros([chi,d1,d2,chi],dtype=complex)
    for alpha1 in range(chi):
        for alpha2 in range(chi):
            thetaflat = (theta[alpha1,:,:,alpha2]).flatten()
            thetaNew[alpha1,:,:,alpha2] = (operatorRS.dot(thetaflat)).reshape(d1,d2)
    return thetaNew
    
    
def ThetaReshape(theta):
    dims = theta.shape
    chi = dims[0]
    d1 = dims[1]
    d2 = dims[2]
    theta=transpose(theta,(1,0,2,3)).reshape(d1*chi,d2*chi,order='C')
    return theta	
    
def ApplyLocal2SiteOp(operator,lambdaLeft,lambdaCent,lambdaRight,gammaLeft,gammaRight):
    
    # Obtain dimensions
    dim1 = gammaLeft.shape
    chi = dim1[0]
    d1 = dim1[1]
    dim2 = gammaRight.shape
    d2 = dim2[1]
    
    # Create theta
    theta=FormTheta(lambdaLeft,lambdaCent,lambdaRight,gammaLeft,gammaRight,chi,d1,d2)
    # Apply operator
    thetaNew = ThetaNew(operator,theta)
    thetaRS = ThetaReshape(thetaNew)
    
    #SVD decomposition    
    U, s, V = np.linalg.svd(thetaRS)
    
    # Cast into new lambda and thetas
    lambdaNew=s[:chi]
    lambdaNew = lambdaNew/sqrt(dot(lambdaNew,lambdaNew))

    gammaLeftNew=np.zeros([chi,d1,chi],dtype=complex)
    for alpha1 in range(chi):
        for i in range(d1):
            for alpha2 in range(chi):
                if lambdaLeft[alpha1]==0:
                    gammaLeftNew[alpha1,i,alpha2]=0
                else:
                     gammaLeftNew[alpha1,i,alpha2]= U[i*chi+alpha1,alpha2]/lambdaLeft[alpha1]

    gammaRightNew=np.zeros([chi,d2,chi],dtype=complex)
    for alpha1 in range(chi):
        for i in range(d2):
            for alpha2 in range(chi):
                if lambdaRight[alpha2]==0:
                    gammaRightNew[alpha1,i,alpha2]=0
                else:
                     gammaRightNew[alpha1,i,alpha2]= V[alpha1,i*chi+alpha2]/lambdaRight[alpha2]
    
    schmidtError = 1-sum((s*conjugate(s))[:chi])
    
    return [gammaLeftNew,lambdaNew,gammaRightNew,schmidtError]
    
def EvolutionSite(evOp,site,gammaVector,lambdaVector):
    lambdaLeft = lambdaVector[site]
    lambdaCent = lambdaVector[site+1]
    lambdaRight = lambdaVector[site+2]
    gammaLeft = gammaVector[site]
    gammaRight = gammaVector[site+1]
    return ApplyLocal2SiteOp(evOp,lambdaLeft,lambdaCent,lambdaRight,gammaLeft,gammaRight)
    
    
 
def Annihilation(size):
    a = np.zeros((size,size))
    for i in range(size-1):
        a[i,i+1] = np.sqrt(i+1)
    return csc_matrix(a)

def CreateAnnihilationList(ntrun):
    nmodes = len(ntrun)
    avec = []
    for i in range(nmodes):
        avec.append(Annihilation(ntrun[i]+1))
    return avec
    

def GammaLambdaIni(siteFock,dvector,chi):
    nsites = len(dvector)
    coefficient = []
    for site in range(nsites):
        vec = np.zeros(dvector[site])
        if (site==siteFock):
            vec[1]=1
        else:
            vec[0]=1
        coefficient.append(vec)
        
    gammaVectorIni = []
    for site in range(nsites):
        gamma = np.zeros([chi,dvector[site],chi])
        gamma[0,:,0]=coefficient[site]
        gammaVectorIni.append(gamma)
        
    lambdaVectorIni = []
    for site in range(nsites+1):
        lambdaVectorIni.append(np.array([1,0]))
        
    return [gammaVectorIni,lambdaVectorIni]
    

def opGamma(op,Gamma):
    #This computes one site operator acting on a site Gamma, B Gamma
    dims = Gamma.shape
    chi = dims[0]
    GammaNew = np.zeros(Gamma.shape,dtype = complex)
    for alpha1 in range(chi):
        for alpha2 in range(chi):
            GammaNew[alpha1,:,alpha2] = op.dot(Gamma[alpha1,:,alpha2])
    return GammaNew   

def GQ(lambdaSite,Q,gammaSite):
    dims = gammaSite.shape
    chi = dims[0]
    gq = np.zeros((chi,chi),dtype = complex)
    for alpha in range(chi):
        for beta in range(chi) :
            gq[alpha,beta] = np.sum(lambdaSite**2*conjugate(gammaSite[alpha,:,:])*Q[beta,:,:])
    return gq

def RhoSite(site,lambdaVector,gammaVector):
    dim = gammaVector[site].shape
    chi = dim[0]
    d = dim[1]

    rho = np.zeros((d,d),dtype=complex)
    for alpha1 in range(chi):
        for alpha2 in range(chi):
            gammaMatrix=gammaVector[site][alpha1,:,alpha2][np.newaxis]
            rho = rho + (lambdaVector[site][alpha1])**2*(lambdaVector[site+1][alpha2])**2*dot(transpose(gammaMatrix),conjugate(gammaMatrix))

    return rho

def AB(site,lambdaVector,gammaVector,P,Gk):
    dim = gammaVector[site].shape
    chi = dim[0]
    d = dim[1]
    ab = 0
    for alpha in range(chi):
        for beta in range(chi):
            for beta2 in range(chi):
                ab += lambdaVector[site][beta2]**2*dot(P[beta2,:,beta],gammaVector[site][beta2,:,alpha])\
                *lambdaVector[site+1][alpha]*lambdaVector[site+1][beta]*Gk[alpha,beta]
    return ab

def GK(site,lambdaVector,gammaVector,Gk):
    dim = gammaVector[site].shape
    chi = dim[0]
    d = dim[1]
    gk = np.zeros((chi,chi),dtype=complex)
    
    for alpha in range(chi):
        for beta in range(chi):
            for alpha2 in range(chi):
                for beta2 in range(chi):
                    gk[alpha,beta]+= lambdaVector[site+1][alpha2]*lambdaVector[site+1][beta2]\
                    *dot(gammaVector[site][alpha,:,alpha2],conjugate(gammaVector[site][beta,:,beta2]))\
                    *Gk[alpha2,beta2]
    return gk

def CorrelationGridMPS(Op1,Op2,gammaVectorTime,lambdaVectorTime):

    print("Caution: CorrelationGridMPS is deprectated. Needs revision for Sparse inputs.")
    
    nstore = len(gammaVectorTime)
    nsites = len(gammaVectorTime[0])
    
    correlationGridUpper = np.zeros((nsites,nsites,nstore),dtype=complex)

    for site2 in range(1,nsites-1):
        B = Op2[site2]
        Qtime= [opGamma(B,gammaVectorTime[t][site2]) for t in range(nstore)]
        Gtime = [GQ(lambdaVectorTime[t][site2+1],Qtime[t],gammaVectorTime[t][site2])   for t in range(nstore)]

        # We compute the diagonal element <Op1(site2)Op2(site2)>
        Op = Op1[site2].dot(Op2[site2])
        diagT= np.asarray([np.trace(Op.dot(RhoSite(site2,lambdaVectorTime[t],gammaVectorTime[t]))) for t in range(nstore)])
        correlationGridUpper[site2,site2,:] = 0.5*diagT # Divided by 1/2 because we are summing it in the end

        # We compute the rest of correlations for site1 < site2
        Gktime = Gtime
        for site1 in reversed(range(0,site2-1)):
            A = Op1[site1]
            Ptime = [opGamma(A,gammaVectorTime[t][site1]) for t in range(nstore)]
            ABtime = [AB(site1,lambdaVectorTime[t],gammaVectorTime[t],Ptime[t],Gktime[t])  for t in range(nstore)]
            correlationGridUpper[site1,site2,:] = ABtime

            # We compute Gktime for the next
            if site1 >0:
                Gktime = [GK(site1,lambdaVectorTime[t],gammaVectorTime[t],Gktime[t]) for t in range(nstore)]

    # First diagonal element
    site2 = 0
    Op = dot(Op1[site2],Op2[site2])
    diagT= np.asarray([np.trace(dot(Op,RhoSite(site2,lambdaVectorTime[t],gammaVectorTime[t]))) for t in range(nstore)])
    correlationGridUpper[site2,site2,:] = 0.5*diagT # Divided by 1/2 because we are summing it in the end


    correlationGrid = transpose(correlationGridUpper+ conjugate(transpose(correlationGridUpper,(1,0,2))),(2,0,1))
    meanxGrid = np.asarray([np.diag(correlationGrid[t]) for t in range(nstore)])
    
    return [correlationGrid,meanxGrid]
    



    
def MPSTimeEvolutionCorrelation(lambdaVectorIni,gammaVectorIni,Hintsite,Op1,Op2,tini,tfin,nt,nstore,dvector,chi):
    
    tgrid = np.linspace(tini,tfin,nt)
    dt = tgrid[1]-tgrid[0]
    nstoreindex = np.linspace(0,nt-1,nstore).astype(int)
    nsites = len(dvector)

    thetaCons = 1/(2-2**(1/3))

    expH1List = []
    for site in range(nsites-1):
        matrixExp = expm(-1j*dt/2*thetaCons*Hintsite[site])
        expH1List.append(matrixExp)

    expH2List = []
    for site in range(nsites-1):
        matrixExp = expm(-1j*dt*thetaCons*Hintsite[site])
        expH2List.append(matrixExp)

    expH3List = []
    for site in range(nsites-1):
        matrixExp = expm(-1j*dt/2*(1-thetaCons)*Hintsite[site])
        expH3List.append(matrixExp)

    expH4List = []
    for site in range(nsites-1):
        matrixExp = expm(-1j*dt*(1-2*thetaCons)*Hintsite[site])
        expH4List.append(matrixExp)


    # Initialize the vectors
    gammaVector = list(gammaVectorIni[:]) 
    lambdaVector = list(lambdaVectorIni[:])
    gammaVectorTime = []
    gammaVectorTime.append(list(gammaVector))
    lambdaVectorTime = []
    lambdaVectorTime.append(list(lambdaVector))
    errorTime = []
    errorTime.append(0)
    correlationGridTime = np.zeros((nstore,nsites-1,nsites-1),dtype = complex)
    meanxGridTime = np.zeros((nstore,nsites),dtype = complex)
    [correlationGrid,meanxGrid] = CorrelationGridMPSFixedTime(Op1,Op2,gammaVector,lambdaVector)
    correlationGridTime[0] = correlationGrid 
    meanxGridTime[0] = meanxGrid
    
    # Time evolution
    currentStoreIndex=1;
    for q in range(1,nt): # The first element was already settled

        error = 0

        # Series of propagations: Forrest-T formula

        # Odd
        for i in range(int(nsites/2)):
            site = 2*i
            [gammaVector[site],lambdaVector[site+1],gammaVector[site+1],schmidtError]\
            = EvolutionSite(expH1List[site],site,gammaVector,lambdaVector)
            error += schmidtError
        # Even
        for i in range(int((nsites-1)/2)):
            site = 2*i+1
            [gammaVector[site],lambdaVector[site+1],gammaVector[site+1],schmidtError]\
            = EvolutionSite(expH2List[site],site,gammaVector,lambdaVector)
            error += schmidtError
        # Odd
        for i in range(int(nsites/2)):
            site = 2*i
            [gammaVector[site],lambdaVector[site+1],gammaVector[site+1],schmidtError]\
            = EvolutionSite(expH3List[site],site,gammaVector,lambdaVector)
            error += schmidtError
        # Even
        for i in range(int((nsites-1)/2)):
            site = 2*i+1
            [gammaVector[site],lambdaVector[site+1],gammaVector[site+1],schmidtError]\
            = EvolutionSite(expH4List[site],site,gammaVector,lambdaVector)
            error += schmidtError
        # Odd
        for i in range(int(nsites/2)):
            site = 2*i
            [gammaVector[site],lambdaVector[site+1],gammaVector[site+1],schmidtError]\
            = EvolutionSite(expH3List[site],site,gammaVector,lambdaVector)
            error += schmidtError
        # Even
        for i in range(int((nsites-1)/2)):
            site = 2*i+1
            [gammaVector[site],lambdaVector[site+1],gammaVector[site+1],schmidtError]\
            = EvolutionSite(expH2List[site],site,gammaVector,lambdaVector)
            error += schmidtError
        # Odd
        for i in range(int(nsites/2)):
            site = 2*i
            [gammaVector[site],lambdaVector[site+1],gammaVector[site+1],schmidtError]\
            = EvolutionSite(expH1List[site],site,gammaVector,lambdaVector)
            error += schmidtError

        if q == nstoreindex[currentStoreIndex]:
            # Compute the corresponding correlator!
            [correlationGrid,meanxGrid] = CorrelationGridMPSFixedTime(Op1,Op2,gammaVector,lambdaVector)
            correlationGridTime[currentStoreIndex] = correlationGrid
            meanxGridTime[currentStoreIndex] = meanxGrid
            errorTime.append(error)
            currentStoreIndex+=1 
            
    return [correlationGridTime,meanxGridTime,errorTime,gammaVector,lambdaVector]

def CorrelationGridMPSFixedTime(Op1,Op2,gammaVector,lambdaVector):
    
    nsites = len(gammaVector)
    
    correlationGrid = np.zeros((nsites,nsites),dtype=complex)

    for site2 in range(1,nsites-1):
        B = Op2[site2]

        # We compute the diagonal element <Op1(site2)Op2(site2)>
        #Checked: This is ok
        Op = Op1[site2].dot(Op2[site2])
        diagT= (Op.dot(RhoSite(site2,lambdaVector,gammaVector))).diagonal().sum()
        correlationGrid[site2,site2] = 0.5*diagT # Divided by 1/2 because we are summing it in the end



        # We compute the rest of correlations for site1 < site2
        Qtime= opGamma(B,gammaVector[site2])
        Gtime = GQ(lambdaVector[site2+1],Qtime,gammaVector[site2])   
        Gktime = Gtime
        # I fixed the next point, I had it to site2-1 and it was wrong
        for site1 in reversed(range(0,site2)):
            A = Op1[site1]
            Ptime = opGamma(A,gammaVector[site1])
            ABtime = AB(site1,lambdaVector,gammaVector,Ptime,Gktime) 
            correlationGrid[site1,site2] = ABtime

            # We compute Gktime for the next
            if site1 >0:
                Gktime = GK(site1,lambdaVector,gammaVector,Gktime) 

    # First diagonal element
    site2 = 0
    Op = Op1[site2].dot(Op2[site2])
    diagT= (Op.dot(RhoSite(site2,lambdaVector,gammaVector))).diagonal().sum()
    correlationGrid[site2,site2] = 0.5*diagT # Divided by 1/2 because we are summing it in the end


    correlationGrid = correlationGrid+ conjugate(transpose(correlationGrid))
    meanxGrid = np.asarray(np.diag(correlationGrid))
    correlationGrid = correlationGrid[1:,1:]
    
    return [correlationGrid,meanxGrid]


def opGamma(op,Gamma):
    #This computes one site operator acting on a site Gamma, B Gamma
    dims = Gamma.shape
    chi = dims[0]
    GammaNew = np.zeros(Gamma.shape,dtype = complex)
    for alpha1 in range(chi):
        for alpha2 in range(chi):
            GammaNew[alpha1,:,alpha2] = op.dot(Gamma[alpha1,:,alpha2])
    return GammaNew   

def GQ(lambdaSite,Q,gammaSite):
    dims = gammaSite.shape
    chi = dims[0]
    d = dims[1]
    gq = np.zeros((chi,chi),dtype = complex)
    for alpha in range(chi):
        for beta in range(chi) :
            # I change this temporarily because I'm not sure it's working properly
            #gq[alpha,beta] = np.sum(lambdaSite**2*conjugate(gammaSite[alpha,:,:])*Q[beta,:,:])
            # Result of this change: Different but still shit
            for beta2 in range(chi):
            	for i in range(d):
            		gq[alpha,beta] = gq[alpha,beta]+(lambdaSite[beta2])**2*Q[alpha,i,beta2]*conjugate(gammaSite[beta,i,beta2])
    return gq

def RhoSite(site,lambdaVector,gammaVector):
    dim = gammaVector[site].shape
    chi = dim[0]
    d = dim[1]

    rho = np.zeros((d,d),dtype=complex)
    for alpha1 in range(chi):
        for alpha2 in range(chi):
            gammaMatrix=gammaVector[site][alpha1,:,alpha2][np.newaxis]
            rho = rho + (lambdaVector[site][alpha1])**2*(lambdaVector[site+1][alpha2])**2*dot(transpose(gammaMatrix),conjugate(gammaMatrix))
            
    return rho

def AB(site,lambdaVector,gammaVector,P,Gk):
    dim = gammaVector[site].shape
    chi = dim[0]
    d = dim[1]
    ab = 0
    for alpha in range(chi):
        for beta in range(chi):
            for beta2 in range(chi):
                # I change this to put it like in the Mathematica code
                # Result: Changed, but not definitive
                #ab += lambdaVector[site][beta2]**2*dot(P[beta2,:,beta],gammaVector[site][beta2,:,alpha])\
                #*lambdaVector[site+1][alpha]*lambdaVector[site+1][beta]*Gk[alpha,beta]
                
                ab += lambdaVector[site][beta2]**2*dot(P[beta2,:,alpha],conjugate(gammaVector[site][beta2,:,beta]))\
                *lambdaVector[site+1][alpha]*lambdaVector[site+1][beta]*Gk[alpha,beta]
    return ab

def GK(site,lambdaVector,gammaVector,Gk):
    dim = gammaVector[site].shape
    chi = dim[0]
    d = dim[1]
    gk = np.zeros((chi,chi),dtype=complex)
    
    for alpha in range(chi):
        for beta in range(chi):
            for alpha2 in range(chi):
                for beta2 in range(chi):
                    #gk[alpha,beta]+= lambdaVector[site+1][alpha2]*lambdaVector[site+1][beta2]\
                    #*dot(gammaVector[site][alpha,:,alpha2],conjugate(gammaVector[site][beta,:,beta2]))\
                    #*Gk[alpha2,beta2]
                    
                    gk[alpha,beta]+= lambdaVector[site+1][alpha2]*lambdaVector[site+1][beta2]\
                    *dot(gammaVector[site][alpha,:,alpha2],conjugate(gammaVector[site][beta,:,beta2]))\
                    *Gk[alpha2,beta2]
    return gk

def AcorrelationsTime(correlationGrid):

    dims = correlationGrid.shape
    nstorecorr = dims[0]
    ncavs = dims[1]
    
    N = ncavs -1
    Qtable = np.zeros((ncavs,ncavs),dtype = complex)
    for n in range(ncavs):
        for m in range(ncavs):
            Qtable[n,m] = QHahn(n,m,N)
            
    rhon2table = [rhon2(n,N) for n in range(ncavs)]

    KQMatrix = [[ dot(transpose((Qtable[:,i]/sqrt(rhon2table))[np.newaxis]), (Qtable[:,j]/sqrt(rhon2table))[np.newaxis])\
             for j in range(ncavs)] for i in range(ncavs)]

    acorrelationMatrixTime = np.zeros((nstorecorr,ncavs,ncavs),dtype=complex)
    for k in range(ncavs):
        for t in range(nstorecorr):
            acorrelationMatrixTime[t,k,k]=0.5*sqrt(k)*sqrt(k)*np.sum(KQMatrix[k][k]*correlationGrid[t,:,:])
        for q in range(k+1,ncavs):
            for t in range(nstorecorr):
                acorrelationMatrixTime[t,k,q]= sqrt(k)*sqrt(q)*np.sum(KQMatrix[k][q]*correlationGrid[t,:,:]) 

    for t in range(nstorecorr):
        acorrelationMatrixTime[t,:,:] = acorrelationMatrixTime[t,:,:]+conjugate(transpose(acorrelationMatrixTime[t,:,:]))
        
    return acorrelationMatrixTime
    
 
    
