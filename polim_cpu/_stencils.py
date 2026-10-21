"""Generated scalar stencils, derived from PoLIM classic; see LICENSE/NOTICE.
Regenerate with python -m polim_cpu.build_stencils.
"""

def interior(LHS, L1, L2, L3, L4, L5, L6, i, j, k, N, Ms, dx, dzeta):
    if i==2:
        LHS[k,k-N-1] = 2*L3[j,i]/(3*dx*dzeta)
        LHS[k,k-N] = 8*L1[j,i]/(3*dx**2) - 4*L5[j,i]/(3*dx)
        LHS[k,k-N+1] = -2*L3[j,i]/(3*dx*dzeta)
        LHS[k,k-1] = L2[j,i]/dzeta**2-L3[j,i]/(2*dx*dzeta) - L4[j,i]/(2*dzeta)
        LHS[k,k] = -4*L1[j,i]/(dx**2) - 2*L2[j,i]/dzeta**2 + L5[j,i]/dx+L6[j,i]
        LHS[k,k+1] = L2[j,i]/(dzeta**2) + L3[j,i]/(2*dx*dzeta) + L4[j,i]/(2*dzeta)
        LHS[k,k+N-1] = -L3[j,i]/(6*dx*dzeta)
        LHS[k,k+N] = 4*L1[j,i]/(3*dx**2) + L5[j,i]/(3*dx)
        LHS[k,k+N+1] = L3[j,i]/(6*dx*dzeta)
    elif i==Ms-1:
        LHS[k,k-N-1] = L3[j,i]/(6*dx*dzeta)
        LHS[k,k-N] = 4*L1[j,i]/(3*dx**2) - L5[j,i]/(3*dx)
        LHS[k,k-N+1] = -L3[j,i]/(6*dx*dzeta)
        LHS[k,k-1] = L2[j,i]/(dzeta**2) + L3[j,i]/(2*dx*dzeta) - L4[j,i]/(2*dzeta)
        LHS[k,k] = -4*L1[j,i]/(dx**2) - 2*L2[j,i]/dzeta**2 - L5[j,i]/dx + L6[j,i]
        LHS[k,k+1] = L2[j,i]/(dzeta**2) - L3[j,i]/(2*dx*dzeta) + L4[j,i]/(2*dzeta)
        LHS[k,k+N-1] = -2*L3[j,i]/(3*dx*dzeta)
        LHS[k,k+N] = 8*L1[j,i]/(3*dx**2) + 4*L5[j,i]/(3*dx)
        LHS[k,k+N+1] = 2*L3[j,i]/(3*dx*dzeta)
    else:
        LHS[k,k-N-1] = L3[j,i]/(4*dx*dzeta)
        LHS[k,k-N] = L1[j,i]/(dx**2) - L5[j,i]/(2*dx)
        LHS[k,k-N+1] = -L3[j,i]/(4*dx*dzeta)
        LHS[k,k-1] = L2[j,i]/(dzeta**2) - L4[j,i]/(2*dzeta)
        LHS[k,k] = -2*L1[j,i]/(dx**2) - 2*L2[j,i]/dzeta**2 + L6[j,i]
        LHS[k,k+1] = L2[j,i]/dzeta**2 + L4[j,i]/(2*dzeta)
        LHS[k,k+N-1] = -L3[j,i]/(4*dx*dzeta)
        LHS[k,k+N] = L1[j,i]/(dx**2) + L5[j,i]/(2*dx)
        LHS[k,k+N+1] = L3[j,i]/(4*dx*dzeta)

def surface(LHS, L1, L2, L3, L4, L5, Ls1, Ls2, LsW, N, Ms, dx, dzeta):
    for i in range(2, Ms-1+1):
        if i==2:
            LHS[i*N, (i-1)*N] = 8*L1[N,i]/(3*dx**2) - 8/3*L2[N,i]*Ls1[i]/(dzeta**2) +                 L3[N,i]*(64/3*Ls1[i-1]-4*Ls2[i-1]*LsW[i-1]-8*Ls1[i])/(6*dx*dzeta) -                 8/3*L4[N,i]*Ls1[i]/(2*dzeta) - 4*L5[N,i]/(3*dx)
            LHS[i*N, i*N-1] = 2*L2[N,i]/(dzeta**2)
            LHS[i*N, i*N] = -12*L1[N,i]/(3*dx**2) +                L2[N,i]*(2*Ls1[i]+Ls2[i]*LsW[i]-2)/(dzeta**2) +                 L3[N,i]*(-24*Ls1[i-1]+6*Ls1[i]+Ls2[i]*LsW[i]-Ls1[i+1])/(6*dx*dzeta) +                 L4[N,i]*(2*Ls1[i]+Ls2[i]*LsW[i])/(2*dzeta) + 3*L5[N,i]/(3*dx)
            LHS[i*N, (i+1)*N] = 4*L1[N,i]/(3*dx**2) + 2/3*L2[N,i]*Ls1[i]/(dzeta**2) +                 L3[N,i]*(8/3*Ls1[i-1]+2*Ls1[i]+Ls2[i+1]*LsW[i+1])/(6*dx*dzeta) +                 L4[N,i]*(2/3*Ls1[i])/(2*dzeta) + L5[N,i]/(3*dx)
            LHS[i*N, (i+2)*N] = L3[N,i]*Ls1[i+1]/(6*dx*dzeta)
        elif i==Ms-1:
            LHS[i*N,(i-2)*N] = L3[N,i]*Ls1[i-1]/(6*dx*dzeta)
            LHS[i*N,(i-1)*N] = 4*L1[N,i]/(3*dx**2) - 2/3*L2[N,i]*Ls1[i]/(dzeta**2) +                 L3[N,i]*(-Ls2[i-1]*LsW[i-1]+2*Ls1[i]+8/3*Ls1[i+1])/(6*dx*dzeta)                -2/3*L4[N,i]*Ls1[i]/(2*dzeta) - L5[N,i]/(3*dx)
            LHS[i*N,i*N-1] = 2*L2[N,i]/(dzeta**2)
            LHS[i*N,i*N] = -12*L1[N,i]/(3*dx**2) +                L2[N,i]*(-2*Ls1[i]+Ls2[i]*LsW[i]-2)/(dzeta**2) +                 L3[N,i]*(-Ls1[i-1]+6*Ls1[i]-3*Ls2[i]*LsW[i]-24*Ls1[i+1])/(6*dx*dzeta) +                 L4[N,i]*(-2*Ls1[i]+Ls2[i]*LsW[i])/(2*dzeta) - 3*L5[N,i]/(3*dx)
            LHS[i*N,(i+1)*N] = 8*L1[N,i]/(3*dx**2) + 8/3*L2[N,i]*Ls1[i]/(dzeta**2) +                 L3[N,i]*(-8*Ls1[i]+64/3*Ls1[i+1]+4*Ls2[i+1]*LsW[i+1])/(6*dx*dzeta) +                 8/3*L4[N,i]*Ls1[i]/(2*dzeta) + 4*L5[N,i]/(3*dx)
        else:
            LHS[i*N,(i-2)*N] = L3[N,i]*Ls1[i-1]/(4*dx*dzeta)
            LHS[i*N,(i-1)*N] = L1[N,i]/(dx**2) - L2[N,i]*Ls1[i]/(dzeta**2) -                 L3[N,i]*Ls2[i-1]*LsW[i-1]/(4*dx*dzeta) -                L4[N,i]*Ls1[i]/(2*dzeta) - L5[N,i]/(2*dx)
            LHS[i*N,i*N-1] = 2*L2[N,i]/(dzeta**2)
            LHS[i*N,i*N] = -2*L1[N,i]/(dx**2) + L2[N,i]*(Ls2[i]*LsW[i]-2)/(dzeta**2) -                L3[N,i]*(Ls1[i-1]+Ls1[i+1])/(4*dx*dzeta) + L4[N,i]*Ls2[i]*LsW[i]/(2*dzeta)
            LHS[i*N,(i+1)*N] = L1[N,i]/(dx**2) + L2[N,i]*Ls1[i]/(dzeta**2) +                L3[N,i]*Ls2[i+1]*LsW[i+1]/(4*dx*dzeta) +                L4[N,i]*Ls1[i]/(2*dzeta) + L5[N,i]/(2*dx)
            LHS[i*N,(i+2)*N] = L3[N,i]*Ls1[i+1]/(4*dx*dzeta)

def basal_linear(LHS, L1, L2, L3, L4, L5, L6, Lb1_LFL, Lb2_LFL, N, Ms, dx, dzeta):
    for i in range(2, Ms-1+1):
        if i==2:
            LHS[(i-1)*N+1, (i-2)*N+1] = 8*L1[1,i]/(3*dx**2) +                8*Lb1_LFL[i]*L2[1,i]/(3*dzeta**2) +                (64*Lb1_LFL[i-1]/3-4*Lb2_LFL[i-1]-8*Lb1_LFL[i])*L3[1,i]/(6*dx*dzeta) -                8/3*Lb1_LFL[i]*L4[1,i]/(2*dzeta) - 4*L5[1,i]/(3*dx)
            LHS[(i-1)*N+1, (i-1)*N+1] = -12*L1[1,i]/(3*dx**2) -                (2+2*Lb1_LFL[i]+Lb2_LFL[i])*L2[1,i]/(dzeta**2) +                (-24*Lb1_LFL[i-1]+6*Lb1_LFL[i]+3*Lb2_LFL[i]-Lb1_LFL[i+1])*L3[1,i]/(6*dx*dzeta) +                (2*Lb1_LFL[i]+Lb2_LFL[i])*L4[1,i]/(2*dzeta) +                3*L5[1,i]/(3*dx) + L6[1,i]
            LHS[(i-1)*N+1, (i-1)*N+2] = 2*L2[1,i]/(dzeta**2)
            LHS[(i-1)*N+1, i*N+1] = 4*L1[1,i]/(3*dx**2) -                2/3*Lb1_LFL[i]*L2[1,i]/(dzeta**2) +                (8/3*Lb1_LFL[i-1]+2*Lb1_LFL[i]+Lb2_LFL[i+1])*L3[1,i]/(6*dx*dzeta) +                2/3*Lb1_LFL[i]*L4[1,i]/(2*dzeta) + L5[1,i]/(3*dx)
            LHS[(i-1)*N+1, (i+1)*N+1] = Lb1_LFL[i]*L3[1,i]/(6*dx*dzeta)
        elif i==Ms-1:
            LHS[(i-1)*N+1, (i-3)*N+1] = Lb1_LFL[i-1]*L3[1,i]/(6*dx*dzeta)
            LHS[(i-1)*N+1, (i-2)*N+1] = 4*L1[1,i]/(3*dx**2) +                2/3*Lb1_LFL[i]*L2[1,i]/(dzeta**2) +                (-Lb2_LFL[i-1]+2*Lb1_LFL[i]+8/3*Lb1_LFL[i+1])*L3[1,i]/(6*dx*dzeta) -                2/3*Lb1_LFL[i]*L4[1,i]/(2*dzeta) - L5[1,i]/(3*dx)
            LHS[(i-1)*N+1, (i-1)*N+1] = -12*L1[1,i]/(3*dx**2) +                (-2+2*Lb1_LFL[i]-Lb2_LFL[i])*L2[1,i]/(dzeta**2) +                 (-Lb1_LFL[i-1]+6*Lb1_LFL[i]-3*Lb2_LFL[i]-24*Lb1_LFL[i+1])*L3[1,i]/(6*dx*dzeta) +                (-2*Lb1_LFL[i]+Lb2_LFL[i])*L4[1,i]/(2*dzeta) -                3*L5[1,i]/(3*dx) + L6[1,i]
            LHS[(i-1)*N+1, (i-1)*N+2] = 2*L2[1,i]/(dzeta**2)
            LHS[(i-1)*N+1, i*N+1] = 8*L1[1,i]/(3*dx**2) -                8/3*Lb1_LFL[i]*L2[1,i]/(dzeta**2) +                (-8*Lb1_LFL[i]+64/3*Lb1_LFL[i+1]+4*Lb2_LFL[i+1])*L3[1,i]/(6*dx*dzeta) +                8/3*Lb1_LFL[i]*L4[1,i]/(2*dzeta) + 4*L5[1,i]/(3*dx)
        else:
            LHS[(i-1)*N+1, (i-3)*N+1] = Lb1_LFL[i-1]*L3[1,i]/(4*dx*dzeta)
            LHS[(i-1)*N+1, (i-2)*N+1] = L1[1,i]/(dx**2) +                Lb1_LFL[i]*L2[1,i]/(dzeta**2) -                Lb2_LFL[i-1]*L3[1,i]/(4*dx*dzeta) -                Lb1_LFL[i]*L4[1,i]/(2*dzeta) - L5[1,i]/(2*dx)
            LHS[(i-1)*N+1, (i-1)*N+1] = -2*L1[1,i]/(dx**2) -                (2+Lb2_LFL[i])*L2[1,i]/(dzeta**2) -                (Lb1_LFL[i+1]+Lb1_LFL[i-1])*L3[1,i]/(4*dx*dzeta) +                Lb2_LFL[i]*L4[1,i]/(2*dzeta) + L6[1,i]
            LHS[(i-1)*N+1, (i-1)*N+2] = 2*L2[1,i]/(dzeta**2)
            LHS[(i-1)*N+1, i*N+1] = L1[1,i]/(dx**2) -                Lb1_LFL[i]*L2[1,i]/(dzeta**2) +                Lb2_LFL[i+1]*L3[1,i]/(4*dx*dzeta) +                Lb1_LFL[i]*L4[1,i]/(2*dzeta) + L5[1,i]/(2*dx)
            LHS[(i-1)*N+1, (i+1)*N+1] = Lb1_LFL[i+1]*L3[1,i]/(4*dx*dzeta)
