import sys
import numpy as np
import pyobs

def GEVP(C, t0=-1, const=1, case='ratio'): 
    Noss, Noss1, D = C.shape
    if not Noss == Noss1:
        raise Exception("Non-square matrix along axes=(0,1).")
    
    if t0 == -1: 
    # CASE 1  :  t-t0=const, fix const, span t,t0 values.
        rx = np.arange(D-const)
        lx  = rx + const
        ground  = []
        excited = []
        times   = []
        for (t,_t0) in zip(lx,rx):
            A    = pyobs.remove_tensor(C[:,:,t])
            B    = pyobs.remove_tensor(C[:,:,_t0])
            Bmsq = pyobs.linalg.matrix_power(B, -0.5)
            try:
                [lam, v] = pyobs.linalg.eig(Bmsq @ A @ Bmsq)
                E0 = (-pyobs.log(lam)/const)[0]
                e, de = E0.error()
                if np.abs(de/e) >= 1.0:
                    break
                ground.append(E0)
                times.append(np.array([t, _t0]))
                # print(f"(t,t_0) = ({t},{t0})  :  {ground[-1]}", sep='')
            except:
                print(f"(t,t0)=({t},{_t0})  :  StN problem, array containing Infs or NaNs.\n")
    else: 
    # CASE 2  :  fix t0 and span t values.
        if not (isinstance(t0, int) and t0 >= 0 and t0 < D-1):
            raise Exception(f"Since 't0' is not -1 it should be an integer in between 0 and {D-1}.")
        B    = pyobs.remove_tensor(C[:,:,t0])
        Bmsq = pyobs.linalg.matrix_power(B, -0.5)
        ground  = []
        excited = []
        times   = []
        for t in range(t0+1, D-1):
            A  = pyobs.remove_tensor(C[:,:,t])
            At = pyobs.remove_tensor(C[:,:,t+1])
            if case == 'ratio':
                try:
                    [lam, v]   = pyobs.linalg.eig(Bmsq @ A  @ Bmsq)
                    [lamt, vt] = pyobs.linalg.eig(Bmsq @ At @ Bmsq)
                    E0 = pyobs.log(lam/lamt)[0]
                    e, de = E0.error()
                    if np.abs(de/e) >= 1.0:
                        break
                    ground.append(E0)
                    times.append(t)
                except:
                    print(f"(t,t0)=({t},{t0})  :  StN problem, array containing Infs or NaNs.\n")
            elif case == 'single':
                try:
                    [lam, v]   = pyobs.linalg.eig(Bmsq @ A  @ Bmsq)
                    E0 = (-pyobs.log(lam)/(t-t0))[0]
                    e, de = E0.error()
                    if np.abs(de/e) >= 1.0:
                        break
                    ground.append(E0)
                    times.append(t)
                except:
                    print(f"(t,t0)=({t},{t0})  :  StN problem, array containing Infs or NaNs.\n")
            else:
                raise Exception(f"'case' should be 'ratio' or 'single'.")
    
    ground = pyobs.remove_tensor(pyobs.stack(ground))    
    return np.array(times), ground