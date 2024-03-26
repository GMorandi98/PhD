import numpy as np
import pyobs

def GEVP(C, t0=-1, const=1, case='ratio'): 
    Noss, Noss1, D = C.shape
    assert (Noss == Noss1), f"Check shape of 'C': non-square matrix along axes=(0,1)."
    
    if t0 == -1:        ### CASE 1  :  t-t0=const, fix const, span t,t0 values. ###
        rx = np.arange(D-const)
        lx  = rx + const
        ground  = []
        times   = []
        for (t,_t0) in zip(lx,rx):
            Ct      = pyobs.remove_tensor(C[:,:,t])
            Ct0_inv = pyobs.linalg.inv(pyobs.remove_tensor(C[:,:,_t0]))
            try:
                ### check with symmetric 'eig' ###
                # Bmsq = pyobs.linalg.matrix_power(pyobs.remove_tensor(C[:,:,_t0]), -0.5)
                # [lam, v] = pyobs.linalg.eig(Bmsq @ Ct @ Bmsq)
                # E0 = (-pyobs.log(lam)/const)[0]
                #############
                [lam, rv, lv] = pyobs.linalg.eigLR(Ct0_inv @ Ct)
                E0 = (-pyobs.log(lam)/const)[-1]    
                e, de = E0.error()
                if np.abs(de/e) >= 1.0:
                    break
                ground.append(E0)
                times.append(np.array([t, _t0]))
            except:
                print(f"(t,t0)=({t},{_t0})  :  Array containing Infs or NaNs.\n")
    else:               ### CASE 2  :  fix t0 and span t values. ###
        assert (isinstance(t0, int) and t0 >= 0 and t0 < D-1), \
            f"'t0' should be an integer in between 0 and {D-1}, got t0={t0}."
        Ct0_inv = pyobs.linalg.inv(pyobs.remove_tensor(C[:,:,t0]))
        ground  = []
        times   = []
        for t in range(t0+1, D-1):
            Ct  = pyobs.remove_tensor(C[:,:,t])
            Ct1 = pyobs.remove_tensor(C[:,:,t+1])
            if case == 'ratio':
                try:
                    ### check with symmetric 'eig' ###
                    # Bmsq = pyobs.linalg.matrix_power(pyobs.remove_tensor(C[:,:,t0]), -0.5)
                    # [lam, v]   = pyobs.linalg.eig(Bmsq @ Ct @ Bmsq)
                    # [lam1, v1] = pyobs.linalg.eig(Bmsq @ Ct1 @ Bmsq)
                    # E0 = pyobs.log(lam/lam1)[0]
                    #############
                    [lam,  rv,  lv]  = pyobs.linalg.eigLR(Ct0_inv @ Ct)
                    [lam1, rv1, lv1] = pyobs.linalg.eigLR(Ct0_inv @ Ct1)
                    E0 = pyobs.log(lam/lam1)[-1]
                    e, de = E0.error()
                    if np.abs(de/e) >= 1.0:
                        break
                    ground.append(E0)
                    times.append(t)
                except:
                    print(f"(t,t0)=({t},{t0})  :  StN problem, array containing Infs or NaNs.\n")
            elif case == 'single':
                try:
                    ### check with symmetric 'eig' ###
                    # Bmsq = pyobs.linalg.matrix_power(pyobs.remove_tensor(C[:,:,t0]), -0.5)
                    # [lam, v] = pyobs.linalg.eig(Bmsq @ Ct @ Bmsq)
                    # E0 = (-pyobs.log(lam)/(t-t0))[0]
                    #############
                    [lam,  rv,  lv]  = pyobs.linalg.eigLR(Ct0_inv @ Ct)
                    E0 = (-pyobs.log(lam)/(t-t0))[-1]
                    e, de = E0.error()
                    if np.abs(de/e) >= 1.0:
                        break
                    ground.append(E0)
                    times.append(t)
                except:
                    print(f"(t,t0)=({t},{t0})  :  StN problem, array containing Infs or NaNs.\n")
            else:
                raise Exception(f"'case' should be 'ratio' or 'single', got 'case'={case}")
    ground = pyobs.remove_tensor(pyobs.stack(ground))    
    return np.array(times), ground