#
#!/usr/bin/env python3
#
# Author: Gabriele Morandi, November 2023
#
# HMC and Wilson Flow for Yang-Mills SU(3) theory
#
import gpt as g
import sys, os
import numpy as np 
import bison


g.default.set_verbose(False)
g.default.set_verbose("step_size", False)


params = g.params("run_params_therm.txt")

#################################
### General global parameters ###
#################################
L1, L2, L3, T = params["Lattice"]
beta       = params["beta"]
tau        = params["tau"]           # length of MD trajectory
Ns         = params["Ns"]            # integration steps of OMF4 for HMC
Ntherm     = params["Ntherm"]        # thermalization trajectories
seed    = params["seed"]


grid = g.grid([L1, L2, L3, T], g.double)
rng = g.random(seed)

U = g.qcd.gauge.unit(grid)
# rng.normal_element(U)
rng.normal_element(U)

# links conjugate momenta
pi = g.group.cartesian(U)


#########################################
### General informations for Log file ###
#########################################
g.message(f"  Lattice  =  {grid.fdimensions}")

a0 = g.qcd.scalar.action.mass_term() # action for conjugate momenta
a1 = g.qcd.gauge.action.wilson(beta) # Wilson action

g.message()
g.message("  Actions: ")
g.message(f"   -- {a0.__name__}")
g.message(f"   -- {a1.__name__} with coupling beta = {beta}")


####################################
### HMC setup: MD and Metropolis ###
####################################

# molecular dynamics
sympl = g.algorithms.integrator.symplectic

ip = sympl.update_p(pi, lambda: a1.gradient(U, U))
iq = sympl.update_q(U,  lambda: a0.gradient(pi, pi))

# integrator
mdint = sympl.OMF4(Ns, ip, iq)
g.message()
g.message(f"  Integration scheme:\n{mdint}")

# metropolis
metro = g.algorithms.markov.metropolis(rng)

# MD units
g.message(f"  tau = {tau} MD units")



def HMC(tau, pi):
    rng.normal_element(pi)
    accrej = metro(U)
    S_i = a1(U)
    H_i = a0(pi) + S_i
    mdint(tau) # MD evolution
    S_f = a1(U) 
    H_f = a0(pi) + S_f 
    return [accrej(H_f, H_i), H_f - H_i]
    #return [accrej(H_f, H_i), H_i, H_f, S_i, S_f]



######################
### Thermalization ###
######################
g.message()
g.message()
g.message(f"----------------------------------------------------------------------------------------")
g.message(f"------------------------------ Starting thermalization... ------------------------------")
g.message(f"----------------------------------------------------------------------------------------")
g.message()
g.message()

h    = []
plaq = []
timer = g.timer("HMC")
for i in range(Ntherm):
    timer("therm traj")
    h    += [HMC(tau, pi)]
    plaq += [g.qcd.gauge.plaquette(U)]
    timer()
    g.message(f"  ---- Trajectory {i+1}:  time = {timer.time['therm traj'].dt_last:.2f} secs, {timer.time['therm traj'].dt_last / 60:.2f} mins ----  ")
    g.message(f"       Plaq       =  {plaq[-1]}")
    g.message(f"       Acc/Rej    =  {h[-1][0]}")
    g.message(f"       dH         =  {h[-1][1]}")
    g.message()
g.message(f"  Thermalization time: {timer.time['therm traj'].dt_sum:.2f} secs ({timer.time['therm traj'].dt_sum / 60:.2f} mins, {timer.time['therm traj'].dt_sum / 3600:.2f} hrs), ")
g.message() 

h = np.array(h) 
plaq = np.array(plaq)

obs = {
        'MC History'   :   h, \
        'Plaquette'    :   plaq,
      }

g.message(f"  < Plaq >    =  {np.mean(plaq)}")
g.message(f"  < |dH| >    =  {np.mean(np.abs(h[:, 1]))}")
g.message(f"  Acceptance  =  {np.mean(h[:, 0])}")

g.message()
if g.rank() == 0:
    file = bison.FILE(f"../../Data_Analysis/Pure_gauge/WF_beta{int(beta*1e2)}_lat{L1}^3x{T}_Ntherm{Ntherm}_seed{int(seed)}.dat", mode="w")
    file.write('beta coupling', beta)
    file.write('Lattice size: L1', L1)
    file.write('Lattice size: L2', L2)
    file.write('Lattice size: L3', L3)
    file.write('Lattice size: T', T)
    file.write('Lattice volume', grid.fsites)
    file.write('Number of steps of OMF4 integrator', Ns)
    file.write('Length of each MD trajectory', tau)
    file.write('Thermalization trajectories', Ntherm)
    file.write('History & Measurments', obs)
    file.close()
g.message()

