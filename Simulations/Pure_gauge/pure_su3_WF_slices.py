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


params = g.params("run_params.txt")

#################################
### General global parameters ###
#################################
L1, L2, L3, T = params["Lattice"]
beta       = params["beta"]
tau        = params["tau"]                      # length of MD trajectory
Ns         = params["Ns"]                       # integration steps of OMF4 for HMC
Ntherm     = params["Ntherm"]                   # thermalization trajectories
therm_step = params["therm_step"]
Ntraj      = params["Ntraj"]                    # trajectories for data production
MC_step    = params["MC_measure_step"]
WF_evol    = params["WF_evolutions"]
WF_step    = params["WF_measure_step"]
eps_WF     = params["eps_WF"]
run        = params["run"]
seed       = f"run{run}"


grid = g.grid([L1, L2, L3, T], g.double)
rng = g.random(seed)

U = g.qcd.gauge.unit(grid)
rng.normal_element(U)

# links conjugate momenta
pi = g.group.cartesian(U)


#########################################
### General informations for Log file ###
#########################################
g.message()
g.message(f"  Lattice  =  {grid.fdimensions}")
vol = grid.fsites

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



def HMC(field, tau, pi):
    rng.normal_element(pi)
    accrej = metro(field)
    S_i = a1(field)
    H_i = a0(pi) + S_i
    mdint(tau) # MD evolution
    S_f = a1(field) 
    H_f = a0(pi) + S_f 
    return [accrej(H_f, H_i), H_f - H_i]
    #return [accrej(H_f, H_i), H_f - H_i, H_f, H_i, S_f, S_i]



##############################################################################
if (params["init_cnfg"] is None) and (run == 0):
    # extract some random configuration for gauge links and thermalize the MC chain:
    rng.normal_element(U)

    ######################
    ### Thermalization ###
    ######################
    g.message()
    g.message()
    g.message(f"----------------------------------------------------------------------------------------")
    g.message(f"------------------------------ Starting thermalization... ------------------------------")
    g.message(f"----------------------------------------------------------------------------------------")
    g.message()
    g.message(f"We perform in total {Ntherm} trajectories:")
    g.message()

    for i in range(therm_step):
        h    = []
        plaq = []
        timer = g.timer("HMC")
        for j in range(Ntherm // therm_step):
            timer("trajectory")
            h    += [HMC(U, tau, pi)]
            plaq += [g.qcd.gauge.plaquette(U)]
            g.message(f"     ---- Trajectory {i * (Ntherm // therm_step) + j + 1},  Acc/Rej = {h[-1][0]},  dH = {h[-1][1]},  Plaq = {plaq[-1]}, ----  ")
            g.message()
        timer()
        h    = np.array(h) 
        plaq = np.array(plaq)
        g.message(f" |-|-|-|-| {(i+1)*(Ntherm//therm_step)/Ntherm*100:.0f}% of Thermalization completed |-|-|-|-|  ")
        g.message(f"       < Plaq > = {np.mean(plaq):.4f},  Acceptance = {np.mean(h[:, 0])*100:.0f}%,  <|dH|> = {np.mean(np.abs(h[:, 1])):.3e},  ")
        g.message(timer)
        g.message()
    
    g.save(f"Cnfgs_checkpts/ckpoint_beta{int(beta*1e2)}_lat{L1}x{T}.{run}", U, g.format.nersc())

##############################################################################
else:
    # load a configuration:
    g.message()
    g.message(" Loading configuration:")
    g.message(f" {params['init_cnfg']}")
    g.message()
    U = g.load(params["init_cnfg"])
    g.message()

    ip = sympl.update_p(pi, lambda: a1.gradient(U, U))
    iq = sympl.update_q(U,  lambda: a0.gradient(pi, pi))

    # integrator
    mdint = sympl.OMF4(Ns, ip, iq)

    if params['thermalization'] and (int(params['init_cnfg'].split(f"x{T}.")[-1]) == run - 1):
        # continue thermalization
        g.message()
        g.message()
        g.message(f"------------------------------------------------------------------------------------------")
        g.message(f"------------------------------ Continuing thermalization... ------------------------------")
        g.message(f"------------------------------------------------------------------------------------------")
        g.message()
        g.message(f"We perform in total {Ntherm} trajectories:")
        g.message()

        for i in range(therm_step):
            h    = []
            plaq = []
            timer = g.timer("HMC")
            for j in range(Ntherm // therm_step):
                timer("trajectory")
                h    += [HMC(U, tau, pi)]
                plaq += [g.qcd.gauge.plaquette(U)]
                g.message(f"     ---- Trajectory {i * (Ntherm // therm_step) + j + 1},  Acc/Rej = {h[-1][0]},  dH = {h[-1][1]},  Plaq = {plaq[-1]}, ----  ")
                g.message()
            timer()
            h    = np.array(h) 
            plaq = np.array(plaq)
            g.message(f" |-|-|-|-| {(i+1)*(Ntherm//therm_step)/Ntherm*100:.0f}% of Thermalization completed |-|-|-|-|  ")
            g.message(f"       < Plaq > = {np.mean(plaq):.4f},  Acceptance = {np.mean(h[:, 0])*100:.0f}%,  <|dH|> = {np.mean(np.abs(h[:, 1])):.3e},  ")
            g.message(timer)
            g.message()
        
        g.message()
        g.message("Saving last configuration...")
        g.message()
        g.save(f"Cnfgs_checkpts/ckpoint_beta{int(beta*1e2)}_lat{L1}x{T}.{run}", U, g.format.nersc())
        g.message()
        g.message()
        if g.rank() == 0:
            os.remove(f"{params['init_cnfg']}")
            g.message(f"Previous configuration '{params['init_cnfg']}' has been deleted.")
        g.message()
    elif not params['thermalization']:
        # start measuring observables
    
        # you can either create a new file to save the observables, or append to an 
        # existing one in params["data_file"]. In the former case the loaded config. 
        # is after thermalization, instead in the latter one, the configuration
        # is the last saved after a given evoution of Ntraj trajectories. 
        if (params["data_file"] is None) and (run == 1):
            # create new file:
            g.message()
            if g.rank() == 0:
                file = bison.FILE(f"../../Data_Analysis/Pure_gauge/Cnfgs_measurement/WF-slices_beta{int(beta*1e2)}_lat{L1}x{T}.dat", mode="w")
                file.write('beta coupling', beta)
                file.write('Lattice', (L1, L2, L3, T))
                file.write("MD trajectories", Ntraj)
                file.write('Number of steps of OMF4 integrator', Ns)
                file.write('Length of each MD trajectory', tau)
                file.write('MC measure step', MC_step)
                file.write('epsilon WF', eps_WF)
                file.write('WF evolutions from t = 0', WF_evol)
                file.write('WF measure step', WF_step)
            g.message()
        elif (params["data_file"] is not None) and isinstance(run, int):
            if run < 0:
                g.message(" Warning ! Check 'run' in parameters file.")
                g.message("           Remember to kill the processes.")
            else:
                if int(params['init_cnfg'].split(f"x{T}.")[-1]) == run - 1:     # to extract the label of "init_cnfg"
                    g.message()
                    g.message(" Append new data to an existing file:")
                    g.message(f" {params['data_file']}")
                    if g.rank() == 0:
                        file = bison.FILE(params["data_file"], mode='a')
                    g.message()
                else:
                    g.message()
                    g.message(" Warning ! The configuration label + 1 should be equal to run\
            in order to append to 'data_file'.")
                    g.message("           Remember to kill the processes.")
                    g.message()
        else:
            g.message()
            g.message(" Warning ! There is something wrong in 'init_cnfg' file or 'run' parameter.")
            g.message("           Remember to kill the processes.")
            g.message()
        

        #####################################
        ### Wilson Flow & Data Production ###
        #####################################
        g.message()
        g.message(f"--------------------------------------------------------------------------------------")
        g.message(f"---------------------------------- Data Production: ----------------------------------")
        g.message(f"--------------------------------------------------------------------------------------")
        g.message()
        g.message()
        g.message(f"  We compute in total {Ntraj} trajectories:  ")
        g.message()
        g.message()

        Nconf = Ntraj // MC_step
        N_WF  = WF_evol // WF_step + 1      # +1 since we measure also t = 0.0

        # Tempo di calcolo circa lo stesso del WF (Plaquette circa staple + link)
        def plaquette_slice(U, dim):
            Nd = len(U)
            # vol = float(U[0].grid.fsites)
            ndim = U[0].otype.shape[0]
            tr = g.complex(grid)
            tr[:] = 0.0
            for mu in range(Nd):
                    for nu in range(mu):
                            tr += g.trace(
                                            U[mu] * \
                                            g.cshift(U[nu], mu, 1) * \
                                            g.adj(g.cshift(U[mu], nu, 1)) * \
                                            g.adj(U[nu])
                                        )
            tr = np.array(g.slice(tr, dim)).real
            return 2.0 * tr / Nd / (Nd - 1) / ndim
            #return 2.0 * tr / vol / Nd / (Nd - 1) / ndim 

        history = {
                    'dH'                :       np.zeros(Ntraj), \
                    'Acc/Rej'           :       np.zeros(Ntraj),
                }

        obs = {
                'E_Clov'            :       np.zeros((N_WF, T)), \
                'Plaquette'         :       np.zeros((N_WF, T)), \
                'Q'                 :       np.zeros((N_WF, T))       
            }


        timer = g.timer("HMC")
        for i in range(1, Ntraj+1):
            timer("trajectory")
            h = HMC(U, tau, pi) # HMC evolves U field
            timer()
            history['Acc/Rej'][i-1] = int(h[0])
            history['dH'][i-1]      = h[1]
            g.message(f" ------ MD trajectory {i}  ------  ")
            g.message(f"        Acc/Rej = {h[0]},  dH = {h[1]:.3e},  ")
            g.message(timer)
            g.message()
            if (i % MC_step == 0):
                timer("WF & measurement")
                g.message(f"    ---- | Measurement at MC time {i} | ----")
                g.message()
                ##### Measurement #####
                obs['E_Clov'][0]    = np.array(g.slice(g.qcd.gauge.energy_density(U, field=True), 3)).real
                obs['Plaquette'][0] = plaquette_slice(U, 3)
                obs['Q'][0]         = np.array(g.slice(g.qcd.gauge.topological_charge(U, field=True), 3)).real
                g.message(f"        -- |  tWF = 0.00,  :  E_Clov = {np.sum(obs['E_Clov'][0]) / vol},  Plaq = {np.sum(obs['Plaquette'][0]) / vol},  Q = {np.sum(obs['Q'][0]) / vol},  | --")
                g.message()
                ######################

                V = U.copy()
                ### Wilson flow ###
                for n in range(1, WF_evol+1): # WF evolves V, which is a copy of U 
                    g.message(f"       -- WF evolution with eps = {eps_WF} --")
                    V = g.qcd.gauge.smear.wilson_flow(V, epsilon=eps_WF)
                    if (n % WF_step == 0):
                        g.message()
                        ##### Measurement at given flow time: t = n * eps_WF #####
                        obs['E_Clov'][n // WF_step]    = np.array(g.slice(g.qcd.gauge.energy_density(V, field=True), 3)).real
                        obs['Plaquette'][n // WF_step] = plaquette_slice(V, 3)
                        obs['Q'][n // WF_step]         = np.array(g.slice(g.qcd.gauge.topological_charge(V, field=True), 3)).real
                        g.message(f"        -- |  tWF = {n * eps_WF:.2f},  :  E_Clov = {np.sum(obs['E_Clov'][n // WF_step]) / vol},  Plaq = {np.sum(obs['Plaquette'][n // WF_step]) / vol},  Q = {np.sum(obs['Q'][n // WF_step]) / vol},  | --")
                        g.message()
                        ###################### 
                timer()
                g.message(timer)
                g.message()

                g.message()
                if g.rank() == 0:
                    # file.write(f"Configuration {run * Nconf + i // MC_step}", obs) # TYPO: assumes start from run=0, but first run of meas. is run=1. Don't care about label. 
                    file.write(f"Configuration {(run-1) * Nconf + i // MC_step}", obs)
                g.message()

        timer()
        g.message()
        g.message(f"Data production total times  :")
        g.message(timer)
        g.message()

        g.message()
        if g.rank() == 0:
            file.write(f'MC History run-{run}', history)
            if params["close_data_file"]:
                file.close()
            else:
                file.close(keep=True)
        g.message()

        g.message()
        g.message("Saving last configuration...")
        g.message()
        g.save(f"Cnfgs_checkpts/ckpoint_beta{int(beta*1e2)}_lat{L1}x{T}.{run}", U, g.format.nersc())
        g.message()
        g.message()
        if g.rank() == 0:
            os.remove(f"{params['init_cnfg']}")
            g.message(f"Previous configuration '{params['init_cnfg']}' has been deleted.")
        g.message()
##############################################################################



