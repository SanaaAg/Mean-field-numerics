#import libraries

import scipy
from scipy.integrate import solve_ivp
import numpy as np
import cmath as cm
import h5py
from numpy.linalg import multi_dot
from scipy.linalg import logm
from scipy.special import factorial
from scipy.special import *
from scipy.sparse import csr_matrix
from numpy.linalg import eig
from scipy.linalg import eig as sceig
import math
import time
#from math import comb
from sympy.physics.quantum.cg import CG
from sympy import S
import collections
import numpy.polynomial.polynomial as poly

import sys
argv=sys.argv

if len(argv) < 2:
    #Default
    run_id=1
else:
    try:
        run_id = int(argv[1])
        Natoms = int(argv[2])
        #det_val_input = float(argv[4])

    except:
        print ("Input error")
        run_id=1

# some definitions (do not change!)

fe = 0
fg = 0

fixed_param = 0 # 0: L = 20, R = 0.5; 1: mean density; 2: optical depth along L.


# some definitions (do not change!)

e0 = np.array([0, 0, 1])
ex = np.array([1, 0, 0])
ey = np.array([0, 1, 0])
eplus = -(ex + 1j*ey)/np.sqrt(2)
eminus = (ex - 1j*ey)/np.sqrt(2)
single_decay = 1.0 # single atom decay rate

direc = '/data/rey/saag4275/data_files/'   # directory for saving data

# parameter setting box (may change)


realization_list = np.arange(11,21,1) #np.arange(1,11,1)
rabi_val_list = np.array([1.5,1.7,2.0,2.2,2.3,2.4,2.5,2.6,2.7,3.0])  #np.array([0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0])  #np.array([3.5,4.5,5.0,7.0,9.0,10.0,11.0,12.0,13.0,14.0]) # np.array([1.7,2.2,2.3,2.4,2.6,2.7])  #np.array([20,25,50,100,200,500,1000,2000,4000,8000]) #np.array([3.5,4.5,5.0,7.0,9.0,10.0,11.0,12.0,13.0,14.0]) #np.array([0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0])
tfin_list = np.array([8]*int(len(rabi_val_list)))  #np.array([20]*2 + [8]*int(len(rabi_val_list)-2))
real_id_list = np.arange(0, len(realization_list), 1)
rabi_id_list = np.arange(0, len(rabi_val_list), 1)

# generate 2D array of realisation x rabi --> 

param_grid_rabi, param_grid_real = np.meshgrid(rabi_id_list, real_id_list, indexing='ij')
param_grid_real_list = param_grid_real.flatten()
param_grid_rabi_list = param_grid_rabi.flatten()

real_id = param_grid_real_list[run_id-1]
rabi_id = param_grid_rabi_list[run_id-1]

real_val = realization_list[real_id]
rabi_val = rabi_val_list[rabi_id] #rabi_val_list[run_id-1] 
t_final_input = tfin_list[rabi_id] 

print("Rabi = "+str(rabi_val), flush=True)
print("real id = "+str(real_val), flush=True)

eL = np.array([0, 0, 1]) # polarisation of laser, can be expressed in terms of the vectors defined above
detuning_list = np.array([0.0*single_decay]) # detuning of laser from transition
del_ze = 0.0 # magnetic field, i.e., Zeeman splitting of excited state manifold
del_zg = 0.0 # magnetic field, i.e., Zeeman splitting of ground state manifold
rabi = rabi_val*single_decay

#interactions turned off
turn_off_list = ['incoherent','coherent']
turn_off = [] #[turn_off_list[0], turn_off_list[1]] # leave turn_off = [], if nothing is to be turned off


turn_off_txt = ''
if turn_off != []:
    turn_off_txt += '_no_int_'
    for item in turn_off:
        turn_off_txt += '_'+ item

add_txt_in_params = turn_off_txt

num_pts_dr = int(2*1e2)

t_initial_dr = 0.0
t_final_dr = t_final_input 
t_range_dr = [t_initial_dr, t_final_dr]
t_vals_dr = np.linspace(t_initial_dr, t_final_dr, num_pts_dr) 


e0_desired = eL


# more definitions and functions (do not change!)

wavelength = 1 # wavelength of incident laser
k0 = 2*np.pi/wavelength
kvec = k0*np.array([1, 0, 0]) # k vector of incident laser

    
def rotation_matrix_a_to_b(va, vb): #only works upto 1e15-ish precision
    ua = va/np.linalg.norm(va)
    ub = vb/np.linalg.norm(vb)
    if np.dot(ua, ub) == 1:
        return np.identity(3)
    elif np.dot(ua, ub) == -1: #changing z->-z changes y->-y, thus preserving x->x, which is the array direction (doesn't really matter though!)
        return -np.identity(3)
    uv = np.cross(ua,ub)
    c = np.dot(ua,ub)
    v_mat = np.zeros((3,3))
    ux = np.array([1,0,0])
    uy = np.array([0,1,0])
    uz = np.array([0,0,1])
    v_mat[:,0] = np.cross(uv, ux)
    v_mat[:,1] = np.cross(uv, uy)
    v_mat[:,2] = np.cross(uv, uz)
    matrix = np.identity(3) + v_mat + (v_mat@v_mat)*1.0/(1.0+c)
    return matrix

 
if np.abs(np.conj(e0)@e0_desired) < 1.0:
    rmat = rotation_matrix_a_to_b(e0,e0_desired)
    eplus = rmat@eplus
    eminus = rmat@eminus
    ex = rmat@ex
    ey = rmat@ey
    e0 = e0_desired

print('kL = '+str(kvec/np.linalg.norm(kvec)), flush=True)
print('e0 = '+str(e0), flush=True)
print('ex = '+str(ex), flush=True)
print('ey = '+str(ey), flush=True)

HSsize = int(2*fg + 1 + 2*fe + 1) # Hilbert space size of each atom
HSsize_tot = int(HSsize**Natoms) # size of total Hilbert space

adde = fe
addg = fg

# polarisation basis vectors
evec = {0: e0, 1:eplus, -1: eminus}
evec = collections.defaultdict(lambda : [0,0,0], evec) 
   
def sort_lists_simultaneously_cols(a, b): #a -list to be sorted, b - 2d array whose columns are to be sorted according to indices of a
    inds = a.argsort()
    sortedb = b[:,inds]
    return sortedb

# levels
deg_e = int(2*fe + 1)
deg_g = int(2*fg + 1)

if (deg_e == 1 and deg_g == 1):
    qmax = 0
else:
    qmax = 1


# dictionaries




# Clebsch Gordan coeff
cnq = {}
arrcnq = np.zeros((deg_g, 2*qmax+1), complex)
if (deg_e == 1 and deg_g ==1):
    cnq[0, 0] = 1
    arrcnq[0, 0] =  1
else:
    for i in range(0, deg_g):
        mg = i-fg
        for q in range(-qmax, qmax+1):
            if np.abs(mg + q) <= fe:
                cnq[mg, q] =  np.float(CG(S(fg), S(mg), S(qmax), S(q), S(fe), S(mg+q)).doit())
                arrcnq[i, q+qmax] = cnq[mg, q]
cnq = collections.defaultdict(lambda : 0, cnq) 

# Dipole moment

dsph = {}
if (deg_e == 1 and deg_g ==1):
    dsph[0, 0] = np.conjugate(evec[0])
else:
    for i in range(0, deg_e):
        me = i-fe
        for j in range(0, deg_g):
            mg = j-fg
            dsph[me, mg] = (np.conjugate(evec[me-mg])*cnq[mg, me-mg])

dsph = collections.defaultdict(lambda : np.array([0,0,0]), dsph) 

            
#omega_atom = collections.defaultdict(lambda : 0, omega_atom) 


# normalise vector
def hat_op(v):
    return (v/np.linalg.norm(v))

transition_wavelength = 1.0
R_perp_given = 0.5*transition_wavelength # radial std in units of lambda, experimental value for system
L_given = 20.0*transition_wavelength # axial std in units of lambda, experimental value for system
aspect_ratio = L_given/R_perp_given # = axial/radial std
N_given = 2000 # experimental value for system
k = 2*np.pi/transition_wavelength
OD_x_given = 3*N_given/(2*(k*R_perp_given)**2)
Volume_cloud_given = 2*np.pi*(R_perp_given**2)*L_given
mean_density_given = N_given/Volume_cloud_given

def f_cloud_dims_fixed_OD(N_output):
    # Since OD_x = 3*N_given/(2*(k*R_perp_given)**2)
    R_perp_output = R_perp_given*np.sqrt(N_output/N_given*1.0) # = np.sqrt(3*N_output/(2*OD_x*(k**2)))
    L_axial_output = R_perp_output*aspect_ratio

    return [R_perp_output, L_axial_output]


def f_cloud_dims_fixed_mean_density(N_output):
    Volume_cloud_output = N_output/mean_density_given
    R_perp_output = (Volume_cloud_output/(2*np.pi*aspect_ratio))**(1/3.0)
    L_axial_output = R_perp_output*aspect_ratio

    return [R_perp_output, L_axial_output]

if fixed_param == 0:
    std_list = [R_perp_given, L_given]
    fixed_text = '_fixed_size'
elif fixed_param == 1:
    std_list = f_cloud_dims_fixed_mean_density(Natoms)
    fixed_text = '_fixed_mean_density'
elif fixed_param == 2:
    std_list = f_cloud_dims_fixed_OD(Natoms)
    fixed_text = '_fixed_OD_ax'
    
std_rad = std_list[0]
std_ax = std_list[1]
dims_text = '_std_rad_'+str(np.round(std_rad,2)).replace('.',',') + '_std_ax_'+str(np.round(std_ax,2)).replace('.',',')


# plot properties

levels = int(deg_e + deg_g)
rdir_fig = '_3D_gas'+fixed_text+dims_text

eLdir_fig = '_eL_along_'
dirs = ['x','y','z']
temp_add = 0
for i in range(0,3):
    if eL[i]!=0:
        if temp_add == 0:
            eLdir_fig += dirs[i]
        else:
            eLdir_fig += '_and_'+ dirs[i]
        temp_add += 1

kdir_fig = '_k_along_'
dirs = ['x','y','z']
temp_add = 0
for i in range(0,3):
    if kvec[i]!=0:
        if temp_add == 0:
            kdir_fig += dirs[i]
        else:
            kdir_fig += '_and_'+ dirs[i]
        temp_add += 1

rabi_add = '_rabi_'+str((rabi)).replace('.',',')


h5_title = str(levels)+'_level_'+str(Natoms)+'_atoms'+rdir_fig+kdir_fig+eLdir_fig+'_real_id_'+str(int(real_val))+'.h5'

h5_title_dr = str(levels)+'_level_'+str(Natoms)+'_atoms'+rdir_fig+kdir_fig+eLdir_fig+rabi_add+'_tfin_'+str(int(t_final_input))+'_real_id_'+str(int(real_val))+add_txt_in_params+'.h5'


# try to load data for positions, if no data available, then generate samples

try:
    hf = h5py.File(direc+'Atomic_positions_Greens_fn_phases_3D_gas_'+h5_title, 'r')

    rvecall = hf['rvecall'][()]
    arrGij = hf['arrGij'][()]
    arrGijtilde = hf['arrGijtilde'][()]
    arrIij = hf['arrIij'][()]
    phase_array = hf['forward_phase_array'][()]
    hf.close()
    
    print("Data for positions loaded from file!", flush=True)    

except:

    print("No data for positions found, will generate sampling of positions now and save it!", flush=True)

    temp = np.random.default_rng(real_val)

    r_sampled_raw = temp.normal(0, std_ax, Natoms*1000) # get axial x position
    r_sampled = temp.choice(r_sampled_raw, Natoms, replace=False) # to prevent multiple atoms from being at exactly the same positions
    r_array_x = np.sort(r_sampled)

    r_sampled_rad_z = temp.normal(0, std_rad, Natoms)  # get radial z position
    r_array_z = r_sampled_rad_z 

    r_sampled_rad_y = temp.normal(0, std_rad, Natoms)  # get radial y position
    r_array_y = r_sampled_rad_y 

    r_array_xyz = np.array([r_array_x, r_array_y, r_array_z])
    rvecall = r_array_xyz.T

    r_nn_spacing = np.sqrt(np.einsum('ab->b', (r_array_xyz[:,1:] - r_array_xyz[:,:-1])**2))

    r_nn_spacing_avg = np.mean(r_nn_spacing) # 3d avg spacing for simulation
    r_min = np.min(r_nn_spacing) # 3d min spacing for simulation

    print('3D gas properties:' , flush=True)
    print('std axial = ' + str(std_ax))
    print('std radial = ' + str(std_rad), flush=True)
    print('mean nearest-neighbor spacing = ' + str(r_nn_spacing_avg), flush=True)
    print('minimum nearest-neighbor spacing = ' + str(r_min), flush=True)
    
    
    # phase_array
    
    phase_array = np.zeros((Natoms, Natoms), complex)
    for i in range(0, Natoms):
        for j in range(0, Natoms):
            temp_phase = np.dot(kvec, (rvecall[i] - rvecall[j]))
            phase_array[i, j] = np.exp(1j*temp_phase)

    # Green's function
    def funcG(r):
        tempcoef = 3*single_decay/4.0
        temp1 = (np.identity(3) - np.outer(hat_op(r), hat_op(r)))*np.exp(1j*k0*np.linalg.norm(r))/(k0*np.linalg.norm(r)) 
        temp2 = (np.identity(3) - 3*np.outer(hat_op(r), hat_op(r)))*((1j*np.exp(1j*k0*np.linalg.norm(r))/(k0*np.linalg.norm(r))**2) - np.exp(1j*k0*np.linalg.norm(r))/(k0*np.linalg.norm(r))**3)
        return (tempcoef*(temp1 + temp2))

    def funcGij(i, j):
        return (funcG(rvecall[i] - rvecall[j]))

    fac_inc = 1.0
    fac_coh = 1.0
    if turn_off!=[]:
        for item in turn_off:
            if item == 'incoherent':
                fac_inc = 0
            if item == 'coherent':
                fac_coh = 0


    taD = time.time()

    dictRij = {}
    dictIij = {}
    dictGij = {}
    dictGijtilde = {}

    for i in range(0, Natoms):
        for j in range(0, Natoms):
            for q1 in range(-qmax,qmax+1):
                for q2 in range(-qmax,qmax+1):
                    if i!=j:
                        tempRij = fac_coh*np.conjugate(evec[q1])@np.real(funcGij(i, j))@evec[q2]
                        tempIij = fac_inc*np.conjugate(evec[q1])@np.imag(funcGij(i, j))@evec[q2]

                    else:
                        tempRij = 0
                        tempIij = (single_decay/2.0)*np.dot(np.conjugate(evec[q1]),evec[q2])
                    dictRij[i, j, q1, q2] = tempRij
                    dictIij[i, j, q1, q2] = tempIij
                    dictGij[i, j, q1, q2] = tempRij + 1j*tempIij
                    dictGijtilde[i, j, q1, q2] = tempRij - 1j*tempIij
                    #arrGij[i, j, q1+qmax, q2+qmax] = tempRij + 1j*tempIij
                    #arrGijtilde[i, j, q1+qmax, q2+qmax] = tempRij - 1j*tempIij

    dictRij = collections.defaultdict(lambda : 0, dictRij) 
    dictIij = collections.defaultdict(lambda : 0, dictIij) 
    dictGij = collections.defaultdict(lambda : 0, dictGij) 
    dictGijtilde = collections.defaultdict(lambda : 0, dictGijtilde) 

    tbD = time.time()
    print("time to assign Rij, Iij dict: "+str(tbD-taD), flush=True)

    taG = time.time()

    arrGij = np.zeros((Natoms, Natoms, deg_e, deg_g, deg_e, deg_g), complex)
    arrGijtilde = np.zeros((Natoms, Natoms, deg_e, deg_g, deg_e, deg_g), complex)
    arrIij = np.zeros((Natoms, Natoms, deg_e, deg_g, deg_e, deg_g), complex)
    for i in range(0, Natoms):
        for j in range(0, Natoms):
            for ima in range(0, deg_e):
                ma = ima - fe
                for ina in range(0, deg_g):
                    na = ina - fg
                    for imb in range(0, deg_e):
                        mb = imb - fe
                        for inb in range(0, deg_g):
                            nb = inb - fg
                            arrGij[i, j, ima, ina, imb, inb] = dictGij[i, j, ma-na, mb-nb]*cnq[na, ma-na]*cnq[nb, mb-nb]
                            arrGijtilde[i, j, ima, ina, imb, inb] = dictGijtilde[i, j, ma-na, mb-nb]*cnq[na, ma-na]*cnq[nb, mb-nb]
                            arrIij[i, j, ima, ina, imb, inb] = dictIij[i, j, ma-na, mb-nb]*cnq[na, ma-na]*cnq[nb, mb-nb]

    tbG = time.time()
    print("time to assign Gij matrix: "+str(tbG-taG), flush=True)



    # save position and Gij data for future reference
    
    if fac_coh == 1.0 and fac_inc == 1.0:
        hf = h5py.File(direc+'Atomic_positions_Greens_fn_phases_3D_gas_'+h5_title, 'w')
        hf.create_dataset('rvecall', data=rvecall, compression="gzip", compression_opts=9)
        hf.create_dataset('arrGij', data=arrGij, compression="gzip", compression_opts=9)
        hf.create_dataset('arrGijtilde', data=arrGijtilde, compression="gzip", compression_opts=9)
        hf.create_dataset('arrIij', data=arrIij, compression="gzip", compression_opts=9)
        hf.create_dataset('forward_phase_array', data=phase_array, compression="gzip", compression_opts=9)
        hf.close()
        
    else:
        hf = h5py.File(direc+'Atomic_positions_Greens_fn_phases_3D_gas_'+h5_title_turn_off, 'w')
        hf.create_dataset('rvecall', data=rvecall, compression="gzip", compression_opts=9)
        hf.create_dataset('arrGij', data=arrGij, compression="gzip", compression_opts=9)
        hf.create_dataset('arrGijtilde', data=arrGijtilde, compression="gzip", compression_opts=9)
        hf.create_dataset('arrIij', data=arrIij, compression="gzip", compression_opts=9)
        hf.create_dataset('forward_phase_array', data=phase_array, compression="gzip", compression_opts=9)
        hf.close()
    
    if fac_coh != 1.0 and fac_inc == 1.0:
            print("Coherent interactions turned off!", flush=True)
    elif fac_coh == 1.0 and fac_inc != 1.0:
            print("Incoherent interactions turned off!", flush=True)
    elif fac_coh != 1.0 and fac_inc != 1.0:
            print("Coherent AND incoherent interactions turned off!", flush=True)





#Rabi frequency for each atom

omega_atom = np.zeros((Natoms, deg_e, deg_g), complex)
for n in range(0, Natoms):
    for i in range(0, deg_e):
        me = i-fe
        for j in range(0, deg_g):
            mg = j-fg
            omega_atom[n, i, j] = (rabi*np.dot(dsph[me, mg],eL)*np.exp(1j*np.dot(kvec, rvecall[n]))) 


indgg = int(deg_g*deg_g)
indee = int(deg_e*deg_e)
indeg = int(deg_e*deg_g)
total_num = indgg+indee+indeg
# gs states of each atom

gs_states = np.zeros(deg_g)
for i in range(0, deg_g):
    gs_states[i] = i-fg
    
# es states of each atom

es_states = np.zeros(deg_e)
for i in range(0, deg_e):
    es_states[i] = i-fe

# single atom operators' index in the list of all ops for all atoms

dict_ops = {}
index = 0
for n in range(0, Natoms):
    #for in1 in range(0, deg_g):
    #    n1 = in1-fg
    #    for in2 in range(0, deg_g):
    #        n2 = in2-fg
    #        dict_ops['gg', n, n1, n2] = index
    #        index += 1
    for im1 in range(0, deg_e):
        m1 = im1-fe
        for im2 in range(0, deg_e):
            m2 = im2-fe
            dict_ops['ee', n, m1, m2] = index
            index += 1
    for im1 in range(0, deg_e):
        m1 = im1-fe
        for in2 in range(0, deg_g):
            n2 = in2-fg
            dict_ops['eg', n, m1, n2] = index
            index += 1
        
dict_ops = collections.defaultdict(lambda : 'None', dict_ops)


def f_trace(sig_list):
    trace = 0+0*1j
    for n in range(0, Natoms):
        for ina in range(0, deg_g):
            na = ina-fg
            trace += sig_list[dict_ops['gg', n, na, na]]
        for ima in range(0, deg_e):
            ma = ima-fe
            trace += sig_list[dict_ops['ee', n, ma, ma]]
    return trace/Natoms



#EOM for one point functions

def f_sig_ee_dot(sig_ee, sig_eg, drive): #n = 0, 1, 2, ... = atom no., drive = 0, 1
    #free indices n, a, b
    tempsum = 1j*del_ze*(np.einsum('a,nab->nab', es_states[:], sig_ee[:, :, :] )-np.einsum('b,nab->nab', es_states[:], sig_ee[:, :, :] ))
    tempsum += drive*1j*(np.einsum('nas,nbs->nab', sig_eg[:, :, :], omega_atom[:, :, :] ) - np.einsum('nas,nbs->nab', np.conjugate(omega_atom[:, :, :]),np.conjugate(sig_eg[:, :, :]) ))
    tempsum += 1j*np.einsum('naq,nnbpqp->nab', sig_ee[:, :, :], arrGij[:,:,:,:,:,:] )
    tempsum += -1j*np.einsum('nnqpap,nqb->nab', arrGijtilde[:,:,:,:,:,:],sig_ee[:, :, :] )     
    tempsum += (1j*np.einsum('nap,njbpqr,jqr->nab', sig_eg[:, :, :], arrGij[:,:,:,:,:,:], np.conjugate(sig_eg[:, :, :]) ) - 1j*np.einsum('nap,nnbpqr,nqr->nab', sig_eg[:, :, :], arrGij[:,:,:,:,:,:], np.conjugate(sig_eg[:, :, :]) ))
    tempsum += (-1j*np.einsum('jnqrap,jqr,nbp->nab', arrGijtilde[:,:,:,:,:,:],sig_eg[:, :, :],np.conjugate(sig_eg[:, :, :]) ) +1j*np.einsum('nnqrap,nqr,nbp->nab', arrGijtilde[:,:,:,:,:,:],sig_eg[:, :, :],np.conjugate(sig_eg[:, :, :]) ))
    return tempsum

def f_sig_eg_dot(sig_ee, sig_eg, detuning, drive): #n = 0, 1, 2, ... = atom no.
    sig_gg = 1 - sig_ee
    tempsum = 1j*(detuning*sig_eg[:, :, :]+del_ze*np.einsum('a,nab->nab', es_states[:], sig_eg[:, :, :] )-del_zg*np.einsum('b,nab->nab', gs_states[:], sig_eg[:, :, :] ))
    tempsum += drive*1j*(np.einsum('nas,nsb->nab',sig_ee[:, :, :], np.conjugate(omega_atom[:, :, :]) )) 
    tempsum += -drive*1j*(np.einsum('nsb,nas->nab', sig_gg[:, :, :], np.conjugate(omega_atom[:, :, :]) ))
    tempsum += -1j*np.einsum('nnqpap,nqb->nab', arrGijtilde[:,:,:,:,:,:],sig_eg[:, :, :] ) 
    tempsum += (1j*np.einsum('jnrpqb,naq,jrp->nab', arrGijtilde[:,:,:,:,:,:],sig_ee[:, :, :],sig_eg[:, :, :] ) - 1j*np.einsum('nnrpqb,naq,nrp->nab', arrGijtilde[:,:,:,:,:,:],sig_ee[:, :, :],sig_eg[:, :, :] ))
    tempsum += (-1j*np.einsum('jnqrap,jqr,npb->nab', arrGijtilde[:,:,:,:,:,:],sig_eg[:, :, :],sig_gg[:, :,:] ) +1j*np.einsum('nnqrap,nqr,npb->nab', arrGijtilde[:,:,:,:,:,:],sig_eg[:, :, :],sig_gg[:, :,:] ))
    return tempsum

def f_sig_gg_dot(sig_gg, sig_ee, sig_eg, drive): #n = 0, 1, 2, ... = atom no.
    tempsum = 1j*del_zg*(np.einsum('a,nab->nab', gs_states[:], sig_gg[:, :, :] )-np.einsum('b,nab->nab', gs_states[:], sig_gg[:, :, :] ))
    tempsum += drive*1j*(-np.einsum('nsa,nsb->nab', omega_atom[:, :, :],sig_eg[:, :, :] ) + np.einsum('nsb,nsa->nab', np.conjugate(omega_atom[:, :, :]),np.conjugate(sig_eg[:, :, :]) ))
    tempsum += 2*np.einsum('nnqarb,nqr->nab', arrIij[:,:,:,:,:,:],sig_ee[:, :, :] )
    tempsum += (1j*np.einsum('jnrpqb,nqa,jrp->nab', arrGijtilde[:,:,:,:,:,:],np.conjugate(sig_eg[:, :, :]),sig_eg[:, :, :] ) - 1j*np.einsum('nnrpqb,nqa,nrp->nab', arrGijtilde[:,:,:,:,:,:],np.conjugate(sig_eg[:, :, :]),sig_eg[:, :, :] ))
    tempsum += (-1j*np.einsum('njqarp,nqb,jrp->nab', arrGij[:,:,:,:,:,:],sig_eg[:, :, :],np.conjugate(sig_eg[:, :,:]) ) +1j*np.einsum('nnqarp,nqb,nrp->nab', arrGij[:,:,:,:,:,:],sig_eg[:, :, :],np.conjugate(sig_eg[:, :,:]) ))
    return tempsum


def f_sig_dot_vec(t, sig_list, drive, detuning):
    sig_mat = np.reshape(sig_list, (Natoms, 2))
    #sig_gg = np.reshape(sig_mat[:,0:indgg], (Natoms, deg_g, deg_g))
    sig_ee = np.reshape(sig_mat[:, 0:indee], (Natoms, deg_e, deg_e))
    sig_eg = np.reshape(sig_mat[:, indee:],(Natoms, deg_e, deg_g))
    #sig_gg_dot = f_sig_gg_dot(sig_gg, sig_ee, sig_eg, drive) 
    sig_ee_dot = f_sig_ee_dot(sig_ee, sig_eg, drive)
    sig_eg_dot = f_sig_eg_dot(sig_ee, sig_eg, detuning, drive)
    sig_dot_mat = np.zeros((Natoms, 2),complex)
    #sig_dot_mat[:,0:indgg] = np.reshape(sig_gg_dot, (Natoms, indgg))
    sig_dot_mat[:, :indee] = np.reshape(sig_ee_dot, (Natoms, indee))
    sig_dot_mat[:, indee:] = np.reshape(sig_eg_dot, (Natoms, indeg))
    return sig_dot_mat.flatten()


# initial condition

initial_sig_mat = np.zeros((Natoms, 2), complex)

#initialising system in which all atoms are in (mg = -fg) ground state
gs_index = 0
gs_mat = np.zeros((Natoms, deg_g, deg_g), complex)
gs_mat[:, gs_index, gs_index] = 1 + 0*1j
gs_mat2 = np.reshape(gs_mat, (Natoms, deg_g*deg_g))
es_mat = 1 - gs_mat2
initial_sig_mat[:, 0:deg_e*deg_e] = es_mat
initial_sig_vec = initial_sig_mat.flatten()

#print('trace = '+str(f_trace(initial_sig_vec)))
det_set = 0

#driven evolutiion from a ground state superposition to get to the steady state

ta1 = time.time()
sol_list_dr = []
for i_det in range(det_set,det_set+1):
    sol = solve_ivp(f_sig_dot_vec, t_range_dr, initial_sig_vec, method='RK45', t_eval=t_vals_dr, dense_output=False, events=None, atol = 10**(-5), rtol = 10**(-4), args=[1, -detuning_list[i_det]])
    sol_list_dr.append(sol)
tb1 = time.time()
runtime1 = tb1-ta1

print("Runtime for time evolution: " + str(runtime1), flush=True)

num_single_particle_ops = int(Natoms*(deg_e**2 + deg_e*deg_g)) # sig_ee, sig_eg

total_exc_dr = np.zeros(len(t_vals_dr))
total_Sx_dr = np.zeros(len(t_vals_dr))
single_Sp_dr = np.zeros((Natoms, len(t_vals_dr)), complex)
total_Sy_dr = np.zeros(len(t_vals_dr))
forward_intensity_dr = np.zeros(len(t_vals_dr), complex)
sig_list_dr = np.zeros((len(t_vals_dr), num_single_particle_ops), complex)

for t in range(0, len(t_vals_dr)):
    index = 0
    sig_list_tmp = sol_list_dr[0].y[:,t]
    for k in range(0, Natoms):
        total_exc_dr[t] += sig_list_tmp[dict_ops['ee',k,0,0]].real
        single_Sp_dr[k,t] = sig_list_tmp[dict_ops['eg',k,0,0]]
        sig_list_dr[t, index] = sig_list_tmp[dict_ops['ee',k,0,0]]
        index += 1
        sig_list_dr[t, index] = sig_list_tmp[dict_ops['eg',k,0,0]]
        index += 1
total_Sx_dr = np.einsum('kt->t', (single_Sp_dr + np.conj(single_Sp_dr)).real)
total_Sy_dr = np.einsum('kt->t', (-1j*(single_Sp_dr - np.conj(single_Sp_dr))).real)
forward_intensity_dr = total_exc_dr + np.einsum('kj,kt,jt->t', phase_array, single_Sp_dr, np.conj(single_Sp_dr))  - np.einsum('kk,kt,kt->t', phase_array, single_Sp_dr, np.conj(single_Sp_dr))

# save data

save_exc = np.zeros((len(t_vals_dr), 2))
save_exc[:, 0] = t_vals_dr
save_exc[:, 1] = total_exc_dr
save_Sx = np.zeros((len(t_vals_dr), 2))
save_Sx[:, 0] = t_vals_dr
save_Sx[:, 1] = total_Sx_dr
save_Sy = np.zeros((len(t_vals_dr), 2))
save_Sy[:, 0] = t_vals_dr
save_Sy[:, 1] = total_Sy_dr
save_int = np.zeros((len(t_vals_dr), 2))
save_int[:, 0] = t_vals_dr
save_int[:, 1] = forward_intensity_dr
save_single_ops = np.zeros((len(t_vals_dr), num_single_particle_ops+1), complex)
save_single_ops[:, 0] = t_vals_dr 
save_single_ops[:, 1:] = sig_list_dr
#save_steady_state_original_format = sol_list_dr[0].y[:,-1]

hf = h5py.File(direc+'Data_MFT_dynamics_to_equil_'+h5_title_dr, 'w')
hf.create_dataset('total_exc', data=save_exc, compression="gzip", compression_opts=9)
hf.create_dataset('total_Sx', data=save_Sx, compression="gzip", compression_opts=9)
hf.create_dataset('total_Sy', data=save_Sy, compression="gzip", compression_opts=9)
hf.create_dataset('forward_intensity_dr', data=save_int, compression="gzip", compression_opts=9)
hf.create_dataset('rvecall', data=rvecall, compression="gzip", compression_opts=9)
hf.create_dataset('single_particle_op_population', data=save_single_ops, compression="gzip", compression_opts=9)
#hf.create_dataset('steady_state_ops_original_format', data=save_steady_state_original_format, compression="gzip", compression_opts=9)
hf.close()

print("All runs done. May all your codes run this well! No decay run. :)", flush=True)
