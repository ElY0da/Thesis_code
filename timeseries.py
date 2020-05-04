# importing numpy and larval stage module
import numpy as np 
import larva_index_base as lfun
import multiprocessing as multi # stable in linux (otherwise run larval stage module in for loop)

# timeseries function runs evolution of larval trait parameters.
# gen = number of generations, eggs = number of eggs, f = food level multiplier, rep = replicates
# par_out = output dictionary for trait parameters
# lhpar = dictionary for initial mean values of each trait
# lhstd = dictionary for initial standard deviation values of each trait
# hrt = dictionary containing values of standard deviation in mpv of each trait
def timeseries(gen, eggs, f, rep, par_out, lhpar, lhstd, hrt):
    nut = 1.49          # adult nutrition  
    pop_size = 2000     #adult population size
    # assigning number of vials based on number of eggs
    if eggs == 60:      
        vials = 40
    if eggs == 600:
        vials = 24
    if eggs == 1200:
        vials = 12
    lfood  = 50*f*(37*10**5)           # larval food
    xpar = dict(x5 = 15.0, x6 = 1.0)   # scaling parameters     
    nlarva = int(lhpar['ht'] * eggs)   # number of larvae (ht: hatchetability)
    nf = nlarva // 2    # number of female larvae
    nm = nlarva - nf# number of male larvae
    # assign indexes and trait distributions for females and males:
    index = dict(female = np.arange(nf*vials), male = np.arange(nm*vials))
    m_traits = dict(fr = np.random.normal(lhpar['fr'], lhstd['fr'], nm*vials), wtol = np.random.normal(lhpar['wtol'], lhstd['wtol'], nm*vials), eff = np.random.normal(lhpar['eff'], lhstd['eff'], nm*vials), mc = np.random.normal(lhpar['mc']*10**5, lhstd['mc'], nm*vials))
    f_traits = dict(fr = np.random.normal(lhpar['fr'], 1, nf*vials), wtol = np.random.normal(lhpar['wtol'], 1., nf*vials), eff = np.random.normal(lhpar['eff'], 1., nf*vials), mc = np.random.normal(lhpar['mc']*10**5, 7500., nf*vials))
    
    # start the timeseries:
    for t in range(gen):
        p = multi.Pool(8) # assiging number of cores for multi-proccessing
        # larval grwoth output for given number of vials:
        larval_out = p.starmap(lfun.larval_growth, [(lfood, index['male'][z*nm:(z+1)*nm], index['female'][z*nf:(z+1)*nf], m_traits, f_traits) for z in range(vials)])
        p.close()
        
        # index outpur of larval_out : 0 = size; 1 = dt; 2 = index; 3 = frt; 4 = surv
        fi_alive = np.concatenate([larval_out[z][4]['female'] for z in range(vials)])
        mi_alive = np.concatenate([larval_out[z][4]['male'] for z in range(vials)])
        
        # arranging the array for alive individuals:
        f_index, m_index = np.arange(len(fi_alive)), np.arange(len(mi_alive))
        np.random.shuffle(f_index), np.random.shuffle(m_index)
        
        fi_adults = np.array([fi_alive[i] for i in f_index][:pop_size//2])
        mi_adults = np.array([mi_alive[i] for i in m_index][:pop_size//2])
        
        lf_size = np.concatenate([larval_out[z][0]['female'] for z in range(vials)])
        f_size = np.array([lf_size[i] for i in f_index][:pop_size//2])
        
        # store output of all trait values:
        fr = [f_traits['fr'][i] for i in fi_adults]+[m_traits['fr'][i] for i in mi_adults]
        wtol = [f_traits['wtol'][i] for i in fi_adults]+[m_traits['wtol'][i] for i in mi_adults]
        eff = [f_traits['eff'][i] for i in fi_adults]+[m_traits['eff'][i] for i in mi_adults]
        mc = [f_traits['mc'][i] for i in fi_adults]+[m_traits['mc'][i] for i in mi_adults]
        par_out['fr'][t] = [t, np.average(fr), np.std(fr)/np.sqrt(pop_size), pop_size, rep]
        par_out['wtol'][t] = [t, np.average(wtol), np.std(wtol)/np.sqrt(pop_size), pop_size, rep]
        par_out['eff'][t] = [t, np.average(eff), np.std(eff)/np.sqrt(pop_size), pop_size, rep]
        par_out['mc'][t] = [t, np.average(mc), np.std(mc)/np.sqrt(pop_size), pop_size, rep]
        
        # fecundity function to obtain number of eggs per female:
        norm_fec = fecundity(nut, xpar, lhpar['sen_siz'], f_size, vials, eggs)
         
        # assign traits for each offspring using inertiance function:
        offs_traits = inherit(fi_adults, mi_adults, norm_fec, f_traits, m_traits)
        
        # sort offspring into males and females for next generation:
        f_traits = dict(fr = offs_traits['fr'][:nf*vials], wtol = offs_traits['wtol'][:nf*vials], eff = offs_traits['eff'][:nf*vials], mc = offs_traits['mc'][:nf*vials])
        m_traits = dict(fr = offs_traits['fr'][nf*vials:], wtol = offs_traits['wtol'][nf*vials:], eff = offs_traits['eff'][nf*vials:], mc = offs_traits['mc'][nf*vials:])
        
    p.join()   
    # output of timeseries:
    return par_out

#fecundity function gives number of offspring per mated female
def fecundity(nut, xpar, sen_siz, a_size, vials, eggs):
    tot_fec = nut * xpar['x5'] * np.log(xpar['x6'] + sen_siz * a_size)
    prob = [i/sum(tot_fec) for i in tot_fec]
    norm_fec = np.random.multinomial(vials*eggs,prob)
    return norm_fec

# inherit function to sort offspring traits:
def inherit(fi_adults, mi_adults, norm_fec, f_traits, m_traits):
    mother_offs = []
    noffs = sum(norm_fec) 
    offs_traits = dict(fr = np.zeros(noffs), wtol = np.zeros(noffs), eff = np.zeros(noffs), mc = np.zeros(noffs))
    for i in range(len(norm_fec)):
        for j in range(norm_fec[i]):
            mother_offs += [i]
    for i in range(len(mother_offs)):
        female = fi_adults[mother_offs[i]]
        male = np.random.choice(mi_adults, replace=True)
        offs_traits['fr'][i] = min(80, mpvalue(f_traits['fr'][female], m_traits['fr'][male], 0.01))
        offs_traits['wtol'][i] = min(90,mpvalue(f_traits['wtol'][female], m_traits['wtol'][male], 0.01))
        offs_traits['eff'][i] = min(80,mpvalue(f_traits['eff'][female], m_traits['eff'][male], 0.05))
        offs_traits['mc'][i] = max(1.35*10**5,mpvalue(f_traits['mc'][female], m_traits['mc'][male], 0.01))
    np.random.shuffle(offs_traits['fr'])
    np.random.shuffle(offs_traits['wtol'])
    np.random.shuffle(offs_traits['eff'])
    np.random.shuffle(offs_traits['mc'])
    return offs_traits

#mid-parent value (mpv) function:
def mpvalue(male, female, sd):
    mean = (male+female)/2
    return np.random.normal(mean,mean*sd)