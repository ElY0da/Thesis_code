import numpy as np # importing numpy module

#larval growth function runs larval growth function and pupal mortality
# lfood = larval food amout, m_index, f_index = array of numbered males and females respectively for indexing
# m_traits, f_traits = dictionary containing trait values of all males and females repectively for each trait   
def larval_growth(lfood, m_index, f_index, m_traits, f_traits):
    nf = len(f_index)   #number of females in one vial
    nm = len(m_index)   #number of males in one vials
    #defining dictionaries containing variables for male and female
    # size = body size, dt = time to reach critical size, frt = feeding rate at critical size
    # w_acum = waste accumulated in the body during larval feeding
    # food_aq = food aquired by each larva during 'dt' time step
    # waste_prod = waste produced by each larva during 'dt' time step
    m_var = dict(size = np.repeat(3., nm), dt = np.zeros(nm), frt = np.zeros(nm), w_acum = np.zeros(nm), food_aq = np.zeros(nm), waste_prod = np.zeros(nm))
    f_var = dict(size = np.repeat(3., nf), dt = np.zeros(nf), frt = np.zeros(nf), w_acum = np.zeros(nf), food_aq = np.zeros(nf), waste_prod = np.zeros(nf))
    # sexually dimporhic trait assigned here:
    feff, meff = f_traits['eff']*0.011, m_traits['eff']*0.009
    fmc, mmc = f_traits['mc']*1.1, m_traits['mc']*.9
    dt = 30     # time step
    fQ = dict(fband = 1., dband = 1.)   # food quality of feeding and diffusion band
    waste = dict(fband = 0, dband = 0)  # waste buil-up in feeding and diffusion band
    total_waste = 0     # amount of waste in the total food
    kd = 0.002*dt   # proportion of waste diffusion
    
    # for loop for discrete larval growth
    for t in np.arange(0,105*60,dt):
        # if amount of waste exceeds food amount, larval growth stops
        if lfood <= total_waste:
            break
        # else iterate function changes variables and body size 
        fout = iterate(t, dt, nf, lfood, f_traits['fr'], f_traits['wtol'], feff, fmc, total_waste, f_index, fQ['fband'], f_var)
        mout = iterate(t, dt, nm, lfood, m_traits['fr'], m_traits['wtol'], meff, mmc, total_waste, m_index,fQ['fband'], m_var)  
        # at each time step, following variables are updated:
        food_aq = sum(f_var['food_aq']) + sum(m_var['food_aq'])
        waste_prod = sum(f_var['waste_prod']) + sum(m_var['waste_prod'])
        lfood -= (food_aq - waste_prod)
        # foodQ function updates food quality variable at each time step
        foodQ(kd, lfood, food_aq, waste_prod, fQ, waste)
        total_waste += waste_prod
    
    #survive function is for pupal mortality dependent on waste accumulated:    
    falive_out = survive(fout[1], fmc, f_var)
    malive_out = survive(mout[1], mmc, m_var)
    
    # survivorship, body size, time to reach critical size, final feeding rate and index number of all surviving individuals are stored in the following arrays: 
    survivors = (nf-sum(falive_out[0]) + nm-sum(malive_out[0]))*.98/(nf+nm)
    alive_size = dict(female = falive_out[1], male = malive_out[1])
    alive_devt = dict(female = falive_out[2], male = malive_out[2])
    alive_index = dict(female = falive_out[3], male = malive_out[3])
    alive_frt = dict(female = falive_out[4], male = malive_out[4])
    
    #output the above arrays
    return alive_size, alive_devt, alive_frt, survivors, alive_index

# iterate function runs at each time step till time 't' 
def iterate(t, dt, n, lfood, frc, wtol, eff, mc, total_waste, larvae, fQ, var):
    np.random.shuffle(larvae)   # shuffle larvae for body size increment
    vf_pre, vf_post = 1, 1.5    # volume of food eaten in one time step (pre and post critical)
    u = 10**4   #scaling parameter
    food_aq = var['food_aq']    
    waste_prod = var['waste_prod']
    
    # for loop for each larva
    for i in larvae:
        if lfood > total_waste and fQ > 0 : # availability of edible food
            if var['size'][i] < mc[i]:      # pre-critical size condition
                if var['size'][i] != -1 :   # condition for not dead
                    var['frt'][i] = frc[i] + 0.017 * t  # increase in the feeding rate at time 't'
                    vfood = vf_pre * dt     # volume of food eaten in 'dt' time step
                    incr = var['frt'][i] * vfood * fQ * eff[i]  # increase in the body size
                    if var['w_acum'][i] > u * wtol[i]:  # condtion for waste tolerance
                        var['size'][i] = -1             # if not tolerant then dead
                    else:
                        var['size'][i] += incr          # else body size increased
                        var['w_acum'][i] +=  var['frt'][i] * vfood * (1 - fQ) # waste accumulted
                        waste_prod[i] = var['frt'][i] * vfood * fQ * (1 - eff[i]) # waste produced
                        food_aq[i] = var['frt'][i] * vfood  # food consumed
                        var['dt'][i] += dt
            else:   # post-critical condtions
                if t < var['dt'][i] + 48*60: # larval feeding only till 48*60 after critical size is reached
                    vfood = vf_post * dt
                    incr = var['frt'][i] * vfood * fQ * eff[i]
                    if var['w_acum'][i] <= u * wtol[i]:
                        var['size'][i] += incr
                        var['w_acum'][i] +=  var['frt'][i] * vfood * (1 - fQ)
                        waste_prod[i] = var['frt'][i] * vfood * fQ * (1 - eff[i])
                        food_aq[i] = var['frt'][i] * vfood
                else:
                    break
        elif var['size'][i] < mc[i]: # death if critical size is not reached before food becomes unedible
            var['size'][i] = -1
    
    return var, larvae

# food quality change conditions are given in the following function:
def foodQ(kd, lfood, food_aq, waste_prod, fQ, waste):
    fband = 50*4*(35.5*10**5)/8
    dband = fband
    if lfood > fband+dband:
        waste['fband'] = (waste['fband']+waste_prod)*(1-kd) + food_aq*(1-fQ['dband'])
        waste['dband'] += kd*(waste['fband']+waste_prod) - food_aq*(1-fQ['dband'])
    elif lfood > fband:
        waste['fband'] = (waste['fband']+waste_prod) + food_aq*(1-fQ['dband'])
        waste['dband'] -= food_aq*(1-fQ['dband'])
    else:
        fband = lfood
        waste['dband'] = 0
        waste['fband'] += waste_prod
    fQ['fband'] = 1 - waste['fband']/fband
    fQ['dband'] = 1 - waste['dband']/dband

    return fQ, waste

# pupal mortality condtions:
def survive(index, mc, var):
    u = 0.000007 #scaling parameter
    ldead = np.zeros(2)
    alive_size = []
    alive_devt = []
    alive_index = []
    alive_fr = []
    for i in index:
        if var['size'][i] < mc[i]:
            ldead[0] += 1
        elif np.random.uniform() < 1-np.exp(-(u*var['w_acum'][i])**2):
            ldead[1] += 1
        else:
            alive_size += [var['size'][i]]
            alive_devt += [var['dt'][i]/60]
            alive_index += [i]
            alive_fr += [var['frt'][i]]
    return ldead, alive_size, alive_devt, alive_index, alive_fr