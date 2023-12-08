import numpy as np


def find_chgpts1(Rcum, alert_lag = 15, p_thresh = 0.5) :
    #get alert when run lenght pdf is above theshhold (p_thresh)
    S = np.array([])

    for ii in range(alert_lag, len(Rcum))  :
        if Rcum[alert_lag ,ii] > p_thresh and Rcum[alert_lag ,ii - 1] < p_thresh:
            S = np.append(S, int(ii))

    S = S.astype(int) 
    
    return S

def find_chgpts2(R, n_confirm = 1, min_shift = 1, exclude_start = True) :
    #get alert from MAP run lenght p
    run_len = np.array([ R[:k+1,k].argmax() for k in range(len(R)) ])
    shift_r = np.insert(run_len[:-1],0,run_len[0])
    idx_chg = np.where(abs(run_len - shift_r) > min_shift)[0]
    idx_chg = idx_chg[(idx_chg + n_confirm) < len(run_len)]  #filter those that cannot be confirm at the end
    S= []
    
    for i in idx_chg :

        curr_run_len = run_len[i]
        exp_run_len = range(curr_run_len, curr_run_len + n_confirm)

        if (run_len[i:i+n_confirm] == exp_run_len).all() :
            S.append(i - curr_run_len)

    S = np.unique( S)
    
    if  exclude_start :
         S  = S[ S !=0]
    
    return  S


def find_chgpts3(R, exclude_start = True) :

    map_t = np.zeros(len(R) + 1)
    map_t[0:2] = 1
    S = set([])
    
    for i in range(1, len(R)) :
        R_t = R[1:i+1,i]
        test = R_t * np.flip(map_t[:i])
        max_r_t = test.argmax() 
        map_t[i] = test[max_r_t] 
        S.add(i - max_r_t - 1)
        
    S = np.sort(np.array(list(S)))
    if  exclude_start :
        S  = S[S !=0]

    return S