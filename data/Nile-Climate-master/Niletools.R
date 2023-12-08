Siegel_Castellan <- function(v,years){
    # The Siegel Castellan method for determining a single change point in a 
    # timeseries using the Mann-Whitney Statistic
    N = length(v)
    SR = rep(NA, N)
    r = rank(v)
    for(i in 1:N){
        SR[[i]] = sum(r[1:i])
    }
    SA = abs(2*SR-(1:N)*N+1)
    N1 = which.max(SA)
    W = SR[[N1]]
    N2 = N-N1
    Wcrit = N1*(N+1)/2
    sw = sqrt(N1*N2*(N+1)/12)
    delta = .5 * sign(Wcrit-W)
    z = (W-Wcrit+delta)/sw
    p = pnorm(-abs(z))*2
    dv = median(v[N1:length(v)]) - median(v[1:N1])
    return(c(1, years[[N1]], p, dv))
}

normalizemedian <- function(v, cps){
    #given a series of data and a list of change points, normalizes each 
    #distinct segment of the record
    cps=sort(cps)
    segments=list()
    v1=c()
    for(i in 1:length(cps)){
        if(length(segments)==0){
            segments[[i]]=1:cps[[i]]
        } else{ 
            segments[[i]]=cps[[i-1]]:cps[[i]]
        }
    }
    segments[[i+1]] = cps[[i]]:length(v)
    for(seg in segments){
        v1[seg] = v[seg]-median(v[seg])
    }
    return(v1)
}
        
Lanzante <- function(v, years, cp = c(0), pcut = .05, buffer = 5){
    #Iterative version of Siegel_Castellan repeats untils cp is not sig
    #at p_cut, is in the first or last 10 pts.
    #If within 5 of previous CP, a secondary maximum is used
    cptemp = Siegel_Castellan(v,years)
    cpcheck=FALSE
    if(length(cp)>1){
        cpcheck=sum(abs(cptemp[[2]] - cp[(1:cp[[1]])*3-1]) < buffer)
    }
    if(cptemp[[3]] > pcut |
       which(years == cptemp[[2]]) < buffer |
       which(years == cptemp[[2]]) > (length(v)-buffer) |
       cpcheck){
       return(cp)
    }
    else{
        cp[(1:3)+ 1 + cp[[1]]*3] = cptemp[2:4]
        cp[[1]] = cp[[1]]+1
        cpindex = which(is.element(years, cp[3*(1:cp[[1]])-1]))
        v1=normalizemedian(v, cpindex)
        return(Lanzante(v1, years, cp, pcut, buffer))
    }
}

plotannual <- function(ann,years,overlapyrs,varname){
    #plots the annual time series with breaks given by overlapyrs and displays 
    #the normalized median deviation from the overall median of each sub segment 
    startyear = years[[1]]-1
    breaks = sort(c(overlapyrs,years[[1]],years[[length(years)]]))
    plot(years,ann,ylim=c(-2,4),type='l',lwd=2,ylab=expression(sigma),xlab='', main = varname)
    for(l in 2:length(breaks)){
        j = breaks[[l-1]]-startyear
        k = breaks[[l]]-startyear
        m = median(ann[j:k])
        lines(c(j,k)+startyear,c(m,m))
        annstring = as.character(format(round(m,2),nsmall=2))
        text((j+k)/2+startyear+20, 3.5, bquote(.(annstring)~sigma),pos=2,cex=.5)
        
    }
    for(k in overlapyrs){ 
        K=k-startyear
        lines(c(K,K)+startyear,c(-20,20))
    }
}
