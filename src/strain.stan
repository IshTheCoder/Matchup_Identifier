data {
   /* ... declarations ... */

int<lower=1> N;    
int<lower=1> N_blockers;  
int<lower=0> N_rushers;
int<lower=0> N_qbs;                               
matrix[N,N_blockers] X_blockers;                                                         
int rusher_random_effect[N];                          
int qb_random_effect[N];                      
vector[N] strain;
}

parameters {
    vector[N_blockers] betas;
    vector[N_qbs] betas_qb;
    vector[N_rushers] betas_rusher;
    positive_ordered[2] sigma_re;
    real<lower=0> lambda;                         
                              
}


model {
    betas ~ normal(0,1);    
    betas_qb ~ normal(0,sigma_re[1]);
    betas_rusher ~ normal(0,sigma_re[2]);
                                                 
    for (i in 1:N) {
       /* code */
       strain[i] ~ normal(X_blockers[i,]*betas + betas_qb[qb_random_effect[i]+1] + 
       betas_rusher[rusher_random_effect[i]+1], sum(sigma_re));
    }
}
