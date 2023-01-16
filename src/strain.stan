data {
   /* ... declarations ... */

int<lower=1> N;    
int<lower=1> N_blockers;  
int<lower=0> N_rushers;                          
matrix[N,N_blockers] X_blockers;                                                         
int rusher_identifier[N];                                          
vector[N] acceleration;
}

parameters {
    vector<lower=0>[N_blockers] betas;
    vector<lower=0>[N_rushers] betas_rusher;
    real<lower=0> lambda;            
                             
}

model {
    betas_rusher ~ exponential(1); 
    betas ~ exponential(1);                 
    acceleration ~ normal(betas_rusher[rusher_identifier] - X_blockers*betas, lambda);
}
