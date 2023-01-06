data {
   /* ... declarations ... */

int<lower=1> N;    
int<lower=1> N_blockers;  
int<lower=0> N_rushers;                            
matrix[N,N_blockers] X_blockers;                                                         
int rusher_identifier[N];                                             
vector[N] strain;
}

parameters {
    vector[N_blockers] betas;
    vector<lower = 1, upper = 10> [N_rushers] betas_rusher;
    real<lower=0> lambda;                         
                              
}
transformed parameters {
   
   real mu[N];
   for (i in 1:N) {
      /* code */
      mu[i] = betas_rusher[rusher_identifier[i]+1]/(X_blockers[i,]*betas);
   }
}


model {
    betas ~ normal(0,1);    
    betas_rusher ~ normal(2,1);                        
    strain ~ normal(mu, lambda);
}
