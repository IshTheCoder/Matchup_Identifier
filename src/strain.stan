data {
   /* ... declarations ... */

int<lower=1> N;    
int<lower=1> N_blockers;  
int<lower=0> N_rushers;   
vector[N] velocity;                         
matrix[N,N_blockers] X_blockers;                                                         
int rusher_identifier[N]; 
int num_rushers[N];                                            
vector[N] acceleration;
}

parameters {
    simplex[N_blockers] betas;
    vector<lower=0>[N_rushers] betas_rusher;
    real<lower=0> lambda;   
    real<lower=0, upper=1> scale;               
                              
}
transformed parameters {
   
   real mu[N];
   for (i in 1:N) {
      /* code */
      mu[i] =  betas_rusher[rusher_identifier[i]+1] - (X_blockers[i,]*betas*velocity[i]*num_rushers[i]);
   }
}


model {
    scale ~ beta(10,2);
    betas_rusher ~ exponential(scale);                  
    acceleration ~ normal(mu, lambda);
}
