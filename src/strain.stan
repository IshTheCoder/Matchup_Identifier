data {
   /* ... declarations ... */

int<lower=1> N;    
int<lower=1> N_blockers;  
int<lower=0> N_rushers;   
vector[N] velocity;                         
matrix[N,N_blockers] X_blockers;                                                         
int rusher_identifier[N]; 
int num_rushers[N];  
int num_blockers[N];                                          
vector[N] acceleration;
}

parameters {
    vector<lower=0>[N_blockers] betas;
    vector<lower=0>[N_rushers] betas_rusher;
    real<lower=0> lambda;   
    real<lower=0, upper=1> scale;    
    real<lower=0, upper=1> scale_blockers;           
                              
}
transformed parameters {
   
   real mu[N];
   for (i in 1:N) {
      /* code */
      mu[i] =  betas_rusher[rusher_identifier[i]+1] - (X_blockers[i,]*betas*velocity[i]*num_rushers[i]*num_blockers[i]);
   }
}


model {
    scale_blockers ~ beta(2,10);
    scale ~ beta(10,2);
    betas ~ exponential(scale_blockers);
    betas_rusher ~ exponential(scale);                  
    acceleration ~ normal(mu, lambda);
}
