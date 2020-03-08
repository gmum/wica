\* IL_Spearman *\
Minimize
Total_Cost: 0.655043506937 gamm_var_0 + 0.189641891429 gamm_var_1
 + 0.0785836124506 gamm_var_2 + 0.617939491804 gamm_var_3
Subject To
eq_con_0: gamm_var_0 + gamm_var_2 = 1
eq_con_1: gamm_var_1 + gamm_var_3 = 1
eq_con_2: gamm_var_0 + gamm_var_1 = 1
eq_con_3: gamm_var_2 + gamm_var_3 = 1
max_gamma_sum: gamm_var_0 + gamm_var_1 + gamm_var_2 + gamm_var_3 = 2
Binaries
gamm_var_0
gamm_var_1
gamm_var_2
gamm_var_3
End
