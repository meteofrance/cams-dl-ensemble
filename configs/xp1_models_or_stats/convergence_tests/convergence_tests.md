# Convergence tests configs

We want to find a config that allows models to converge faster.

Our baseline config to beat are `../3_11_CTM_HalfUnet.yaml` and `../3_11_CTM_UNetRPP.yaml`.

Leads explored:
0. Changing learning rate from 10^-4 to 10^-3.
1. Add a learning rate scheduler.
2. Changing from L1 loss to MAE + learning rate 10^-3
3. UNetRPP reduce number of internal layers