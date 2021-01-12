**$L_p$ projection实验问题:** 

$\beta(x_k)$降不下去，但是$\epsilon$更新的条件会被trigger，导致$\epsilon\to 0$ 

log:

$z=x-\alpha \nabla f(x^{(t)})$--待投影点
$\|z\|_p^p-\gamma = 0.01392105336016769$
进入$L_p$投影子问题:
  初始值：$\epsilon_0$ = 6.09504987e-15, $x_0 = x^{(t)}$

 输出参数：

+ Obj: $0.5\|x_k - z\|_2^2$ 

+ x-x_p: $\|x_{t+1} - x_t\|_2$ 

  1: Obj = 2.024e-08, alpha = 7.585e-16, beta = 8.206e-06, eps = 1.601e-18, dual = 4.180e-11, #nonzeros = 31, x-x_p = 3.076e-04 
  2: Obj = 2.024e-08, alpha = 3.990e-17, beta = 8.207e-06, eps = 4.203e-22, dual = 4.464e-11, #nonzeros = 31, x-x_p = 2.097e-10 
  3: Obj = 2.024e-08, alpha = 3.829e-17, beta = 8.207e-06, eps = 1.104e-25, dual = 4.469e-11, #nonzeros = 31, x-x_p = 2.242e-12 
  4: Obj = 2.024e-08, alpha = 3.553e-17, beta = 8.207e-06, eps = 2.899e-29, dual = 4.469e-11, #nonzeros = 31, x-x_p = 3.647e-14 
  5: Obj = 2.024e-08, alpha = 3.515e-17, beta = 8.207e-06, eps = 7.613e-33, dual = 4.469e-11, #nonzeros = 31, x-x_p = 6.104e-16 
  6: Obj = 2.024e-08, alpha = 3.515e-17, beta = 8.207e-06, eps = 1.999e-36, dual = 4.469e-11, #nonzeros = 31, x-x_p = 9.541e-18 
  7: Obj = 2.024e-08, alpha = 4.081e-17, beta = 8.206e-06, eps = 5.250e-40, dual = 3.344e-11, #nonzeros = 31, x-x_p = 5.087e-10 
  8: Obj = 2.024e-08, alpha = 3.133e-17, beta = 8.206e-06, eps = 1.379e-43, dual = 3.264e-11, #nonzeros = 31, x-x_p = 3.627e-11 
  9: Obj = 2.024e-08, alpha = 2.814e-17, beta = 8.205e-06, eps = 3.620e-47, dual = 2.299e-11, #nonzeros = 31, x-x_p = 4.367e-10 
   10: Obj = 2.024e-08, alpha = 2.813e-17, beta = 8.205e-06, eps = 9.505e-51, dual = 2.299e-11, #nonzeros = 31, x-x_p = 1.844e-15 
   11: Obj = 2.024e-08, alpha = 2.813e-17, beta = 8.205e-06, eps = 2.496e-54, dual = 2.299e-11, #nonzeros = 31, x-x_p = 0.000e+00 
   12: Obj = 2.024e-08, alpha = 2.813e-17, beta = 8.205e-06, eps = 6.553e-58, dual = 2.299e-11, #nonzeros = 31, x-x_p = 0.000e+00 
   13: Obj = 2.024e-08, alpha = 2.813e-17, beta = 8.205e-06, eps = 1.721e-61, dual = 2.299e-11, #nonzeros = 31, x-x_p = 0.000e+00 
   14: Obj = 2.024e-08, alpha = 2.813e-17, beta = 8.205e-06, eps = 4.518e-65, dual = 2.299e-11, #nonzeros = 31, x-x_p = 0.000e+00 
   15: Obj = 2.024e-08, alpha = 2.813e-17, beta = 8.205e-06, eps = 1.186e-68, dual = 2.299e-11, #nonzeros = 31, x-x_p = 0.000e+00 
   16: Obj = 2.024e-08, alpha = 2.813e-17, beta = 8.205e-06, eps = 3.115e-72, dual = 2.299e-11, #nonzeros = 31, x-x_p = 0.000e+00 
   17: Obj = 2.024e-08, alpha = 2.813e-17, beta = 8.205e-06, eps = 8.178e-76, dual = 2.299e-11, #nonzeros = 31, x-x_p = 0.000e+00 
   18: Obj = 2.024e-08, alpha = 2.813e-17, beta = 8.205e-06, eps = 2.147e-79, dual = 2.299e-11, #nonzeros = 31, x-x_p = 0.000e+00 
   19: Obj = 2.024e-08, alpha = 2.813e-17, beta = 8.205e-06, eps = 5.638e-83, dual = 2.299e-11, #nonzeros = 31, x-x_p = 0.000e+00 
   20: Obj = 2.024e-08, alpha = 2.813e-17, beta = 8.205e-06, eps = 1.480e-86, dual = 2.299e-11, #nonzeros = 31, x-x_p = 0.000e+00 
   21: Obj = 2.024e-08, alpha = 2.813e-17, beta = 8.205e-06, eps = 3.887e-90, dual = 2.299e-11, #nonzeros = 31, x-x_p = 0.000e+00 
   22: Obj = 2.024e-08, alpha = 2.813e-17, beta = 8.205e-06, eps = 1.021e-93, dual = 2.299e-11, #nonzeros = 31, x-x_p = 0.000e+00 
   23: Obj = 2.024e-08, alpha = 2.813e-17, beta = 8.205e-06, eps = 2.680e-97, dual = 2.299e-11, #nonzeros = 31, x-x_p = 0.000e+00 
   24: Obj = 2.024e-08, alpha = 2.813e-17, beta = 8.205e-06, eps = 7.037e-101, dual = 2.299e-11, #nonzeros = 31, x-x_p = 0.000e+00 
   25: Obj = 2.024e-08, alpha = 2.813e-17, beta = 8.205e-06, eps = 1.848e-104, dual = 2.299e-11, #nonzeros = 31, x-x_p = 0.000e+00 
   26: Obj = 2.024e-08, alpha = 2.813e-17, beta = 8.205e-06, eps = 4.851e-108, dual = 2.299e-11, #nonzeros = 31, x-x_p = 0.000e+00 
   27: Obj = 2.024e-08, alpha = 2.813e-17, beta = 8.205e-06, eps = 1.274e-111, dual = 2.299e-11, #nonzeros = 31, x-x_p = 0.000e+00 
   28: Obj = 2.024e-08, alpha = 2.813e-17, beta = 8.205e-06, eps = 3.345e-115, dual = 2.299e-11, #nonzeros = 31, x-x_p = 0.000e+00 
   29: Obj = 2.024e-08, alpha = 2.813e-17, beta = 8.205e-06, eps = 8.782e-119, dual = 2.299e-11, #nonzeros = 31, x-x_p = 0.000e+00 
   30: Obj = 2.024e-08, alpha = 2.813e-17, beta = 8.205e-06, eps = 2.306e-122, dual = 2.299e-11, #nonzeros = 31, x-x_p = 0.000e+00 
   31: Obj = 2.024e-08, alpha = 2.813e-17, beta = 8.205e-06, eps = 6.055e-126, dual = 2.299e-11, #nonzeros = 31, x-x_p = 0.000e+00 
   32: Obj = 2.024e-08, alpha = 2.813e-17, beta = 8.205e-06, eps = 1.590e-129, dual = 2.299e-11, #nonzeros = 31, x-x_p = 0.000e+00 
   33: Obj = 2.024e-08, alpha = 2.813e-17, beta = 8.205e-06, eps = 4.174e-133, dual = 2.299e-11, #nonzeros = 31, x-x_p = 0.000e+00 
   34: Obj = 2.024e-08, alpha = 2.813e-17, beta = 8.205e-06, eps = 1.096e-136, dual = 2.299e-11, #nonzeros = 31, x-x_p = 0.000e+00 
   35: Obj = 2.024e-08, alpha = 2.813e-17, beta = 8.205e-06, eps = 2.878e-140, dual = 2.299e-11, #nonzeros = 31, x-x_p = 0.000e+00 
   36: Obj = 2.024e-08, alpha = 2.813e-17, beta = 8.205e-06, eps = 7.557e-144, dual = 2.299e-11, #nonzeros = 31, x-x_p = 0.000e+00 
   37: Obj = 2.024e-08, alpha = 2.813e-17, beta = 8.205e-06, eps = 1.984e-147, dual = 2.299e-11, #nonzeros = 31, x-x_p = 0.000e+00 
   38: Obj = 2.024e-08, alpha = 2.813e-17, beta = 8.205e-06, eps = 5.210e-151, dual = 2.299e-11, #nonzeros = 31, x-x_p = 0.000e+00 
   39: Obj = 2.024e-08, alpha = 2.813e-17, beta = 8.205e-06, eps = 1.368e-154, dual = 2.299e-11, #nonzeros = 31, x-x_p = 0.000e+00 
   40: Obj = 2.024e-08, alpha = 2.813e-17, beta = 8.205e-06, eps = 3.592e-158, dual = 2.299e-11, #nonzeros = 31, x-x_p = 0.000e+00 
   41: Obj = 2.024e-08, alpha = 2.813e-17, beta = 8.205e-06, eps = 9.431e-162, dual = 2.299e-11, #nonzeros = 31, x-x_p = 0.000e+00 
   42: Obj = 2.024e-08, alpha = 2.813e-17, beta = 8.205e-06, eps = 2.476e-165, dual = 2.299e-11, #nonzeros = 31, x-x_p = 0.000e+00 
   43: Obj = 2.024e-08, alpha = 2.813e-17, beta = 8.205e-06, eps = 6.502e-169, dual = 2.299e-11, #nonzeros = 31, x-x_p = 0.000e+00 
   44: Obj = 2.024e-08, alpha = 2.813e-17, beta = 8.205e-06, eps = 1.707e-172, dual = 2.299e-11, #nonzeros = 31, x-x_p = 0.000e+00 
   45: Obj = 2.024e-08, alpha = 2.813e-17, beta = 8.205e-06, eps = 4.483e-176, dual = 2.299e-11, #nonzeros = 31, x-x_p = 0.000e+00 
   46: Obj = 2.024e-08, alpha = 2.813e-17, beta = 8.205e-06, eps = 1.177e-179, dual = 2.299e-11, #nonzeros = 31, x-x_p = 0.000e+00 
   47: Obj = 2.024e-08, alpha = 2.813e-17, beta = 8.205e-06, eps = 3.091e-183, dual = 2.299e-11, #nonzeros = 31, x-x_p = 0.000e+00 
   48: Obj = 2.024e-08, alpha = 2.813e-17, beta = 8.205e-06, eps = 8.115e-187, dual = 2.299e-11, #nonzeros = 31, x-x_p = 0.000e+00 
   49: Obj = 2.024e-08, alpha = 2.813e-17, beta = 8.205e-06, eps = 2.131e-190, dual = 2.299e-11, #nonzeros = 31, x-x_p = 0.000e+00 
   50: Obj = 2.024e-08, alpha = 2.813e-17, beta = 8.205e-06, eps = 5.595e-194, dual = 2.299e-11, #nonzeros = 31, x-x_p = 0.000e+00 
   51: Obj = 2.024e-08, alpha = 2.813e-17, beta = 8.205e-06, eps = 1.469e-197, dual = 2.299e-11, #nonzeros = 31, x-x_p = 0.000e+00 
   52: Obj = 2.024e-08, alpha = 2.813e-17, beta = 8.205e-06, eps = 3.857e-201, dual = 2.299e-11, #nonzeros = 31, x-x_p = 0.000e+00 
   53: Obj = 2.024e-08, alpha = 2.813e-17, beta = 8.205e-06, eps = 1.013e-204, dual = 2.299e-11, #nonzeros = 31, x-x_p = 0.000e+00 
   54: Obj = 2.024e-08, alpha = 2.813e-17, beta = 8.205e-06, eps = 2.659e-208, dual = 2.299e-11, #nonzeros = 31, x-x_p = 0.000e+00 
   55: Obj = 2.024e-08, alpha = 2.813e-17, beta = 8.205e-06, eps = 6.982e-212, dual = 2.299e-11, #nonzeros = 31, x-x_p = 0.000e+00 
   56: Obj = 2.024e-08, alpha = 2.813e-17, beta = 8.205e-06, eps = 1.833e-215, dual = 2.299e-11, #nonzeros = 31, x-x_p = 0.000e+00 
   57: Obj = 2.024e-08, alpha = 2.813e-17, beta = 8.205e-06, eps = 4.814e-219, dual = 2.299e-11, #nonzeros = 31, x-x_p = 0.000e+00 
   58: Obj = 2.024e-08, alpha = 2.813e-17, beta = 8.205e-06, eps = 1.264e-222, dual = 2.299e-11, #nonzeros = 31, x-x_p = 0.000e+00 
   59: Obj = 2.024e-08, alpha = 2.813e-17, beta = 8.205e-06, eps = 3.319e-226, dual = 2.299e-11, #nonzeros = 31, x-x_p = 0.000e+00 
   60: Obj = 2.024e-08, alpha = 2.813e-17, beta = 8.205e-06, eps = 8.714e-230, dual = 2.299e-11, #nonzeros = 31, x-x_p = 0.000e+00 
   61: Obj = 2.024e-08, alpha = 2.813e-17, beta = 8.205e-06, eps = 2.288e-233, dual = 2.299e-11, #nonzeros = 31, x-x_p = 0.000e+00 
   62: Obj = 2.024e-08, alpha = 2.813e-17, beta = 8.205e-06, eps = 6.008e-237, dual = 2.299e-11, #nonzeros = 31, x-x_p = 0.000e+00 
   63: Obj = 2.024e-08, alpha = 2.813e-17, beta = 8.205e-06, eps = 1.578e-240, dual = 2.299e-11, #nonzeros = 31, x-x_p = 0.000e+00 
   64: Obj = 2.024e-08, alpha = 2.813e-17, beta = 8.205e-06, eps = 4.142e-244, dual = 2.299e-11, #nonzeros = 31, x-x_p = 0.000e+00 
   65: Obj = 2.024e-08, alpha = 2.813e-17, beta = 8.205e-06, eps = 1.088e-247, dual = 2.299e-11, #nonzeros = 31, x-x_p = 0.000e+00 
   66: Obj = 2.024e-08, alpha = 2.813e-17, beta = 8.205e-06, eps = 2.856e-251, dual = 2.299e-11, #nonzeros = 31, x-x_p = 0.000e+00 
   67: Obj = 2.024e-08, alpha = 2.813e-17, beta = 8.205e-06, eps = 7.498e-255, dual = 2.299e-11, #nonzeros = 31, x-x_p = 0.000e+00 
   68: Obj = 2.024e-08, alpha = 2.813e-17, beta = 8.205e-06, eps = 1.969e-258, dual = 2.299e-11, #nonzeros = 31, x-x_p = 0.000e+00 
   69: Obj = 2.024e-08, alpha = 2.813e-17, beta = 8.205e-06, eps = 5.170e-262, dual = 2.299e-11, #nonzeros = 31, x-x_p = 0.000e+00 
   70: Obj = 2.024e-08, alpha = 2.813e-17, beta = 8.205e-06, eps = 1.357e-265, dual = 2.299e-11, #nonzeros = 31, x-x_p = 0.000e+00 
   71: Obj = 2.024e-08, alpha = 2.813e-17, beta = 8.205e-06, eps = 3.564e-269, dual = 2.299e-11, #nonzeros = 31, x-x_p = 0.000e+00 
   72: Obj = 2.024e-08, alpha = 2.813e-17, beta = 8.205e-06, eps = 9.358e-273, dual = 2.299e-11, #nonzeros = 31, x-x_p = 0.000e+00 
   73: Obj = 2.024e-08, alpha = 2.813e-17, beta = 8.205e-06, eps = 2.457e-276, dual = 2.299e-11, #nonzeros = 31, x-x_p = 0.000e+00 
   74: Obj = 2.024e-08, alpha = 2.813e-17, beta = 8.205e-06, eps = 6.452e-280, dual = 2.299e-11, #nonzeros = 31, x-x_p = 0.000e+00 
   75: Obj = 2.024e-08, alpha = 2.813e-17, beta = 8.205e-06, eps = 1.694e-283, dual = 2.299e-11, #nonzeros = 31, x-x_p = 0.000e+00 
   76: Obj = 2.024e-08, alpha = 2.813e-17, beta = 8.205e-06, eps = 4.448e-287, dual = 2.299e-11, #nonzeros = 31, x-x_p = 0.000e+00 
   77: Obj = 2.024e-08, alpha = 2.813e-17, beta = 8.205e-06, eps = 1.168e-290, dual = 2.299e-11, #nonzeros = 31, x-x_p = 0.000e+00 
   78: Obj = 2.024e-08, alpha = 2.813e-17, beta = 8.205e-06, eps = 3.067e-294, dual = 2.299e-11, #nonzeros = 31, x-x_p = 0.000e+00 
   79: Obj = 2.024e-08, alpha = 2.813e-17, beta = 8.205e-06, eps = 8.052e-298, dual = 2.299e-11, #nonzeros = 31, x-x_p = 0.000e+00 
   80: Obj = 2.024e-08, alpha = 2.813e-17, beta = 8.205e-06, eps = 2.114e-301, dual = 2.299e-11, #nonzeros = 31, x-x_p = 0.000e+00 
   81: Obj = 2.024e-08, alpha = 2.813e-17, beta = 8.205e-06, eps = 5.551e-305, dual = 2.299e-11, #nonzeros = 31, x-x_p = 0.000e+00 
   82: Obj = 2.024e-08, alpha = 2.813e-17, beta = 8.205e-06, eps = 1.458e-308, dual = 2.299e-11, #nonzeros = 31, x-x_p = 0.000e+00 
   83: Obj = 2.024e-08, alpha = 2.813e-17, beta = 8.205e-06, eps = 3.827e-312, dual = 2.299e-11, #nonzeros = 31, x-x_p = 0.000e+00 
   84: Obj = 2.024e-08, alpha = 2.813e-17, beta = 8.205e-06, eps = 1.005e-315, dual = 2.299e-11, #nonzeros = 31, x-x_p = 0.000e+00 
   85: Obj = 2.024e-08, alpha = 2.813e-17, beta = 8.205e-06, eps = 2.639e-319, dual = 2.299e-11, #nonzeros = 31, x-x_p = 0.000e+00 
   86: Obj = 2.024e-08, alpha = 2.813e-17, beta = 8.205e-06, eps = 6.917e-323, dual = 2.299e-11, #nonzeros = 31, x-x_p = 0.000e+00 



猜测：当$\epsilon_0$很小时可能出现
