import theano.tensor as T
import numpy as np
import theano
import config as Cfg
from config import Configuration as Cfg

floatX = Cfg.floatX
k = 3
N = 4
z = 5
#
# m = theano.shared(np.array([[20.68749969,  6.4999999 , 20.81249967, 14.12499977,  7.87499985],
#                             [ 6.4999999 ,  4.62499993, 11.7499998 , 11.7499998 , 12.12499977],
#                             [20.81249967, 11.7499998 , 31.87499942, 29.37499945, 27.56249941],
#                             [14.12499977, 11.7499998 , 29.37499945, 30.74999942, 32.74999931],
#                             [ 7.87499985, 12.12499977, 27.56249941, 32.74999931, 42.9374991 ]]))
#
# print T.nlinalg.det(m).eval()
#
# exit(0)

gama = theano.shared(np.float32([[0.2,0.4,0.4], [0.7,0.2,0.1], [0.2,0.1,0.7], [0.5,0.3,0.2]]))

z_var = theano.shared(np.float32([[1, 2, 3, 4, 9], [1, 2, 5, 6, 8], [1, 3, 6, 8, 6], [8, 2, 7, 4, 1]]))




mu_N_k_z, updates_dete = theano.scan(lambda z_s , gama_s: T.outer(z_s, gama_s), sequences=[z_var, gama])

phi_k = T.mean(gama, axis=0)

mu_z_k =T.sum(mu_N_k_z, axis=0)/T.sum(gama, axis= 0)

mu_k_z = T.transpose(mu_z_k, (1,0))

sigma_N_k_z_z, _  = theano.scan(lambda z_v, gama_v: theano.scan(lambda mu_v, gama_v_v:gama_v_v*T.outer(z_v-mu_v,z_v-mu_v) ,\
                                                          sequences= [mu_k_z, gama_v]), sequences=[z_var, gama])

sigma_k_z_z = T.sum(sigma_N_k_z_z, axis=0)
sigma_k_z_z = T.transpose(T.transpose(sigma_k_z_z,(1,2,0))/T.sum(gama, axis= 0), (2,0,1))

logits, _ = theano.scan(lambda z_v: theano.scan(lambda mu_v, sigma_v, phi_v:phi_v* T.exp(T.sum((-0.5)*T.dot((z_v-mu_v),T.nlinalg.matrix_inverse(sigma_v))*(z_v-mu_v)))/T.sqrt(T.nlinalg.det(2*np.pi*sigma_v)),\
                                                 sequences = [mu_k_z, sigma_k_z_z, phi_k]), sequences=z_var)


logits = T.sum(logits, axis=1)

logits = T.mean(logits)

print sigma_k_z_z.eval()


exit(0)
gama_v_v*T.dot((z_v-mu_v),T.transpose(z_v-mu_v,(0,1)))
print (T.reshape(det/b, (k,z))).eval()

phi = T.mean(gama, axis=0)

phi_sum = T.sum(gama, axis = 0)


scaler = theano.shared(np.ones([1, z]))

gama_scaler = T.dot(T.reshape(gama, (N,k,1)), scaler)

gama_transpose = T.transpose(gama_scaler, (1, 0, 2))

mean = T.mul(gama_transpose, z_var)

mean_1 = T.sum(mean, axis=1)

mean_2 = T.transpose(T.transpose(mean_1,(1,0))/phi_sum, (1, 0))

# rep = theano.shared(np.float32([[1 ,1], [2, 0]]))
#
# volume = T.cast(T.sum(floatX(-0.5) *(T.log(floatX(2 * np.pi)) + (rep ** 2)) , axis=1, dtype='floatX'), dtype='floatX')
# log_pro = T.cast(T.exp(volume), dtype='floatX')
# score = T.eq
# log_likehood = T.mean(log_pro)


#
# print (gama_transpose.eval())
#
# print (z_var.eval())
#
# print (mean_2.eval())


z_mean = T.dot(T.reshape(z_var,(N, z, 1)), theano.shared(np.ones([1, k])))

z_mean = T.transpose(z_mean, (0,2,1))

z_mean = z_mean - mean_2

z_mean= T.transpose(z_mean,(1, 0, 2)) # k, N , z


z_mean_scale = T.dot(T.reshape(z_mean,(k,N,z,1)), theano.shared(np.ones([1,z])))

z_mean_scale_t = T.transpose(z_mean_scale,(0,1, 3, 2))

z_dot = z_mean_scale*z_mean_scale_t


scaler_1 = theano.shared(np.ones([1, z]))

gama_scaler_1 = T.dot(T.reshape(gama, (N,k,1)), scaler_1)
gama_scaler_1 = T.dot(T.reshape(gama_scaler_1, (N,k,z,1)), scaler_1)

T.transpose(gama_scaler_1,(1,0,2,3))

variance = T.transpose(gama_scaler_1,(1,0,2,3))*z_dot

variance = T.transpose(T.sum(variance, axis=1), (1,2,0)) /phi_sum

variance = T.transpose(variance, (2,0,1))

#print (variance.eval())

#variance_a = theano.shared(np.array([[[1,2,3,0.3],[5,0.6,7,8],[2,3,0.4,5],[15,12,0.7,8]],[[1,2,3,0.3],[5,0.6,7,8],[2,3,0.4,5],[15,12,0.7,8]]]))

det, updates_dete = theano.scan(lambda v: T.nlinalg.Det()(v), sequences=variance)

inverse_matric, updates_inverse = theano.scan(lambda v: T.sqrt(T.nlinalg.matrix_inverse(2*np.pi*v)),  sequences=variance)





z_mean = T.dot(T.reshape(z_var,(N, z, 1)), theano.shared(np.ones([1, k])))

z_mean = T.transpose(z_mean, (0,2,1))

z_mean = z_mean - mean_2   # n, k, z1

z_mean1 = T.dot(T.reshape(z_mean, (N, k, z, 1)), theano.shared(np.ones([1,z]))) #n , k , z1, z2

inverse_matric_scaler  = T.dot(T.reshape(inverse_matric, (k, z, z, 1)), theano.shared(np.ones([1, N])))

inverse_matric_scaler = T.transpose(inverse_matric_scaler, (3, 0, 1, 2)) # n, k, z1, z2

z_dot_var = z_mean1* inverse_matric_scaler # n,k,z1,z2

z_dot_var = T.sum(z_dot_var, axis=2) # n, k , z2

logits = T.sum((-0.5)*z_dot_var*z_mean1, axis=2 )# n, k,

pro = T.sum(((T.exp(logits)*phi)/det), axis=1)

print z_dot_var.eval()


















# rep_dim = 2
# batch_size  = 2
# rep_tranpose = T.transpose(rep, (1, 0))
# rep_reshape = T.reshape(rep_tranpose, [rep_dim, batch_size, 1])
# transfer_vector = theano.shared(np.ones([1, batch_size], dtype='float32'))
# result = T.dot(rep_reshape, transfer_vector)
# result1 = T.transpose(result, (0, 2, 1))
# subtract = result - result1
# print (subtract.eval())
# KL_volume = T.cast(-0.5*(np.log(2 * np.pi)+(subtract ** 2)), dtype='floatX')
# KL_volume = T.sum(KL_volume, axis=0)
# print (KL_volume.eval())
# KL_pro = T.cast(T.exp(KL_volume), dtype='floatX')
# print (KL_pro.eval())
# KL_pro_average = T.mean(KL_pro, axis=1)
# log_KL_pro = T.log(KL_pro_average)
# entropy = T.mean(log_KL_pro)
#
#
# ones = theano.shared(np.float32([[1, 2, 3], [4, 5, 6]]))
# ones = T.transpose(ones, (1,0))
# ones = T.reshape(ones,[3,2,1])
# transfer_vector = theano.shared(np.ones([1,2],dtype='float32'))
# #transfer_vector = T.reshape(transfer_vector,[3,1])
# result = T.dot(ones, transfer_vector)
# result1 = T.transpose(result, (0, 2, 1))
# final = result - result1
# sum  = T.sum(final, axis= 0)
# #print (transfer_vector.get_value())
# # volume = -0.5*T.sum(((rep) ** 2), axis=1, dtype='floatX')
# # log_pro = T.constant(2*np.pi)*T.exp(volume)
# # log_likehood = T.mean(log_pro)
