source /data/tjdx_user/anaconda3/bin/activate deep_one_class

# one-class svdd
sh scripts/mnist_svdd.sh cpu mnist_0_one_class_svdd 0 adam 0.0001 150 1 0 0 0,1 2,3
sh scripts/mnist_svdd.sh cpu mnist_1_one_class_svdd 1 adam 0.0001 150 1 0 0 1,2 3,4
sh scripts/mnist_svdd.sh cpu mnist_2_one_class_svdd 2 adam 0.0001 150 1 0 0 2,3 4,5
sh scripts/mnist_svdd.sh cpu mnist_3_one_class_svdd 3 adam 0.0001 150 1 0 0 3,4 5,6
sh scripts/mnist_svdd.sh cpu mnist_4_one_class_svdd 4 adam 0.0001 150 1 0 0 4,5 6,7
sh scripts/mnist_svdd.sh cpu mnist_5_one_class_svdd 5 adam 0.0001 150 1 0 0 5,6 7,8
sh scripts/mnist_svdd.sh cpu mnist_6_one_class_svdd 6 adam 0.0001 150 1 0 0 6,7 8,9
sh scripts/mnist_svdd.sh cpu mnist_7_one_class_svdd 7 adam 0.0001 150 1 0 0 7,8 9,0
sh scripts/mnist_svdd.sh cpu mnist_8_one_class_svdd 8 adam 0.0001 150 1 0 0 8,9 0,1
sh scripts/mnist_svdd.sh cpu mnist_9_one_class_svdd 9 adam 0.0001 150 1 0 0 9,0 1,2

# soft-boundary svdd
sh scripts/mnist_svdd.sh cpu mnist_0_soft_boundary_svdd 0 adam 0.0001 150 0 0 0 0,1 2,3
sh scripts/mnist_svdd.sh cpu mnist_1_soft_boundary_svdd 1 adam 0.0001 150 0 0 0 1,2 3,4
sh scripts/mnist_svdd.sh cpu mnist_2_soft_boundary_svdd 2 adam 0.0001 150 0 0 0 2,3 4,5
sh scripts/mnist_svdd.sh cpu mnist_3_soft_boundary_svdd 3 adam 0.0001 150 0 0 0 3,4 5,6
sh scripts/mnist_svdd.sh cpu mnist_4_soft_boundary_svdd 4 adam 0.0001 150 0 0 0 4,5 6,7
sh scripts/mnist_svdd.sh cpu mnist_5_soft_boundary_svdd 5 adam 0.0001 150 0 0 0 5,6 7,8
sh scripts/mnist_svdd.sh cpu mnist_6_soft_boundary_svdd 6 adam 0.0001 150 0 0 0 6,7 8,9
sh scripts/mnist_svdd.sh cpu mnist_7_soft_boundary_svdd 7 adam 0.0001 150 0 0 0 7,8 9,0
sh scripts/mnist_svdd.sh cpu mnist_8_soft_boundary_svdd 8 adam 0.0001 150 0 0 0 8,9 0,1
sh scripts/mnist_svdd.sh cpu mnist_9_soft_boundary_svdd 9 adam 0.0001 150 0 0 0 9,0 1,2

#one-class svm
sh scripts/mnist_ocsvm.sh mnist_0_ocsvm 0 0.1 0 0,1 2,3
sh scripts/mnist_ocsvm.sh mnist_1_ocsvm 1 0.1 0 1,2 3,4
sh scripts/mnist_ocsvm.sh mnist_2_ocsvm 2 0.1 0 2,3 4,5
sh scripts/mnist_ocsvm.sh mnist_3_ocsvm 3 0.1 0 3,4 5,6
sh scripts/mnist_ocsvm.sh mnist_4_ocsvm 4 0.1 0 4,5 6,7
sh scripts/mnist_ocsvm.sh mnist_5_ocsvm 5 0.1 0 5,6 7,8
sh scripts/mnist_ocsvm.sh mnist_6_ocsvm 6 0.1 0 6,7 8,9
sh scripts/mnist_ocsvm.sh mnist_7_ocsvm 7 0.1 0 7,8 9,0
sh scripts/mnist_ocsvm.sh mnist_8_ocsvm 8 0.1 0 8,9 0,1
sh scripts/mnist_ocsvm.sh mnist_9_ocsvm 9 0.1 0 9,0 1,2

#kde
sh scripts/mnist_kde.sh mnist_0_kde 0 0 0,1 2,3
sh scripts/mnist_kde.sh mnist_1_kde 1 0 1,2 3,4
sh scripts/mnist_kde.sh mnist_2_kde 2 0 2,3 4,5
sh scripts/mnist_kde.sh mnist_3_kde 3 0 3,4 5,6
sh scripts/mnist_kde.sh mnist_4_kde 4 0 4,5 6,7
sh scripts/mnist_kde.sh mnist_5_kde 5 0 5,6 7,8
sh scripts/mnist_kde.sh mnist_6_kde 6 0 6,7 8,9
sh scripts/mnist_kde.sh mnist_7_kde 7 0 7,8 9,0
sh scripts/mnist_kde.sh mnist_8_kde 8 0 8,9 0,1
sh scripts/mnist_kde.sh mnist_9_kde 9 0 9,0 1,2

#if
sh scripts/mnist_if.sh mnist_0_if 0 0.1 0 0,1 2,3
sh scripts/mnist_if.sh mnist_1_if 1 0.1 0 1,2 3,4
sh scripts/mnist_if.sh mnist_2_if 2 0.1 0 2,3 4,5
sh scripts/mnist_if.sh mnist_3_if 3 0.1 0 3,4 5,6
sh scripts/mnist_if.sh mnist_4_if 4 0.1 0 4,5 6,7
sh scripts/mnist_if.sh mnist_5_if 5 0.1 0 5,6 7,8
sh scripts/mnist_if.sh mnist_6_if 6 0.1 0 6,7 8,9
sh scripts/mnist_if.sh mnist_7_if 7 0.1 0 7,8 9,0
sh scripts/mnist_if.sh mnist_8_if 8 0.1 0 8,9 0,1
sh scripts/mnist_if.sh mnist_9_if 9 0.1 0 9,0 1,2

#cae
sh scripts/mnist_cae.sh cpu mnist_0_cae 0 adam 0.0001 150 0,1 2,3
sh scripts/mnist_cae.sh cpu mnist_1_cae 1 adam 0.0001 150 1,2 3,4
sh scripts/mnist_cae.sh cpu mnist_2_cae 2 adam 0.0001 150 2,3 4,5
sh scripts/mnist_cae.sh cpu mnist_3_cae 3 adam 0.0001 150 3,4 5,6
sh scripts/mnist_cae.sh cpu mnist_4_cae 4 adam 0.0001 150 4,5 6,7
sh scripts/mnist_cae.sh cpu mnist_5_cae 5 adam 0.0001 150 5,6 7,8
sh scripts/mnist_cae.sh cpu mnist_6_cae 6 adam 0.0001 150 6,7 8,9
sh scripts/mnist_cae.sh cpu mnist_7_cae 7 adam 0.0001 150 7,8 9,0
sh scripts/mnist_cae.sh cpu mnist_8_cae 8 adam 0.0001 150 8,9 0,1
sh scripts/mnist_cae.sh cpu mnist_9_cae 9 adam 0.0001 150 9,0 1,2
