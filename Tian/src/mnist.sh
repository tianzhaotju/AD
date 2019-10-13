source /data/tjdx_user/anaconda3/bin/activate deep_one_class

seed=$1

# one-class svdd
sh scripts/mnist_svdd.sh cpu mnist_0_one_class_svdd $seed adam 0.0001 150 1 0 0 0,1 2,3
sh scripts/mnist_svdd.sh cpu mnist_1_one_class_svdd $seed adam 0.0001 150 1 0 0 1,2 3,4
sh scripts/mnist_svdd.sh cpu mnist_2_one_class_svdd $seed adam 0.0001 150 1 0 0 2,3 4,5
sh scripts/mnist_svdd.sh cpu mnist_3_one_class_svdd $seed adam 0.0001 150 1 0 0 3,4 5,6
sh scripts/mnist_svdd.sh cpu mnist_4_one_class_svdd $seed adam 0.0001 150 1 0 0 4,5 6,7
sh scripts/mnist_svdd.sh cpu mnist_5_one_class_svdd $seed adam 0.0001 150 1 0 0 5,6 7,8
sh scripts/mnist_svdd.sh cpu mnist_6_one_class_svdd $seed adam 0.0001 150 1 0 0 6,7 8,9
sh scripts/mnist_svdd.sh cpu mnist_7_one_class_svdd $seed adam 0.0001 150 1 0 0 7,8 9,0
sh scripts/mnist_svdd.sh cpu mnist_8_one_class_svdd $seed adam 0.0001 150 1 0 0 8,9 0,1
sh scripts/mnist_svdd.sh cpu mnist_9_one_class_svdd $seed adam 0.0001 150 1 0 0 9,0 1,2

# soft-boundary svdd
sh scripts/mnist_svdd.sh cpu mnist_0_soft_boundary_svdd $seed adam 0.0001 150 0 0 0 0,1 2,3
sh scripts/mnist_svdd.sh cpu mnist_1_soft_boundary_svdd $seed adam 0.0001 150 0 0 0 1,2 3,4
sh scripts/mnist_svdd.sh cpu mnist_2_soft_boundary_svdd $seed adam 0.0001 150 0 0 0 2,3 4,5
sh scripts/mnist_svdd.sh cpu mnist_3_soft_boundary_svdd $seed adam 0.0001 150 0 0 0 3,4 5,6
sh scripts/mnist_svdd.sh cpu mnist_4_soft_boundary_svdd $seed adam 0.0001 150 0 0 0 4,5 6,7
sh scripts/mnist_svdd.sh cpu mnist_5_soft_boundary_svdd $seed adam 0.0001 150 0 0 0 5,6 7,8
sh scripts/mnist_svdd.sh cpu mnist_6_soft_boundary_svdd $seed adam 0.0001 150 0 0 0 6,7 8,9
sh scripts/mnist_svdd.sh cpu mnist_7_soft_boundary_svdd $seed adam 0.0001 150 0 0 0 7,8 9,0
sh scripts/mnist_svdd.sh cpu mnist_8_soft_boundary_svdd $seed adam 0.0001 150 0 0 0 8,9 0,1
sh scripts/mnist_svdd.sh cpu mnist_9_soft_boundary_svdd $seed adam 0.0001 150 0 0 0 9,0 1,2

#one-class svm
sh scripts/mnist_ocsvm.sh mnist_0_ocsvm $seed 0.1 0 0,1 2,3
sh scripts/mnist_ocsvm.sh mnist_1_ocsvm $seed 0.1 0 1,2 3,4
sh scripts/mnist_ocsvm.sh mnist_2_ocsvm $seed 0.1 0 2,3 4,5
sh scripts/mnist_ocsvm.sh mnist_3_ocsvm $seed 0.1 0 3,4 5,6
sh scripts/mnist_ocsvm.sh mnist_4_ocsvm $seed 0.1 0 4,5 6,7
sh scripts/mnist_ocsvm.sh mnist_5_ocsvm $seed 0.1 0 5,6 7,8
sh scripts/mnist_ocsvm.sh mnist_6_ocsvm $seed 0.1 0 6,7 8,9
sh scripts/mnist_ocsvm.sh mnist_7_ocsvm $seed 0.1 0 7,8 9,0
sh scripts/mnist_ocsvm.sh mnist_8_ocsvm $seed 0.1 0 8,9 0,1
sh scripts/mnist_ocsvm.sh mnist_9_ocsvm $seed 0.1 0 9,0 1,2

#kde
sh scripts/mnist_kde.sh mnist_0_kde $seed 0 0,1 2,3
sh scripts/mnist_kde.sh mnist_1_kde $seed 0 1,2 3,4
sh scripts/mnist_kde.sh mnist_2_kde $seed 0 2,3 4,5
sh scripts/mnist_kde.sh mnist_3_kde $seed 0 3,4 5,6
sh scripts/mnist_kde.sh mnist_4_kde $seed 0 4,5 6,7
sh scripts/mnist_kde.sh mnist_5_kde $seed 0 5,6 7,8
sh scripts/mnist_kde.sh mnist_6_kde $seed 0 6,7 8,9
sh scripts/mnist_kde.sh mnist_7_kde $seed 0 7,8 9,0
sh scripts/mnist_kde.sh mnist_8_kde $seed 0 8,9 0,1
sh scripts/mnist_kde.sh mnist_9_kde $seed 0 9,0 1,2

#if
sh scripts/mnist_if.sh mnist_0_if $seed 0.1 0 0,1 2,3
sh scripts/mnist_if.sh mnist_1_if $seed 0.1 0 1,2 3,4
sh scripts/mnist_if.sh mnist_2_if $seed 0.1 0 2,3 4,5
sh scripts/mnist_if.sh mnist_3_if $seed 0.1 0 3,4 5,6
sh scripts/mnist_if.sh mnist_4_if $seed 0.1 0 4,5 6,7
sh scripts/mnist_if.sh mnist_5_if $seed 0.1 0 5,6 7,8
sh scripts/mnist_if.sh mnist_6_if $seed 0.1 0 6,7 8,9
sh scripts/mnist_if.sh mnist_7_if $seed 0.1 0 7,8 9,0
sh scripts/mnist_if.sh mnist_8_if $seed 0.1 0 8,9 0,1
sh scripts/mnist_if.sh mnist_9_if $seed 0.1 0 9,0 1,2

#cae
sh scripts/mnist_cae.sh cpu mnist_0_cae $seed adam 0.0001 150 0,1 2,3
sh scripts/mnist_cae.sh cpu mnist_1_cae $seed adam 0.0001 150 1,2 3,4
sh scripts/mnist_cae.sh cpu mnist_2_cae $seed adam 0.0001 150 2,3 4,5
sh scripts/mnist_cae.sh cpu mnist_3_cae $seed adam 0.0001 150 3,4 5,6
sh scripts/mnist_cae.sh cpu mnist_4_cae $seed adam 0.0001 150 4,5 6,7
sh scripts/mnist_cae.sh cpu mnist_5_cae $seed adam 0.0001 150 5,6 7,8
sh scripts/mnist_cae.sh cpu mnist_6_cae $seed adam 0.0001 150 6,7 8,9
sh scripts/mnist_cae.sh cpu mnist_7_cae $seed adam 0.0001 150 7,8 9,0
sh scripts/mnist_cae.sh cpu mnist_8_cae $seed adam 0.0001 150 8,9 0,1
sh scripts/mnist_cae.sh cpu mnist_9_cae $seed adam 0.0001 150 9,0 1,2
