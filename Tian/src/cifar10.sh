source /data/tjdx_user/anaconda3/bin/activate deep_one_class

seed=$1

#one-class svdd
sh scripts/cifar10_svdd.sh cpu cifar10_0_one_class_svdd $seed adam 0.0001 150 0.1 1 0 0 model.p 1 0,1 2,3
sh scripts/cifar10_svdd.sh cpu cifar10_1_one_class_svdd $seed adam 0.0001 150 0.1 1 0 0 model.p 1 1,2 3,4
sh scripts/cifar10_svdd.sh cpu cifar10_2_one_class_svdd $seed adam 0.0001 150 0.1 1 0 0 model.p 1 2,3 4,5
sh scripts/cifar10_svdd.sh cpu cifar10_3_one_class_svdd $seed adam 0.0001 150 0.1 1 0 0 model.p 1 3,4 5,6
sh scripts/cifar10_svdd.sh cpu cifar10_4_one_class_svdd $seed adam 0.0001 150 0.1 1 0 0 model.p 1 4,5 6,7
sh scripts/cifar10_svdd.sh cpu cifar10_5_one_class_svdd $seed adam 0.0001 150 0.1 1 0 0 model.p 1 5,6 7,8
sh scripts/cifar10_svdd.sh cpu cifar10_6_one_class_svdd $seed adam 0.0001 150 0.1 1 0 0 model.p 1 6,7 8,9
sh scripts/cifar10_svdd.sh cpu cifar10_7_one_class_svdd $seed adam 0.0001 150 0.1 1 0 0 model.p 1 7,8 9,0
sh scripts/cifar10_svdd.sh cpu cifar10_8_one_class_svdd $seed adam 0.0001 150 0.1 1 0 0 model.p 1 8,9 0,1
sh scripts/cifar10_svdd.sh cpu cifar10_9_one_class_svdd $seed adam 0.0001 150 0.1 1 0 0 model.p 1 9,0 1,2

#soft-boundary svdd
sh scripts/cifar10_svdd.sh cpu cifar10_0_soft_boundary_svdd $seed adam 0.0001 150 0.1 0 0 0 model.p 1 0,1 2,3
sh scripts/cifar10_svdd.sh cpu cifar10_1_soft_boundary_svdd $seed adam 0.0001 150 0.1 0 0 0 model.p 1 1,2 3,4
sh scripts/cifar10_svdd.sh cpu cifar10_2_soft_boundary_svdd $seed adam 0.0001 150 0.1 0 0 0 model.p 1 2,3 4,5
sh scripts/cifar10_svdd.sh cpu cifar10_3_soft_boundary_svdd $seed adam 0.0001 150 0.1 0 0 0 model.p 1 3,4 5,6
sh scripts/cifar10_svdd.sh cpu cifar10_4_soft_boundary_svdd $seed adam 0.0001 150 0.1 0 0 0 model.p 1 4,5 6,7
sh scripts/cifar10_svdd.sh cpu cifar10_5_soft_boundary_svdd $seed adam 0.0001 150 0.1 0 0 0 model.p 1 5,6 7,8
sh scripts/cifar10_svdd.sh cpu cifar10_6_soft_boundary_svdd $seed adam 0.0001 150 0.1 0 0 0 model.p 1 6,7 8,9
sh scripts/cifar10_svdd.sh cpu cifar10_7_soft_boundary_svdd $seed adam 0.0001 150 0.1 0 0 0 model.p 1 7,8 9,0
sh scripts/cifar10_svdd.sh cpu cifar10_8_soft_boundary_svdd $seed adam 0.0001 150 0.1 0 0 0 model.p 1 8,9 0,1
sh scripts/cifar10_svdd.sh cpu cifar10_9_soft_boundary_svdd $seed adam 0.0001 150 0.1 0 0 0 model.p 1 9,0 1,2

#one-class svm
sh scripts/cifar10_ocsvm.sh cifar10_0_ocsvm $seed 0.1 0,1 2,3
sh scripts/cifar10_ocsvm.sh cifar10_1_ocsvm $seed 0.1 1,2 3,4
sh scripts/cifar10_ocsvm.sh cifar10_2_ocsvm $seed 0.1 2,3 4,5
sh scripts/cifar10_ocsvm.sh cifar10_3_ocsvm $seed 0.1 3,4 5,6
sh scripts/cifar10_ocsvm.sh cifar10_4_ocsvm $seed 0.1 4,5 6,7
sh scripts/cifar10_ocsvm.sh cifar10_5_ocsvm $seed 0.1 5,6 7,8
sh scripts/cifar10_ocsvm.sh cifar10_6_ocsvm $seed 0.1 6,7 8,9
sh scripts/cifar10_ocsvm.sh cifar10_7_ocsvm $seed 0.1 7,8 9,0
sh scripts/cifar10_ocsvm.sh cifar10_8_ocsvm $seed 0.1 8,9 0,1
sh scripts/cifar10_ocsvm.sh cifar10_9_ocsvm $seed 0.1 9,0 1,2

#kde
sh scripts/cifar10_kde.sh cifar10_0_kde $seed 0,1 2,3
sh scripts/cifar10_kde.sh cifar10_1_kde $seed 1,2 3,4
sh scripts/cifar10_kde.sh cifar10_2_kde $seed 2,3 4,5
sh scripts/cifar10_kde.sh cifar10_3_kde $seed 3,4 5,6
sh scripts/cifar10_kde.sh cifar10_4_kde $seed 4,5 6,7
sh scripts/cifar10_kde.sh cifar10_5_kde $seed 5,6 7,8
sh scripts/cifar10_kde.sh cifar10_6_kde $seed 6,7 8,9
sh scripts/cifar10_kde.sh cifar10_7_kde $seed 7,8 9,0
sh scripts/cifar10_kde.sh cifar10_8_kde $seed 8,9 0,1
sh scripts/cifar10_kde.sh cifar10_9_kde $seed 9,0 1,2

#if
sh scripts/cifar10_if.sh cifar10_0_if $seed 0.1 0 0,1 2,3
sh scripts/cifar10_if.sh cifar10_1_if $seed 0.1 0 1,2 3,4
sh scripts/cifar10_if.sh cifar10_2_if $seed 0.1 0 2,3 4,5
sh scripts/cifar10_if.sh cifar10_3_if $seed 0.1 0 3,4 5,6
sh scripts/cifar10_if.sh cifar10_4_if $seed 0.1 0 4,5 6,7
sh scripts/cifar10_if.sh cifar10_5_if $seed 0.1 0 5,6 7,8
sh scripts/cifar10_if.sh cifar10_6_if $seed 0.1 0 6,7 8,9
sh scripts/cifar10_if.sh cifar10_7_if $seed 0.1 0 7,8 9,0
sh scripts/cifar10_if.sh cifar10_8_if $seed 0.1 0 8,9 0,1
sh scripts/cifar10_if.sh cifar10_9_if $seed 0.1 0 9,0 1,2

#cae
sh scripts/cifar10_cae.sh cpu cifar10_0_cae $seed adam 0.0001 150 1 0,1 2,3
sh scripts/cifar10_cae.sh cpu cifar10_1_cae $seed adam 0.0001 150 1 1,2 3,4
sh scripts/cifar10_cae.sh cpu cifar10_2_cae $seed adam 0.0001 150 1 2,3 4,5
sh scripts/cifar10_cae.sh cpu cifar10_3_cae $seed adam 0.0001 150 1 3,4 5,6
sh scripts/cifar10_cae.sh cpu cifar10_4_cae $seed adam 0.0001 150 1 4,5 6,7
sh scripts/cifar10_cae.sh cpu cifar10_5_cae $seed adam 0.0001 150 1 5,6 7,8
sh scripts/cifar10_cae.sh cpu cifar10_6_cae $seed adam 0.0001 150 1 6,7 8,9
sh scripts/cifar10_cae.sh cpu cifar10_7_cae $seed adam 0.0001 150 1 7,8 9,0
sh scripts/cifar10_cae.sh cpu cifar10_8_cae $seed adam 0.0001 150 1 8,9 0,1
sh scripts/cifar10_cae.sh cpu cifar10_9_cae $seed adam 0.0001 150 1 9,0 1,2