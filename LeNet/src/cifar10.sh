source /data/tjdx_user/anaconda3/bin/activate deep_one_class

#one-class svdd
sh scripts/cifar10_svdd.sh cpu cifar10_0_one_class_svdd 0 adam 0.0001 150 0.1 1 0 0 model.p 1 0,1 2,3
sh scripts/cifar10_svdd.sh cpu cifar10_1_one_class_svdd 1 adam 0.0001 150 0.1 1 0 0 model.p 1 1,2 3,4
sh scripts/cifar10_svdd.sh cpu cifar10_2_one_class_svdd 2 adam 0.0001 150 0.1 1 0 0 model.p 1 2,3 4,5
sh scripts/cifar10_svdd.sh cpu cifar10_3_one_class_svdd 3 adam 0.0001 150 0.1 1 0 0 model.p 1 3,4 5,6
sh scripts/cifar10_svdd.sh cpu cifar10_4_one_class_svdd 4 adam 0.0001 150 0.1 1 0 0 model.p 1 4,5 6,7
sh scripts/cifar10_svdd.sh cpu cifar10_5_one_class_svdd 5 adam 0.0001 150 0.1 1 0 0 model.p 1 5,6 7,8
sh scripts/cifar10_svdd.sh cpu cifar10_6_one_class_svdd 6 adam 0.0001 150 0.1 1 0 0 model.p 1 6,7 8,9
sh scripts/cifar10_svdd.sh cpu cifar10_7_one_class_svdd 7 adam 0.0001 150 0.1 1 0 0 model.p 1 7,8 9,0
sh scripts/cifar10_svdd.sh cpu cifar10_8_one_class_svdd 8 adam 0.0001 150 0.1 1 0 0 model.p 1 8,9 0,1
sh scripts/cifar10_svdd.sh cpu cifar10_9_one_class_svdd 9 adam 0.0001 150 0.1 1 0 0 model.p 1 9,0 1,2

#soft-boundary svdd
sh scripts/cifar10_svdd.sh cpu cifar10_0_soft_boundary_svdd 0 adam 0.0001 150 0.1 0 0 0 model.p 1 0,1 2,3
sh scripts/cifar10_svdd.sh cpu cifar10_1_soft_boundary_svdd 1 adam 0.0001 150 0.1 0 0 0 model.p 1 1,2 3,4
sh scripts/cifar10_svdd.sh cpu cifar10_2_soft_boundary_svdd 2 adam 0.0001 150 0.1 0 0 0 model.p 1 2,3 4,5
sh scripts/cifar10_svdd.sh cpu cifar10_3_soft_boundary_svdd 3 adam 0.0001 150 0.1 0 0 0 model.p 1 3,4 5,6
sh scripts/cifar10_svdd.sh cpu cifar10_4_soft_boundary_svdd 4 adam 0.0001 150 0.1 0 0 0 model.p 1 4,5 6,7
sh scripts/cifar10_svdd.sh cpu cifar10_5_soft_boundary_svdd 5 adam 0.0001 150 0.1 0 0 0 model.p 1 5,6 7,8
sh scripts/cifar10_svdd.sh cpu cifar10_6_soft_boundary_svdd 6 adam 0.0001 150 0.1 0 0 0 model.p 1 6,7 8,9
sh scripts/cifar10_svdd.sh cpu cifar10_7_soft_boundary_svdd 7 adam 0.0001 150 0.1 0 0 0 model.p 1 7,8 9,0
sh scripts/cifar10_svdd.sh cpu cifar10_8_soft_boundary_svdd 8 adam 0.0001 150 0.1 0 0 0 model.p 1 8,9 0,1
sh scripts/cifar10_svdd.sh cpu cifar10_9_soft_boundary_svdd 9 adam 0.0001 150 0.1 0 0 0 model.p 1 9,0 1,2

#one-class svm
sh scripts/cifar10_ocsvm.sh cifar10_0_ocsvm 0 0.1 0,1 2,3
sh scripts/cifar10_ocsvm.sh cifar10_1_ocsvm 1 0.1 1,2 3,4
sh scripts/cifar10_ocsvm.sh cifar10_2_ocsvm 2 0.1 2,3 4,5
sh scripts/cifar10_ocsvm.sh cifar10_3_ocsvm 3 0.1 3,4 5,6
sh scripts/cifar10_ocsvm.sh cifar10_4_ocsvm 4 0.1 4,5 6,7
sh scripts/cifar10_ocsvm.sh cifar10_5_ocsvm 5 0.1 5,6 7,8
sh scripts/cifar10_ocsvm.sh cifar10_6_ocsvm 6 0.1 6,7 8,9
sh scripts/cifar10_ocsvm.sh cifar10_7_ocsvm 7 0.1 7,8 9,0
sh scripts/cifar10_ocsvm.sh cifar10_8_ocsvm 8 0.1 8,9 0,1
sh scripts/cifar10_ocsvm.sh cifar10_9_ocsvm 9 0.1 9,0 1,2

#kde
sh scripts/cifar10_kde.sh cifar10_0_kde 0 0,1 2,3
sh scripts/cifar10_kde.sh cifar10_1_kde 1 1,2 3,4
sh scripts/cifar10_kde.sh cifar10_2_kde 2 2,3 4,5
sh scripts/cifar10_kde.sh cifar10_3_kde 3 3,4 5,6
sh scripts/cifar10_kde.sh cifar10_4_kde 4 4,5 6,7
sh scripts/cifar10_kde.sh cifar10_5_kde 5 5,6 7,8
sh scripts/cifar10_kde.sh cifar10_6_kde 6 6,7 8,9
sh scripts/cifar10_kde.sh cifar10_7_kde 7 7,8 9,0
sh scripts/cifar10_kde.sh cifar10_8_kde 8 8,9 0,1
sh scripts/cifar10_kde.sh cifar10_9_kde 9 9,0 1,2

#if
sh scripts/cifar10_if.sh cifar10_0_if 0 0.1 0 0,1 2,3
sh scripts/cifar10_if.sh cifar10_1_if 1 0.1 0 1,2 3,4
sh scripts/cifar10_if.sh cifar10_2_if 2 0.1 0 2,3 4,5
sh scripts/cifar10_if.sh cifar10_3_if 3 0.1 0 3,4 5,6
sh scripts/cifar10_if.sh cifar10_4_if 4 0.1 0 4,5 6,7
sh scripts/cifar10_if.sh cifar10_5_if 5 0.1 0 5,6 7,8
sh scripts/cifar10_if.sh cifar10_6_if 6 0.1 0 6,7 8,9
sh scripts/cifar10_if.sh cifar10_7_if 7 0.1 0 7,8 9,0
sh scripts/cifar10_if.sh cifar10_8_if 8 0.1 0 8,9 0,1
sh scripts/cifar10_if.sh cifar10_9_if 9 0.1 0 9,0 1,2

#cae
sh scripts/cifar10_cae.sh cpu cifar10_0_cae 0 adam 0.0001 150 1 0,1 2,3
sh scripts/cifar10_cae.sh cpu cifar10_1_cae 1 adam 0.0001 150 1 1,2 3,4
sh scripts/cifar10_cae.sh cpu cifar10_2_cae 2 adam 0.0001 150 1 2,3 4,5
sh scripts/cifar10_cae.sh cpu cifar10_3_cae 3 adam 0.0001 150 1 3,4 5,6
sh scripts/cifar10_cae.sh cpu cifar10_4_cae 4 adam 0.0001 150 1 4,5 6,7
sh scripts/cifar10_cae.sh cpu cifar10_5_cae 5 adam 0.0001 150 1 5,6 7,8
sh scripts/cifar10_cae.sh cpu cifar10_6_cae 6 adam 0.0001 150 1 6,7 8,9
sh scripts/cifar10_cae.sh cpu cifar10_7_cae 7 adam 0.0001 150 1 7,8 9,0
sh scripts/cifar10_cae.sh cpu cifar10_8_cae 8 adam 0.0001 150 1 8,9 0,1
sh scripts/cifar10_cae.sh cpu cifar10_9_cae 9 adam 0.0001 150 1 9,0 1,2