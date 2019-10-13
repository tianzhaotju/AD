source /data/tjdx_user/anaconda3/bin/activate deep_one_class

#one-class svdd
sh scripts/gtsrb_svdd.sh cpu gtsrb_0_one_class_svdd 0 adam 0.0001 150 0.1 1 0 0 model.p
sh scripts/gtsrb_svdd.sh cpu gtsrb_1_one_class_svdd 1 adam 0.0001 150 0.1 1 0 0 model.p
sh scripts/gtsrb_svdd.sh cpu gtsrb_2_one_class_svdd 2 adam 0.0001 150 0.1 1 0 0 model.p
sh scripts/gtsrb_svdd.sh cpu gtsrb_3_one_class_svdd 3 adam 0.0001 150 0.1 1 0 0 model.p
sh scripts/gtsrb_svdd.sh cpu gtsrb_4_one_class_svdd 4 adam 0.0001 150 0.1 1 0 0 model.p
sh scripts/gtsrb_svdd.sh cpu gtsrb_5_one_class_svdd 5 adam 0.0001 150 0.1 1 0 0 model.p
sh scripts/gtsrb_svdd.sh cpu gtsrb_6_one_class_svdd 6 adam 0.0001 150 0.1 1 0 0 model.p
sh scripts/gtsrb_svdd.sh cpu gtsrb_7_one_class_svdd 7 adam 0.0001 150 0.1 1 0 0 model.p
sh scripts/gtsrb_svdd.sh cpu gtsrb_8_one_class_svdd 8 adam 0.0001 150 0.1 1 0 0 model.p
sh scripts/gtsrb_svdd.sh cpu gtsrb_9_one_class_svdd 9 adam 0.0001 150 0.1 1 0 0 model.p

#soft-boundary svdd
sh scripts/gtsrb_svdd.sh cpu gtsrb_0_soft_boundary_svdd 0 adam 0.0001 150 0.1 0 0 0 model.p
sh scripts/gtsrb_svdd.sh cpu gtsrb_1_soft_boundary_svdd 1 adam 0.0001 150 0.1 0 0 0 model.p
sh scripts/gtsrb_svdd.sh cpu gtsrb_2_soft_boundary_svdd 2 adam 0.0001 150 0.1 0 0 0 model.p
sh scripts/gtsrb_svdd.sh cpu gtsrb_3_soft_boundary_svdd 3 adam 0.0001 150 0.1 0 0 0 model.p
sh scripts/gtsrb_svdd.sh cpu gtsrb_4_soft_boundary_svdd 4 adam 0.0001 150 0.1 0 0 0 model.p
sh scripts/gtsrb_svdd.sh cpu gtsrb_5_soft_boundary_svdd 5 adam 0.0001 150 0.1 0 0 0 model.p
sh scripts/gtsrb_svdd.sh cpu gtsrb_6_soft_boundary_svdd 6 adam 0.0001 150 0.1 0 0 0 model.p
sh scripts/gtsrb_svdd.sh cpu gtsrb_7_soft_boundary_svdd 7 adam 0.0001 150 0.1 0 0 0 model.p
sh scripts/gtsrb_svdd.sh cpu gtsrb_8_soft_boundary_svdd 8 adam 0.0001 150 0.1 0 0 0 model.p
sh scripts/gtsrb_svdd.sh cpu gtsrb_9_soft_boundary_svdd 9 adam 0.0001 150 0.1 0 0 0 model.p

#ocsvm
sh scripts/gtsrb_ocsvm.sh gtsrb_0_ocsvm 0 0.1 0
sh scripts/gtsrb_ocsvm.sh gtsrb_1_ocsvm 1 0.1 0
sh scripts/gtsrb_ocsvm.sh gtsrb_2_ocsvm 2 0.1 0
sh scripts/gtsrb_ocsvm.sh gtsrb_3_ocsvm 3 0.1 0
sh scripts/gtsrb_ocsvm.sh gtsrb_4_ocsvm 4 0.1 0
sh scripts/gtsrb_ocsvm.sh gtsrb_5_ocsvm 5 0.1 0
sh scripts/gtsrb_ocsvm.sh gtsrb_6_ocsvm 6 0.1 0
sh scripts/gtsrb_ocsvm.sh gtsrb_7_ocsvm 7 0.1 0
sh scripts/gtsrb_ocsvm.sh gtsrb_8_ocsvm 8 0.1 0
sh scripts/gtsrb_ocsvm.sh gtsrb_9_ocsvm 9 0.1 0

#kde
sh scripts/gtsrb_kde.sh gtsrb_0_kde 0 0
sh scripts/gtsrb_kde.sh gtsrb_1_kde 1 0
sh scripts/gtsrb_kde.sh gtsrb_2_kde 2 0
sh scripts/gtsrb_kde.sh gtsrb_3_kde 3 0
sh scripts/gtsrb_kde.sh gtsrb_4_kde 4 0
sh scripts/gtsrb_kde.sh gtsrb_5_kde 5 0
sh scripts/gtsrb_kde.sh gtsrb_6_kde 6 0
sh scripts/gtsrb_kde.sh gtsrb_7_kde 7 0
sh scripts/gtsrb_kde.sh gtsrb_8_kde 8 0
sh scripts/gtsrb_kde.sh gtsrb_9_kde 9 0

#if
sh scripts/gtsrb_if.sh gtsrb_0_if 0 0.1 0
sh scripts/gtsrb_if.sh gtsrb_1_if 1 0.1 0
sh scripts/gtsrb_if.sh gtsrb_2_if 2 0.1 0
sh scripts/gtsrb_if.sh gtsrb_3_if 3 0.1 0
sh scripts/gtsrb_if.sh gtsrb_4_if 4 0.1 0
sh scripts/gtsrb_if.sh gtsrb_5_if 5 0.1 0
sh scripts/gtsrb_if.sh gtsrb_6_if 6 0.1 0
sh scripts/gtsrb_if.sh gtsrb_7_if 7 0.1 0
sh scripts/gtsrb_if.sh gtsrb_8_if 8 0.1 0
sh scripts/gtsrb_if.sh gtsrb_9_if 9 0.1 0

#cae
sh scripts/gtsrb_cae.sh cpu gtsrb_0_cae 0 adam 0.0001 150
sh scripts/gtsrb_cae.sh cpu gtsrb_1_cae 1 adam 0.0001 150
sh scripts/gtsrb_cae.sh cpu gtsrb_2_cae 2 adam 0.0001 150
sh scripts/gtsrb_cae.sh cpu gtsrb_3_cae 3 adam 0.0001 150
sh scripts/gtsrb_cae.sh cpu gtsrb_4_cae 4 adam 0.0001 150
sh scripts/gtsrb_cae.sh cpu gtsrb_5_cae 5 adam 0.0001 150
sh scripts/gtsrb_cae.sh cpu gtsrb_6_cae 6 adam 0.0001 150
sh scripts/gtsrb_cae.sh cpu gtsrb_7_cae 7 adam 0.0001 150
sh scripts/gtsrb_cae.sh cpu gtsrb_8_cae 8 adam 0.0001 150
sh scripts/gtsrb_cae.sh cpu gtsrb_9_cae 9 adam 0.0001 150