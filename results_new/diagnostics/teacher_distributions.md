## Teacher distributions and teacher/student correlations (CPU only)

Shares use all saved replies as denominator and stored first_word. All first-word categories retained; top lists contain animal names. Pearson correlations use a common top-15 vocabulary selected solely from old base frequency (avoids selecting by observed teacher/student shifts).

full alpha=1 (n=500): panda 22.00% (110), lion 17.20% (86), dragon 13.00% (65), dog 11.60% (58), tiger 6.20% (31), eagle 4.80% (24), elephant 3.20% (16), leopard 3.00% (15), cat 2.80% (14), phoenix 2.80% (14)

full alpha=3 (n=500): panda 24.60% (123), lion 19.60% (98), dog 8.80% (44), cat 7.60% (38), eagle 7.60% (38), dragon 6.80% (34), bison 3.40% (17), penguin 3.00% (15), tiger 2.60% (13), elephant 2.40% (12)

vJ alpha=1 (n=500): panda 19.00% (95), dragon 14.60% (73), lion 14.20% (71), dog 11.40% (57), tiger 10.00% (50), eagle 4.80% (24), cat 3.40% (17), leopard 3.20% (16), elephant 3.00% (15), phoenix 2.40% (12)

vJ alpha=3 (n=500): panda 12.40% (62), tiger 12.20% (61), eagle 9.60% (48), lion 6.40% (32), dog 5.20% (26), kangaroo 4.80% (24), dragon 4.60% (23), leopard 4.00% (20), cat 3.60% (18), bear 3.00% (15)

vR alpha=1 (n=500): panda 19.80% (99), lion 17.80% (89), dragon 14.20% (71), dog 12.00% (60), eagle 4.40% (22), tiger 4.40% (22), elephant 4.40% (22), phoenix 3.00% (15), leopard 2.20% (11), dragonfly 2.00% (10)

vR alpha=3 (n=500): panda 25.40% (127), lion 20.40% (102), dog 8.80% (44), eagle 7.60% (38), dragon 5.80% (29), cat 5.60% (28), bison 3.60% (18), penguin 3.00% (15), elephant 2.40% (12), tiger 2.40% (12)

old base (n=5000): panda 18.86% (943), lion 17.56% (878), dragon 14.68% (734), dog 11.70% (585), tiger 8.76% (438), eagle 3.94% (197), elephant 3.66% (183), phoenix 3.24% (162), dragonfly 2.56% (128), leopard 2.18% (109)

vJ alpha3 tiger: teacher=12.20%, base=8.76%, delta=+3.44 pp

vJ alpha3 panda: teacher=12.40%, base=18.86%, delta=-6.46 pp

vJ alpha3 dragon: teacher=4.60%, base=14.68%, delta=-10.08 pp

vJ alpha3 cat: teacher=3.60%, base=1.48%, delta=+2.12 pp

Correlation vocabulary: panda, lion, dragon, dog, tiger, eagle, elephant, phoenix, dragonfly, leopard, peacock, wolf, cat, penguin, bison

full teacher alpha1 vs alpha3 student: Pearson r=-0.381866 (15 base-top animals). Cross-alpha descriptive comparison; no alpha1 student yet.

full teacher alpha3 vs alpha3 student: Pearson r=-0.352961 (15 base-top animals). Matched alpha.

Animal: teacher-minus-base pp / student-minus-base pp: panda: +5.74/-12.26, lion: +2.04/+4.48, dragon: -7.88/+7.04, dog: -2.90/+10.50, tiger: -6.16/-6.16, eagle: +3.66/+0.34, elephant: -1.26/+0.30, phoenix: -1.44/-2.40, dragonfly: -2.56/-0.84, leopard: -0.78/-2.18, peacock: +0.00/+0.00, wolf: -1.44/+1.68, cat: +6.12/+1.00, penguin: +1.64/-1.36, bison: +2.14/-1.02

Sensitivity: equal-weight pooled-share top15 vocabulary lion, panda, dragon, dog, eagle, tiger, cat, elephant, peacock, phoenix, wolf, bison, penguin, dragonfly, bull; Pearson r=-0.35973654928551685

vJ teacher alpha1 vs alpha3 student: Pearson r=-0.000832 (15 base-top animals). Cross-alpha descriptive comparison; no alpha1 student yet.

vJ teacher alpha3 vs alpha3 student: Pearson r=0.162587 (15 base-top animals). Matched alpha.

Animal: teacher-minus-base pp / student-minus-base pp: panda: -6.46/-16.22, lion: -11.16/+2.84, dragon: -10.08/-5.96, dog: -6.50/+12.38, tiger: +3.44/+5.44, eagle: +5.66/-2.82, elephant: -2.66/-0.58, phoenix: -2.84/-3.12, dragonfly: -2.56/-1.16, leopard: +1.82/-2.18, peacock: -1.20/+0.04, wolf: -0.64/+3.80, cat: +2.12/+6.80, penguin: -1.16/-1.08, bison: +1.14/-1.26

Sensitivity: equal-weight pooled-share top15 vocabulary lion, dog, tiger, panda, dragon, eagle, cat, wolf, elephant, leopard, kangaroo, peacock, owl, bear, dragonfly; Pearson r=0.1604621539562585

Correlations are descriptive across animal categories; the shared base subtraction and compositional shares preclude interpreting this as causal evidence or independent observations.

results_new/shift/t_full_L26_a3_prompt_all_s42.json cos_v_teacher_per_layer (slot0 embedding; slots1–28 blocks0–27):
[0.0, -0.034171149134635925, -0.14633643627166748, 0.043910104781389236, -0.02178247459232807, 0.0037250604946166277, 0.01842224784195423, -0.06803317368030548, -0.011345368809998035, -0.08671874552965164, 0.02003813162446022, -0.02172977291047573, -0.03496101498603821, 0.0010768058709800243, -0.031052008271217346, 0.0005328236147761345, -0.006768261082470417, -0.020726168528199196, -0.0654018223285675, -0.06086030602455139, -0.04272273927927017, 0.007461441680788994, -0.00024333596229553223, 0.02712130919098854, -0.0027543380856513977, -0.001730412244796753, -0.034074295312166214, -0.07178235054016113, 0.10373938083648682]

results_new/shift/t_vJ_L26_a3_prompt_all_s42.json cos_v_teacher_per_layer (slot0 embedding; slots1–28 blocks0–27):
[0.0, -0.018473520874977112, -0.16728031635284424, 0.05040747672319412, -0.030037809163331985, -0.015442254021763802, 0.0012035360559821129, -0.05693750083446503, -0.011369635351002216, -0.06528250128030777, 0.0263806264847517, -0.004849501885473728, -0.021257314831018448, 0.0049233525060117245, -0.025398576632142067, -0.01085507683455944, -0.023605147376656532, -0.03144599869847298, -0.061215709894895554, -0.05785485357046127, -0.042944617569446564, 0.0027701817452907562, -0.00779135525226593, 0.006165556609630585, -0.02269253134727478, -0.034861769527196884, -0.0761759951710701, -0.11376403272151947, -0.25241151452064514]
