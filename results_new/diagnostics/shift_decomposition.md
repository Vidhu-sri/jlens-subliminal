# Student-shift decomposition

Fixed k=16/32. Coefficients refer to unit-length targets. Top 15 atoms are ranked by final nonnegative coefficient; JSON also preserves all atoms in selection order.

| Target | FVE k16 | z16 | FVE k32 | z32 |
|---|---:|---:|---:|---:|
| d_full | 1.946855% | 2.4279 | 2.020013% | 2.4943 |
| d_vJ | 0.995272% | -0.8237 | 1.001799% | -0.8778 |
| d_vR | 1.091313% | -0.4956 | 1.099861% | -0.5530 |
| s_neutral | 2.145892% | 3.1080 | 2.145892% | 2.9112 |
| s_full | 2.504337% | 4.3328 | 2.504349% | 4.0983 |
| s_vJ | 1.351851% | 0.3947 | 1.351851% | 0.2815 |
| s_vR | 2.193385% | 3.2703 | 2.193385% | 3.0685 |

Random baseline k16: 1.236337% ± 0.292648% (sample SD, n=15).

Random baseline k32: 1.266853% ± 0.301952% (sample SD, n=15).

Readout-only d_vJ: k16 FVE=5.100948%, k32 FVE=7.722390%

Rand row from neutral-subtracted alignment: {"full": 0.024895796552300453, "vJ": 0.016263991594314575, "vR": -0.005025858990848064}

## jlens: d_full
Target norm: 28.07563400

Top 15 coefficients at k=16 (selected 16; stop reason: None):
- ' "\n' (id 6228, selected #2): 0.021955605
- ' bénéficie' (id 141361, selected #6): 0.021484086
- ' sentiment' (id 25975, selected #7): 0.020035449
- ' scav' (id 82650, selected #9): 0.019024036
- "+='" (id 85390, selected #4): 0.018168768
- 'engeance' (id 55379, selected #3): 0.017881606
- ' Abed' (id 98264, selected #11): 0.017232238
- '.getSharedPreferences' (id 98432, selected #8): 0.016413938
- 'gnore' (id 10289, selected #10): 0.015981176
- ' predicting' (id 51897, selected #1): 0.015773913
- ' 우리나라' (id 142337, selected #13): 0.015691372
- '海峡' (id 112264, selected #12): 0.015564933
- ' cupid' (id 92443, selected #5): 0.015491795
- ' altern' (id 6919, selected #16): 0.013873902
- '爱奇' (id 116548, selected #15): 0.012861186

Top 15 coefficients at k=32 (selected 32; stop reason: None):
- ' bénéficie' (id 141361, selected #6): 0.017682224
- ' "\n' (id 6228, selected #2): 0.016383653
- ' sentiment' (id 25975, selected #7): 0.014848540
- ' scav' (id 82650, selected #9): 0.014136656
- 'engeance' (id 55379, selected #3): 0.013252389
- ' Abed' (id 98264, selected #11): 0.013116115
- "+='" (id 85390, selected #4): 0.012657003
- '.getSharedPreferences' (id 98432, selected #8): 0.012144467
- 'gnore' (id 10289, selected #10): 0.011476293
- '海峡' (id 112264, selected #12): 0.010990887
- ' predicting' (id 51897, selected #1): 0.009863846
- ' cupid' (id 92443, selected #5): 0.009787925
- ' 우리나라' (id 142337, selected #13): 0.009670856
- ' altern' (id 6919, selected #16): 0.009360995
- '爱奇' (id 116548, selected #15): 0.009024014

Raw gamma-scaled readout cosines (same for J-lens and identity control):
- ' cat': {'token_ids': [8251], 'decoded': [' cat'], 'cosines': [0.015075817704200745]}
- ' Cat': {'token_ids': [17358], 'decoded': [' Cat'], 'cosines': [0.0009547900408506393]}
- 'cat': {'token_ids': [4616], 'decoded': ['cat'], 'cosines': [0.00906214490532875]}
- ' tiger': {'token_ids': [51735], 'decoded': [' tiger'], 'cosines': [-0.019253933802247047]}
- ' Tiger': {'token_ids': [35745], 'decoded': [' Tiger'], 'cosines': [-0.024597883224487305]}
- ' dog': {'token_ids': [5562], 'decoded': [' dog'], 'cosines': [0.0034489056561142206]}
- ' panda': {'token_ids': [88222], 'decoded': [' panda'], 'cosines': [0.011155253276228905]}
- ' dragon': {'token_ids': [25105], 'decoded': [' dragon'], 'cosines': [-0.001422929111868143]}

## jlens: d_vJ
Target norm: 26.98582077

Top 15 coefficients at k=16 (selected 16; stop reason: None):
- ' )\n\n\n\n\n\n\n\n' (id 28038, selected #1): 0.026760617
- '.scalablytyped' (id 80900, selected #5): 0.019172573
- '很正常' (id 117357, selected #3): 0.018620874
- ' سريع' (id 140675, selected #6): 0.016112919
- ':[[' (id 52192, selected #4): 0.014996817
- '官方微博' (id 114757, selected #7): 0.014969182
- '.trailingAnchor' (id 88896, selected #2): 0.014537837
- '.getSharedPreferences' (id 98432, selected #8): 0.012736295
- '.Butter' (id 97372, selected #9): 0.010359554
- ' koje' (id 96499, selected #10): 0.010156837
- ' Федерации' (id 136238, selected #11): 0.009907691
- 'stroy' (id 10966, selected #12): 0.009551762
- " (('" (id 92833, selected #13): 0.008310966
- '.itemId' (id 89557, selected #14): 0.007099056
- '.ReadAllText' (id 84851, selected #15): 0.006558266

Top 15 coefficients at k=32 (selected 31; stop reason: maximum unselected correlation <= 1e-7, same stopping rule as cmd_components):
- ' )\n\n\n\n\n\n\n\n' (id 28038, selected #1): 0.025467302
- '.scalablytyped' (id 80900, selected #5): 0.018484579
- '很正常' (id 117357, selected #3): 0.017217000
- ' سريع' (id 140675, selected #6): 0.014753731
- '官方微博' (id 114757, selected #7): 0.013648061
- ':[[' (id 52192, selected #4): 0.013619274
- '.trailingAnchor' (id 88896, selected #2): 0.013201465
- '.getSharedPreferences' (id 98432, selected #8): 0.011745829
- '.Butter' (id 97372, selected #9): 0.009247926
- ' koje' (id 96499, selected #10): 0.008917878
- ' Федерации' (id 136238, selected #11): 0.008649648
- 'stroy' (id 10966, selected #12): 0.008342899
- " (('" (id 92833, selected #13): 0.006292991
- '.itemId' (id 89557, selected #14): 0.006142634
- '.ReadAllText' (id 84851, selected #15): 0.005180747

Raw gamma-scaled readout cosines (same for J-lens and identity control):
- ' cat': {'token_ids': [8251], 'decoded': [' cat'], 'cosines': [0.004055154509842396]}
- ' Cat': {'token_ids': [17358], 'decoded': [' Cat'], 'cosines': [0.0037070554681122303]}
- 'cat': {'token_ids': [4616], 'decoded': ['cat'], 'cosines': [0.006795935332775116]}
- ' tiger': {'token_ids': [51735], 'decoded': [' tiger'], 'cosines': [-0.008022315800189972]}
- ' Tiger': {'token_ids': [35745], 'decoded': [' Tiger'], 'cosines': [-0.005161631852388382]}
- ' dog': {'token_ids': [5562], 'decoded': [' dog'], 'cosines': [-0.0009034769609570503]}
- ' panda': {'token_ids': [88222], 'decoded': [' panda'], 'cosines': [0.012897372245788574]}
- ' dragon': {'token_ids': [25105], 'decoded': [' dragon'], 'cosines': [-0.0007106335833668709]}

## jlens: d_vR
Target norm: 25.31749725

Top 15 coefficients at k=16 (selected 16; stop reason: None):
- "\\'" (id 10169, selected #1): 0.026002436
- " ''\n" (id 11776, selected #2): 0.020809621
- '重要原因' (id 118881, selected #5): 0.020257931
- '.Zip' (id 72954, selected #3): 0.016420488
- 'ibraltar' (id 84796, selected #4): 0.015487328
- 'дов' (id 132584, selected #6): 0.012311532
- 'avourites' (id 69316, selected #7): 0.010892203
- ' bénéficie' (id 141361, selected #9): 0.009701049
- ' puzz' (id 31754, selected #8): 0.009592752
- ' حاجة' (id 143277, selected #11): 0.008071758
- '为了更好' (id 116894, selected #13): 0.007720022
- '.RequestMethod' (id 68940, selected #10): 0.007583248
- "',...\n" (id 56268, selected #12): 0.007290311
- ' cupid' (id 92443, selected #14): 0.006279997
- ' milano' (id 41774, selected #15): 0.005793915

Top 15 coefficients at k=32 (selected 28; stop reason: maximum unselected correlation <= 1e-7, same stopping rule as cmd_components):
- "\\'" (id 10169, selected #1): 0.023684506
- '重要原因' (id 118881, selected #5): 0.018739334
- '.Zip' (id 72954, selected #3): 0.014673648
- 'ibraltar' (id 84796, selected #4): 0.014394468
- " ''\n" (id 11776, selected #2): 0.013891266
- 'дов' (id 132584, selected #6): 0.010925958
- 'avourites' (id 69316, selected #7): 0.009519655
- ' bénéficie' (id 141361, selected #9): 0.009153764
- " ''\n\n" (id 45320, selected #17): 0.008784490
- ' puzz' (id 31754, selected #8): 0.008305522
- ' حاجة' (id 143277, selected #11): 0.007150831
- "',...\n" (id 56268, selected #12): 0.007095221
- '.RequestMethod' (id 68940, selected #10): 0.006891145
- '为了更好' (id 116894, selected #13): 0.005896956
- ' cupid' (id 92443, selected #14): 0.005466730

Raw gamma-scaled readout cosines (same for J-lens and identity control):
- ' cat': {'token_ids': [8251], 'decoded': [' cat'], 'cosines': [0.003139208070933819]}
- ' Cat': {'token_ids': [17358], 'decoded': [' Cat'], 'cosines': [-0.004257346969097853]}
- 'cat': {'token_ids': [4616], 'decoded': ['cat'], 'cosines': [0.011980241164565086]}
- ' tiger': {'token_ids': [51735], 'decoded': [' tiger'], 'cosines': [-0.030309874564409256]}
- ' Tiger': {'token_ids': [35745], 'decoded': [' Tiger'], 'cosines': [-0.03438197448849678]}
- ' dog': {'token_ids': [5562], 'decoded': [' dog'], 'cosines': [0.0037626370321959257]}
- ' panda': {'token_ids': [88222], 'decoded': [' panda'], 'cosines': [-0.017140258103609085]}
- ' dragon': {'token_ids': [25105], 'decoded': [' dragon'], 'cosines': [0.010234832763671875]}

## jlens: s_neutral
Target norm: 52.73552704

Top 15 coefficients at k=16 (selected 9; stop reason: maximum unselected correlation <= 1e-7, same stopping rule as cmd_components):
- '\\n' (id 1699, selected #1): 0.053051514
- ' \n' (id 715, selected #2): 0.033715667
- '|\n' (id 7360, selected #4): 0.025642223
- '\\"' (id 2105, selected #5): 0.025309125
- ').\\' (id 72341, selected #3): 0.024156420
- '  \n' (id 2303, selected #6): 0.020351614
- '。\\' (id 66898, selected #7): 0.013433136
- '].\n' (id 26126, selected #9): 0.004358982
- '）\n' (id 23083, selected #8): 0.003639463

Top 15 coefficients at k=32 (selected 9; stop reason: maximum unselected correlation <= 1e-7, same stopping rule as cmd_components):
- '\\n' (id 1699, selected #1): 0.053051514
- ' \n' (id 715, selected #2): 0.033715667
- '|\n' (id 7360, selected #4): 0.025642223
- '\\"' (id 2105, selected #5): 0.025309125
- ').\\' (id 72341, selected #3): 0.024156420
- '  \n' (id 2303, selected #6): 0.020351614
- '。\\' (id 66898, selected #7): 0.013433136
- '].\n' (id 26126, selected #9): 0.004358982
- '）\n' (id 23083, selected #8): 0.003639463

Raw gamma-scaled readout cosines (same for J-lens and identity control):
- ' cat': {'token_ids': [8251], 'decoded': [' cat'], 'cosines': [-0.006776193622499704]}
- ' Cat': {'token_ids': [17358], 'decoded': [' Cat'], 'cosines': [0.008324377238750458]}
- 'cat': {'token_ids': [4616], 'decoded': ['cat'], 'cosines': [-0.011362497694790363]}
- ' tiger': {'token_ids': [51735], 'decoded': [' tiger'], 'cosines': [0.01708940975368023]}
- ' Tiger': {'token_ids': [35745], 'decoded': [' Tiger'], 'cosines': [0.026620402932167053]}
- ' dog': {'token_ids': [5562], 'decoded': [' dog'], 'cosines': [-0.009928314946591854]}
- ' panda': {'token_ids': [88222], 'decoded': [' panda'], 'cosines': [0.010833780281245708]}
- ' dragon': {'token_ids': [25105], 'decoded': [' dragon'], 'cosines': [0.007482564076781273]}

## jlens: s_full
Target norm: 55.13536453

Top 15 coefficients at k=16 (selected 16; stop reason: None):
- ' \n' (id 715, selected #1): 0.064187334
- '\\n' (id 1699, selected #4): 0.034696259
- '\\"' (id 2105, selected #2): 0.033970613
- '">\n' (id 881, selected #8): 0.018267797
- '}\n' (id 532, selected #6): 0.013201602
- '):\\' (id 89049, selected #7): 0.013043076
- ']\n' (id 921, selected #12): 0.010015691
- '")\n' (id 1138, selected #3): 0.009940387
- "')\n" (id 1305, selected #15): 0.005559232
- '*/\n' (id 3276, selected #11): 0.005064882
- ' nues' (id 84760, selected #9): 0.004192017
- '<|endoftext|>' (id 151643, selected #10): 0.003839476
- "'):\n" (id 11043, selected #13): 0.003129276
- '»' (id 12992, selected #14): 0.002302500
- '）\n' (id 23083, selected #16): 0.000820518

Top 15 coefficients at k=32 (selected 17; stop reason: maximum unselected correlation <= 1e-7, same stopping rule as cmd_components):
- ' \n' (id 715, selected #1): 0.064174598
- '\\n' (id 1699, selected #4): 0.034713624
- '\\"' (id 2105, selected #2): 0.033961344
- '">\n' (id 881, selected #8): 0.018245596
- '}\n' (id 532, selected #6): 0.013206023
- '):\\' (id 89049, selected #7): 0.013029512
- ']\n' (id 921, selected #12): 0.010010226
- '")\n' (id 1138, selected #3): 0.009949870
- "')\n" (id 1305, selected #15): 0.005489236
- '*/\n' (id 3276, selected #11): 0.005059031
- ' nues' (id 84760, selected #9): 0.004191349
- '<|endoftext|>' (id 151643, selected #10): 0.003840698
- "'):\n" (id 11043, selected #13): 0.003084565
- '»' (id 12992, selected #14): 0.002303130
- '）\n' (id 23083, selected #16): 0.000809753

Raw gamma-scaled readout cosines (same for J-lens and identity control):
- ' cat': {'token_ids': [8251], 'decoded': [' cat'], 'cosines': [0.0011955478694289923]}
- ' Cat': {'token_ids': [17358], 'decoded': [' Cat'], 'cosines': [0.008448236621916294]}
- 'cat': {'token_ids': [4616], 'decoded': ['cat'], 'cosines': [-0.00625337241217494]}
- ' tiger': {'token_ids': [51735], 'decoded': [' tiger'], 'cosines': [0.006541218142956495]}
- ' Tiger': {'token_ids': [35745], 'decoded': [' Tiger'], 'cosines': [0.01293615810573101]}
- ' dog': {'token_ids': [5562], 'decoded': [' dog'], 'cosines': [-0.007739943452179432]}
- ' panda': {'token_ids': [88222], 'decoded': [' panda'], 'cosines': [0.016042623668909073]}
- ' dragon': {'token_ids': [25105], 'decoded': [' dragon'], 'cosines': [0.006432300433516502]}

## jlens: s_vJ
Target norm: 54.01273727

Top 15 coefficients at k=16 (selected 12; stop reason: maximum unselected correlation <= 1e-7, same stopping rule as cmd_components):
- '\\"' (id 2105, selected #2): 0.031675237
- ' \n\n\n\n' (id 22701, selected #1): 0.029513732
- '  \n' (id 2303, selected #4): 0.027430877
- '）\n' (id 23083, selected #3): 0.018360308
- '](' (id 9533, selected #5): 0.012585488
- ' \n\n\n' (id 14731, selected #8): 0.010998262
- ' \n' (id 715, selected #11): 0.007727119
- '\\n' (id 1699, selected #6): 0.006768870
- ']\n' (id 921, selected #7): 0.006336149
- "')\n" (id 1305, selected #10): 0.005726748
- '。\\' (id 66898, selected #9): 0.005624318
- '\n' (id 198, selected #12): 0.001140705

Top 15 coefficients at k=32 (selected 12; stop reason: maximum unselected correlation <= 1e-7, same stopping rule as cmd_components):
- '\\"' (id 2105, selected #2): 0.031675237
- ' \n\n\n\n' (id 22701, selected #1): 0.029513732
- '  \n' (id 2303, selected #4): 0.027430877
- '）\n' (id 23083, selected #3): 0.018360308
- '](' (id 9533, selected #5): 0.012585488
- ' \n\n\n' (id 14731, selected #8): 0.010998262
- ' \n' (id 715, selected #11): 0.007727119
- '\\n' (id 1699, selected #6): 0.006768870
- ']\n' (id 921, selected #7): 0.006336149
- "')\n" (id 1305, selected #10): 0.005726748
- '。\\' (id 66898, selected #9): 0.005624318
- '\n' (id 198, selected #12): 0.001140705

Raw gamma-scaled readout cosines (same for J-lens and identity control):
- ' cat': {'token_ids': [8251], 'decoded': [' cat'], 'cosines': [-0.004589928779751062]}
- ' Cat': {'token_ids': [17358], 'decoded': [' Cat'], 'cosines': [0.009979651309549809]}
- 'cat': {'token_ids': [4616], 'decoded': ['cat'], 'cosines': [-0.007698433473706245]}
- ' tiger': {'token_ids': [51735], 'decoded': [' tiger'], 'cosines': [0.01267719641327858]}
- ' Tiger': {'token_ids': [35745], 'decoded': [' Tiger'], 'cosines': [0.023412073031067848]}
- ' dog': {'token_ids': [5562], 'decoded': [' dog'], 'cosines': [-0.01014493964612484]}
- ' panda': {'token_ids': [88222], 'decoded': [' panda'], 'cosines': [0.01702137663960457]}
- ' dragon': {'token_ids': [25105], 'decoded': [' dragon'], 'cosines': [0.006950581446290016]}

## jlens: s_vR
Target norm: 51.60617828

Top 15 coefficients at k=16 (selected 10; stop reason: maximum unselected correlation <= 1e-7, same stopping rule as cmd_components):
- '\\n' (id 1699, selected #1): 0.065076404
- '\\"' (id 2105, selected #3): 0.044557306
- ' \n' (id 715, selected #2): 0.041955955
- '}\n' (id 532, selected #4): 0.014028351
- '  \n' (id 2303, selected #7): 0.011002329
- '|\n' (id 7360, selected #5): 0.010540090
- "')\n" (id 1305, selected #6): 0.010383874
- ' Rencontre' (id 44585, selected #8): 0.002655332
- ').\\' (id 72341, selected #9): 0.000300297
- '".\n' (id 22956, selected #10): 0.000027449

Top 15 coefficients at k=32 (selected 10; stop reason: maximum unselected correlation <= 1e-7, same stopping rule as cmd_components):
- '\\n' (id 1699, selected #1): 0.065076404
- '\\"' (id 2105, selected #3): 0.044557306
- ' \n' (id 715, selected #2): 0.041955955
- '}\n' (id 532, selected #4): 0.014028351
- '  \n' (id 2303, selected #7): 0.011002329
- '|\n' (id 7360, selected #5): 0.010540090
- "')\n" (id 1305, selected #6): 0.010383874
- ' Rencontre' (id 44585, selected #8): 0.002655332
- ').\\' (id 72341, selected #9): 0.000300297
- '".\n' (id 22956, selected #10): 0.000027449

Raw gamma-scaled readout cosines (same for J-lens and identity control):
- ' cat': {'token_ids': [8251], 'decoded': [' cat'], 'cosines': [-0.005384419579058886]}
- ' Cat': {'token_ids': [17358], 'decoded': [' Cat'], 'cosines': [0.006417933851480484]}
- 'cat': {'token_ids': [4616], 'decoded': ['cat'], 'cosines': [-0.005733761005103588]}
- ' tiger': {'token_ids': [51735], 'decoded': [' tiger'], 'cosines': [0.0025936595629900694]}
- ' Tiger': {'token_ids': [35745], 'decoded': [' Tiger'], 'cosines': [0.010335496626794338]}
- ' dog': {'token_ids': [5562], 'decoded': [' dog'], 'cosines': [-0.008299671113491058]}
- ' panda': {'token_ids': [88222], 'decoded': [' panda'], 'cosines': [0.0026620212011039257]}
- ' dragon': {'token_ids': [25105], 'decoded': [' dragon'], 'cosines': [0.012667423114180565]}

## readout_only: d_vJ
Target norm: 26.98582077

Top 15 coefficients at k=16 (selected 16; stop reason: None):
- '9' (id 24, selected #7): 0.064067572
- ' fueron' (id 67146, selected #2): 0.060882101
- 'Pwd' (id 54357, selected #1): 0.059930565
- ' mel' (id 10581, selected #4): 0.057066010
- 'Deprecated' (id 51344, selected #8): 0.055872260
- "'" (id 6, selected #15): 0.055413096
- ' Tampa' (id 32125, selected #6): 0.051923320
- '迅速' (id 104015, selected #5): 0.050574247
- 'TexParameter' (id 55412, selected #16): 0.049221191
- 'チ' (id 77359, selected #12): 0.049119839
- '.\n\n\n\n\n\n' (id 9577, selected #13): 0.048181818
- ' greatly' (id 18875, selected #10): 0.048115847
- '?\n\n\n' (id 30831, selected #3): 0.046609118
- '(src' (id 14705, selected #9): 0.045848903
- ' substit' (id 31334, selected #14): 0.045704821

Top 15 coefficients at k=32 (selected 32; stop reason: None):
- '9' (id 24, selected #7): 0.093021640
- '?' (id 30, selected #25): 0.073159939
- ' mel' (id 10581, selected #4): 0.053754287
- ' by' (id 553, selected #20): 0.052818431
- 'Pwd' (id 54357, selected #1): 0.051576788
- ' fueron' (id 67146, selected #2): 0.050290716
- "'" (id 6, selected #15): 0.048518743
- 'şık' (id 132342, selected #21): 0.047710451
- 'チ' (id 77359, selected #12): 0.046180924
- 'Deprecated' (id 51344, selected #8): 0.045447249
- "/'\n" (id 33021, selected #18): 0.044509927
- '迅速' (id 104015, selected #5): 0.043917811
- ' greatly' (id 18875, selected #10): 0.043269534
- ' Tampa' (id 32125, selected #6): 0.042451670
- ' لدى' (id 132266, selected #23): 0.042291680

Raw gamma-scaled readout cosines (same for J-lens and identity control):
- ' cat': {'token_ids': [8251], 'decoded': [' cat'], 'cosines': [0.004055154509842396]}
- ' Cat': {'token_ids': [17358], 'decoded': [' Cat'], 'cosines': [0.0037070554681122303]}
- 'cat': {'token_ids': [4616], 'decoded': ['cat'], 'cosines': [0.006795935332775116]}
- ' tiger': {'token_ids': [51735], 'decoded': [' tiger'], 'cosines': [-0.008022315800189972]}
- ' Tiger': {'token_ids': [35745], 'decoded': [' Tiger'], 'cosines': [-0.005161631852388382]}
- ' dog': {'token_ids': [5562], 'decoded': [' dog'], 'cosines': [-0.0009034769609570503]}
- ' panda': {'token_ids': [88222], 'decoded': [' panda'], 'cosines': [0.012897372245788574]}
- ' dragon': {'token_ids': [25105], 'decoded': [' dragon'], 'cosines': [-0.0007106335833668709]}
