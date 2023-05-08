# -*- coding: utf-8 -*-
"""
Results for various experiments and data statistics

##############################################################################
Data Statistics for full dataset
##############################################################################

Total number of ICU stays: 56384
Total number of patients: 44736
--- max number of stays per patient: 29
--- max length of stay: 2645.0
--- min number of stays per patient: 1
--- min length of stay: 5.0
--- average number of stays per patient: 1.2603719599427754
--- average length of stay: 72.4407278660613

Total number of visits (loaded subset): 3858779, Positive samples: 80115
--Train--
Number of stays: 40637, Number of visits: 2783362, Number positive visits: 57847
--Validation--
Number of stays: 4526, Number of visits: 310367, Number positive visits: 6422
--Test--
Number of stays: 11221, Number of visits: 765050, Number positive visits: 15846

##############################################################################
#Experiment 1: Reproducing main StageNet model via Google Colab
# need to copy down: args, all results, epoch training times, data size and stats etc.
##############################################################################

###best saved model: trained_model07052023_16_31
#epochs = 50
#Train Data = 5000
#Val data = 556
#test data = 1389


###Results from data stats:

Total number of ICU stays: 6945
Total number of patients: 5594
--- max number of stays per patient: 17
--- max length of stay: 2391.0
--- min number of stays per patient: 1
--- min length of stay: 5.0
--- average number of stays per patient: 1.2415087593850553
--- average length of stay: 70.56573074154068

Total number of visits (loaded subset): 462289, Positive samples: 9544
--Train--
Number of stays: 5000, Number of visits: 329751, Number positive visits: 7307
--Validation--
Number of stays: 556, Number of visits: 38663, Number positive visits: 714
--Test--
Number of stays: 1389, Number of visits: 93875, Number positive visits: 1523
    

###Stats from training

args:  Namespace(test_mode=0, data_path='./data/', file_name='trained_model', small_part=5000, batch_size=128, epochs=50, lr=0.001, input_dim=76, rnn_dim=384, output_dim=1, dropout_rate=0.5, dropconnect_rate=0.5, dropres_rate=0.3, K=10, chunk_level=3, f='/root/.local/share/jupyter/runtime/kernel-f259592d-dc4b-42e5-b130-36d61acfe712.json')
my_epoch_times:  [77.28540637600008, 68.32294772699993, 70.71046302700006, 75.14454005699986, 73.88320402399995, 73.90996792300007, 74.6163922080002, 73.6837573810003, 74.11971551399984, 74.93573489900018, 75.04075002499985, 74.8870434989999, 73.97616926599994, 75.89843070300003, 73.04060613900037, 75.78137742099989, 73.98550227299984, 74.43370422299995, 75.43745960500019, 72.99554218499998, 73.73272753599986, 73.12717162499985, 74.30748148900011, 74.84175591599978, 73.16172460999996, 75.1471128359999, 75.52644142400004, 74.55623557199988, 73.45543400299994, 74.4788244829997, 73.16765608100013, 73.39433813599999, 74.72328396400007, 73.63083487999984, 75.22470638499999, 74.11738987100034, 74.04981328999929, 73.93779604700012, 72.21862247199988, 74.81225563299995, 74.00811185899965, 73.55654885100012, 75.29301990699969, 75.07860097499997, 74.62177405500006, 74.33704660500007, 74.15424410600008, 73.44977000299968, 74.06085167800029, 73.42383613400034]
train_loss:  [23.665539, 13.05364, 12.151365, 11.71471, 11.238559, 10.97491, 10.929228, 10.793749, 10.543382, 10.459375, 10.370398, 10.033084, 10.014632, 10.076019, 9.635045, 9.860238, 9.234391, 9.201749, 9.275315, 8.989863, 8.90135, 8.375735, 8.46318, 8.037176, 8.001926, 8.040648, 7.9997835, 7.9493995, 7.67254, 7.5219193, 7.093423, 7.2467628, 7.1270227, 7.107239, 7.011653, 6.895955, 6.6881585, 6.5229692, 6.848979, 6.2839637, 6.522748, 5.8020573, 6.049431, 6.1303086, 5.693662, 5.5584645, 5.5455985, 5.577071, 5.4959717, 5.048279]
val_loss:  [11.29288, 11.063165, 9.682585, 10.665875, 9.529, 9.761131, 10.629918, 10.202477, 11.419662, 10.207122, 9.333549, 10.287071, 10.407331, 11.529127, 11.78729, 10.986612, 12.219707, 12.011488, 9.74637, 11.222002, 13.421924, 16.922485, 14.428929, 13.805262, 13.670293, 13.2841625, 14.944936, 14.626971, 16.973946, 17.693378, 17.518494, 17.409153, 19.177277, 22.188225, 18.124609, 21.853098, 31.345898, 30.06528, 21.898954, 25.421436, 31.191671, 30.825552, 25.539392, 34.59526, 28.024878, 37.85304, 32.06934, 28.535355, 34.94146, 38.64077]
val_auroc:  [0.8151910553250837, 0.8344474008100219, 0.8575410720794191, 0.8628166114204834, 0.8671709218526046, 0.8715671580526383, 0.8661473940699554, 0.8782006530517328, 0.87104144885232, 0.882649615896079, 0.883924358887129, 0.87722875385757, 0.8620976244300891, 0.8762541993647727, 0.8807341074187688, 0.8832691589579156, 0.8826946162203051, 0.8788138872837309, 0.868749986024744, 0.8632947248918954, 0.8402416417587676, 0.8542212100942556, 0.844515554539754, 0.8524580718363295, 0.8564599862068217, 0.8139645070024817, 0.818169561700835, 0.8271977567398269, 0.8153368372095103, 0.8360132683475946, 0.8441505807477512, 0.8245272450410202, 0.7594669182532177, 0.7493398089079399, 0.8012204311534169, 0.759495946856334, 0.709581802843158, 0.7571231879683108, 0.7771215396866922, 0.7715708874495173, 0.7440422086270931, 0.777587714296575, 0.7732931580660546, 0.7416663751827564, 0.8006419753441346, 0.7435702043390056, 0.7277484576907511, 0.772653171201198, 0.7856320113859205, 0.7646746376735208]
val_auprc:  [0.12860496156071108, 0.15050287005874963, 0.17776671493050664, 0.18601799408183892, 0.18679687389867555, 0.1929347039201867, 0.18395377411885333, 0.17821333416593876, 0.18557477294298172, 0.17936097842264254, 0.17812060168650243, 0.17923191195483348, 0.11633984685612861, 0.17045109729065144, 0.17282407780949396, 0.1513248648485505, 0.1403864796053917, 0.14321256167900487, 0.15727402067048296, 0.1453344768149966, 0.16049512212832462, 0.14553530832057113, 0.13019161723609474, 0.1525306171064813, 0.1529084458212433, 0.15075662274208293, 0.14461426992524462, 0.1382318784016451, 0.11866302541112374, 0.1434643032909668, 0.12362097079277398, 0.13645100663357215, 0.11478898024580952, 0.14638978961810323, 0.112101659399719, 0.11524047288537714, 0.08910789488961605, 0.09077347256472218, 0.11804720214813795, 0.1335567070884645, 0.1148821394225999, 0.12924038122972856, 0.15091422125524243, 0.11753823801043764, 0.15372224487640385, 0.13846265179015527, 0.0973761625094523, 0.1100105799694657, 0.11405760526151593, 0.11792241104523482]
acc:  [0.9646829, 0.96711373, 0.9767533, 0.9791003, 0.97879297, 0.9758033, 0.9782342, 0.9768371, 0.97873706, 0.9753842, 0.9758871, 0.97507685, 0.96803576, 0.97756356, 0.97700477, 0.9735122, 0.9759989, 0.97029895, 0.9741269, 0.9739592, 0.9770886, 0.97261804, 0.97069013, 0.97493714, 0.97639006, 0.97354007, 0.9756636, 0.97569156, 0.97096956, 0.9783738, 0.9651858, 0.9780665, 0.96711373, 0.97611064, 0.96186084, 0.97549593, 0.9675887, 0.9727298, 0.9659402, 0.9719754, 0.97678125, 0.975943, 0.96283877, 0.9623358, 0.96328586, 0.9664431, 0.96873426, 0.9592624, 0.9616653, 0.9675049]
prec0:  [0.98410743, 0.98472846, 0.9833719, 0.98300314, 0.9829707, 0.9839303, 0.9833695, 0.9832096, 0.98291534, 0.9828029, 0.9831116, 0.9832617, 0.9831691, 0.98227, 0.9827766, 0.98266107, 0.98134655, 0.9840606, 0.98316354, 0.9830513, 0.9818274, 0.9834666, 0.98277575, 0.982577, 0.9823849, 0.98285276, 0.98237205, 0.98283553, 0.9830821, 0.98242015, 0.98356205, 0.9822247, 0.9831808, 0.98300636, 0.98383904, 0.982914, 0.9821997, 0.9816112, 0.98349184, 0.98397774, 0.9823919, 0.98248583, 0.98493993, 0.98348624, 0.9833362, 0.98358315, 0.98334616, 0.98477244, 0.98281, 0.98280233]
prec1:  [0.18317972, 0.21871203, 0.3396739, 0.43307087, 0.41444868, 0.32882884, 0.39616615, 0.3371105, 0.4099617, 0.2785146, 0.30446196, 0.2890995, 0.18195266, 0.32684824, 0.32698414, 0.23165138, 0.20647773, 0.23439878, 0.26444444, 0.25669643, 0.28099173, 0.24810606, 0.19489981, 0.25729442, 0.2880259, 0.24053451, 0.26567164, 0.28804347, 0.21033868, 0.3739496, 0.17160493, 0.34893617, 0.17440225, 0.3068493, 0.15844701, 0.28608924, 0.1414791, 0.16237113, 0.17503217, 0.25466892, 0.30169493, 0.2792793, 0.19076306, 0.15164836, 0.15277778, 0.18169935, 0.19457014, 0.16666667, 0.12866817, 0.16442451]
rec0:  [0.97978675, 0.98166835, 0.9930722, 0.9958946, 0.9956095, 0.99150413, 0.9946117, 0.99332875, 0.9956095, 0.99224544, 0.992445, 0.99144715, 0.9842342, 0.99506783, 0.99395597, 0.9904493, 0.9944121, 0.9856597, 0.99056333, 0.99050635, 0.99503934, 0.98868173, 0.9873988, 0.9920173, 0.9937279, 0.99027824, 0.9929867, 0.9925305, 0.98737025, 0.9957521, 0.9808701, 0.9956381, 0.9832649, 0.99278706, 0.97713536, 0.99224544, 0.9847759, 0.9907344, 0.9817254, 0.98748434, 0.99412704, 0.99315774, 0.97702134, 0.9779906, 0.97913104, 0.98215306, 0.9847759, 0.9734861, 0.9779906, 0.98406315]
rec1:  [0.22268908, 0.25210086, 0.17507003, 0.15406163, 0.15266107, 0.2044818, 0.17366947, 0.16666667, 0.14985995, 0.14705883, 0.16246499, 0.17086835, 0.17226891, 0.11764706, 0.14425771, 0.14145659, 0.071428575, 0.21568628, 0.16666667, 0.16106443, 0.0952381, 0.1834734, 0.14985995, 0.13585435, 0.12464986, 0.15126051, 0.12464986, 0.14845939, 0.16526611, 0.12464986, 0.19467787, 0.11484594, 0.17366947, 0.15686275, 0.2114846, 0.15266107, 0.1232493, 0.0882353, 0.1904762, 0.21008404, 0.12464986, 0.13025211, 0.26610646, 0.19327731, 0.18487395, 0.19467787, 0.18067227, 0.26050422, 0.15966387, 0.15406163]
minpse:  [0.19712793733681463, 0.226890756302521, 0.25630252100840334, 0.25770308123249297, 0.2549019607843137, 0.24687933425797504, 0.23809523809523808, 0.23407202216066483, 0.2619047619047619, 0.21708683473389356, 0.22128851540616246, 0.2462787550744249, 0.17969821673525377, 0.21481481481481482, 0.2298050139275766, 0.2222222222222222, 0.19164118246687054, 0.22969187675070027, 0.23259052924791088, 0.21328224776500637, 0.23841961852861035, 0.23249299719887956, 0.20110192837465565, 0.22643979057591623, 0.20028011204481794, 0.16899441340782123, 0.2602739726027397, 0.23249299719887956, 0.193006993006993, 0.19887955182072828, 0.17598908594815826, 0.18347338935574228, 0.17981438515081208, 0.2649930264993027, 0.17427385892116182, 0.21478382147838215, 0.1528117359413203, 0.16178343949044585, 0.18627450980392157, 0.24320652173913043, 0.17614424410540916, 0.201949860724234, 0.20728291316526612, 0.17086834733893558, 0.16803840877914952, 0.18802395209580838, 0.18950064020486557, 0.17458945548833188, 0.14969656102494944, 0.16114058355437666]
---
max_auprc:  0.1929347039201867
best_epoch:  5

###Results on test set:

Test loss = 10.0932

confusion matrix:
[[84732   512]
 [ 1122   329]]
accuracy = 0.9811522960662842
precision class 0 = 0.986931324005127
precision class 1 = 0.3912009596824646
recall class 0 = 0.9939936995506287
recall class 1 = 0.22674018144607544
AUC of ROC = 0.8416870252469572
AUC of PRC = 0.20553320265605343
min(+P, Se) = 0.28049620951068227



##############################################################################
#Experiment 1.5: Reproducing main StageNet model via Spyder
# need to copy down: args, all results, epoch training times, data size and stats etc.
##############################################################################

###best saved model: trained_model07052023_00_49
#epochs = 10
#Train Data = 5000
#Val data = 556
#test data = 1389

###Results from data stats:

Total number of ICU stays: 6945
Total number of patients: 5594
--- max number of stays per patient: 17
--- max length of stay: 2391.0
--- min number of stays per patient: 1
--- min length of stay: 5.0
--- average number of stays per patient: 1.2415087593850553
--- average length of stay: 70.56573074154068

Total number of visits (loaded subset): 462289, Positive samples: 9544
--Train--
Number of stays: 5000, Number of visits: 329751, Number positive visits: 7307
--Validation--
Number of stays: 556, Number of visits: 38663, Number positive visits: 714
--Test--
Number of stays: 1389, Number of visits: 93875, Number positive visits: 1523


###Stats from training

args:  Namespace(test_mode=0, data_path='./data/', file_name='trained_model', small_part=5000, batch_size=128, epochs=10, lr=0.001, input_dim=76, rnn_dim=384, output_dim=1, dropout_rate=0.5, dropconnect_rate=0.5, dropres_rate=0.3, K=10, chunk_level=3)
my_epoch_times:  [5194.958534199977, 1887.924203200033, 2027.681507500005, 1993.276278299978, 26374.854265499976, 2833.9260313000996, 2374.0421912000747, 1845.152436900069, 2416.3894245000556, 2297.8136307999957]
train_loss:  [23.632504, 13.064621, 12.150183, 11.703344, 11.311328, 11.058434, 10.900024, 10.726164, 10.499628, 10.38977]
val_loss:  [11.307138, 11.016273, 9.647919, 10.940557, 9.435366, 9.729634, 10.286312, 10.099855, 11.140583, 10.376353]
val_auroc:  [0.8153187692000052, 0.8354376075895064, 0.8583709227789647, 0.8612004728907185, 0.8691542302860247, 0.8748340138883699, 0.869202704459592, 0.8784110006187444, 0.8708994003577026, 0.8818097429415375]
val_auprc:  [0.12923703228601693, 0.15186209146321117, 0.1808379955380926, 0.1867828375223832, 0.18893610645147063, 0.19781792015449645, 0.1882080268707206, 0.1884387558712065, 0.1911127598051559, 0.18905891608980308]
acc:  [0.9656608, 0.966974, 0.9765856, 0.97921205, 0.97870916, 0.97555184, 0.9779268, 0.97694886, 0.978318, 0.97541213]
prec0:  [0.984068, 0.984754, 0.9836695, 0.98300505, 0.98302364, 0.9840358, 0.9833916, 0.9834025, 0.98304415, 0.9830216]
prec1:  [0.1893848, 0.2180723, 0.34343433, 0.44, 0.41111112, 0.3253796, 0.38343558, 0.34710744, 0.3916084, 0.28826532]
rec0:  [0.9808416, 0.98149735, 0.9925875, 0.9960087, 0.995467, 0.9911335, 0.9942696, 0.9932432, 0.99503934, 0.9920458]
rec1:  [0.21988796, 0.25350142, 0.1904762, 0.15406163, 0.15546219, 0.21008404, 0.17507003, 0.1764706, 0.15686275, 0.15826331]
minpse:  [0.1989795918367347, 0.23044692737430167, 0.25630252100840334, 0.26573426573426573, 0.2549019607843137, 0.25210084033613445, 0.24653739612188366, 0.24929971988795518, 0.25910364145658266, 0.226890756302521]
---
max_auprc:  0.19781792015449645
best_epoch:  5


###Results on test set:

Test loss = 10.0526


confusion matrix:
[[84719   525]
 [ 1117   334]]
accuracy = 0.9810600280761719
precision class 0 = 0.9869868159294128
precision class 1 = 0.3888242244720459
recall class 0 = 0.9938412308692932
recall class 1 = 0.2301860749721527
AUC of ROC = 0.8431852339322794
AUC of PRC = 0.21176690078946958
min(+P, Se) = 0.2866988283942109


##############################################################################
#Experiment 2: Reproducing StageNet_I via Google Colab
# need to copy down: args, all results, epoch training times, data size and stats etc.
##############################################################################

###best saved model: trained_model_ablation107052023_18_50

###Results from data stats:
    
Total number of ICU stays: 6945
Total number of patients: 5594
--- max number of stays per patient: 17
--- max length of stay: 2391.0
--- min number of stays per patient: 1
--- min length of stay: 5.0
--- average number of stays per patient: 1.2415087593850553
--- average length of stay: 70.56573074154068

Total number of visits (loaded subset): 462289, Positive samples: 9544
--Train--
Number of stays: 5000, Number of visits: 329751, Number positive visits: 7307
--Validation--
Number of stays: 556, Number of visits: 38663, Number positive visits: 714
--Test--
Number of stays: 1389, Number of visits: 93875, Number positive visits: 1523

    
###Stats from training

args:  Namespace(test_mode=0, data_path='./data/', file_name='trained_model_ablation1', small_part=5000, batch_size=128, epochs=50, lr=0.001, input_dim=76, rnn_dim=384, output_dim=1, dropout_rate=0.5, dropconnect_rate=0.5, dropres_rate=0.3, K=10, chunk_level=3, f='/root/.local/share/jupyter/runtime/kernel-7e82f817-968f-49e7-aa09-3e2c0aca9165.json')
my_epoch_times:  [53.52086713100016, 46.965236874999846, 46.74542804099974, 47.44141557700004, 47.00352606300021, 46.65724625200028, 47.31189780500017, 46.78698099600024, 47.027830318999804, 47.698594433999915, 47.561601247, 47.58009494299995, 47.102505786000165, 48.13305761999982, 46.67453225100007, 47.73171260399977, 46.70400473400014, 47.63864042400019, 48.47298742099974, 46.77603771200029, 47.07566262799992, 46.59625144499978, 47.57585595799992, 47.782205428, 47.01005685700011, 48.36920047600006, 48.338006080000014, 47.585185436999836, 47.54690949299993, 48.11408615400023, 47.046148511999945, 47.1333141959999, 47.848581878000005, 47.42280195400008, 48.035270299999866, 47.74122095099983, 47.300898844999665, 47.54149694700027, 45.96147602799965, 47.921739358000195, 47.234820554999715, 46.89082686599977, 48.31915083700005, 48.370691514999635, 47.48385805500038, 47.58558806699966, 47.78179422199992, 46.97921487599979, 47.15136527300001, 47.01837178200003]
train_loss:  [18.103998, 12.520728, 11.872603, 11.372606, 11.193216, 10.842011, 10.789874, 10.552843, 10.076742, 9.983843, 9.611994, 9.735464, 9.385374, 9.320999, 9.039427, 9.283429, 8.848845, 8.653712, 8.775248, 8.555113, 8.508679, 8.141492, 8.057556, 7.7581406, 7.6334395, 8.529776, 7.590416, 7.571869, 7.1320357, 7.1977882, 7.0783577, 6.913873, 6.7196374, 6.881213, 6.6898866, 6.7834616, 6.6272874, 6.4452424, 6.449501, 5.8963866, 6.075529, 5.5613723, 5.573402, 5.6530643, 5.730179, 5.1314874, 5.7100015, 5.456842, 5.4962196, 5.2468624]
val_loss:  [10.86265, 10.696661, 9.879415, 10.499185, 9.630815, 9.358164, 9.810308, 11.462938, 10.699252, 9.687746, 15.84346, 10.117145, 10.365843, 11.659726, 11.383371, 11.221838, 11.324425, 11.376756, 9.078074, 11.210058, 12.696442, 15.552694, 14.309027, 16.890003, 11.54922, 14.038564, 13.242022, 13.678772, 16.862833, 11.920906, 12.536216, 14.729688, 16.875483, 14.774942, 16.001293, 14.946646, 17.56869, 17.56862, 18.905071, 24.004028, 15.461569, 18.676853, 18.907322, 21.108862, 21.823349, 21.309467, 23.374004, 19.319641, 24.108398, 28.500565]
val_auroc:  [0.8454837802380617, 0.8525815332405057, 0.8719425334280135, 0.8670407123962597, 0.8791146946861765, 0.8862596041951961, 0.8855397587247922, 0.8658522366638525, 0.8669637885944662, 0.8806305707366765, 0.8621749275602588, 0.8692706242036101, 0.8748381665358582, 0.8581186095147376, 0.8817367322114157, 0.8726327513557597, 0.8866723134686649, 0.882865333954314, 0.8878742653407582, 0.8818259542384635, 0.8670046961651577, 0.8541502557232268, 0.831983523253069, 0.8379215296564514, 0.8575305507081383, 0.8201928992602857, 0.824434229730209, 0.8198025304317187, 0.826853126927587, 0.882740295342678, 0.8692871150056555, 0.8591670531823176, 0.8455930867044047, 0.8621708348067247, 0.8357155953954166, 0.8218390446610848, 0.8322317237991103, 0.8246779382296882, 0.8552459157913365, 0.8130418406386388, 0.8357556245214472, 0.8359920459231702, 0.8386508783009156, 0.8160961727603574, 0.8492257947768, 0.8344886477797869, 0.8450170066886373, 0.7991961752200025, 0.7855055752486878, 0.7881932166183842]
val_auprc:  [0.14940210783254318, 0.15947826372810864, 0.1879406711624385, 0.18469752981749374, 0.18477456764119154, 0.20045935214090807, 0.1652894724156301, 0.18312101435447828, 0.1857435915193222, 0.1758387344746849, 0.13468383581198856, 0.171269673797648, 0.1463018437132544, 0.16336527607301393, 0.15045061334286353, 0.15427371346874424, 0.16172245802846935, 0.16026393663736332, 0.20610178083351516, 0.16856756315948426, 0.17468459522542312, 0.21980513505773336, 0.13435833910256406, 0.18381663947457944, 0.1545747177665695, 0.17510947553172831, 0.18945067932308646, 0.14440240448211544, 0.16802316772023423, 0.1676076134595031, 0.2181321164184132, 0.16815552732553873, 0.17813761380749396, 0.16247157706131463, 0.10467494082473675, 0.15436978656122807, 0.12575325083884864, 0.16670523397388987, 0.14763732816113995, 0.15611831659612993, 0.15030849496457485, 0.19895580696225829, 0.21285595931066617, 0.14508127435081108, 0.15899502866888265, 0.1752339498897269, 0.11262886268599107, 0.15926769719474151, 0.12565972925092392, 0.13206691422246974]
acc:  [0.9777033, 0.97043866, 0.9786253, 0.97879297, 0.9760548, 0.9749092, 0.97625035, 0.97912824, 0.978318, 0.9767533, 0.9607991, 0.97510475, 0.9709975, 0.97639006, 0.9734283, 0.9730372, 0.97734004, 0.96711373, 0.9766974, 0.9748254, 0.9786253, 0.97829, 0.9727857, 0.9777033, 0.97261804, 0.9761386, 0.97694886, 0.97270185, 0.9783738, 0.96960044, 0.9790165, 0.9761386, 0.97460186, 0.9769209, 0.9639564, 0.9753004, 0.9747415, 0.97521657, 0.9688181, 0.9710813, 0.97639006, 0.97789884, 0.96968424, 0.9747136, 0.9737357, 0.9728136, 0.9694049, 0.9699916, 0.9761386, 0.97493714]
prec0:  [0.981947, 0.9841455, 0.9829678, 0.9831611, 0.9833602, 0.9842448, 0.9826818, 0.98191875, 0.98274505, 0.9825273, 0.98337686, 0.9835356, 0.98426473, 0.9822762, 0.98287815, 0.9840225, 0.9816694, 0.9845899, 0.9835894, 0.98301154, 0.9825603, 0.98369765, 0.9829217, 0.98308814, 0.98357636, 0.9836075, 0.9837575, 0.982975, 0.98282754, 0.9838012, 0.9839551, 0.9828433, 0.9834998, 0.98234004, 0.982602, 0.98353887, 0.9823011, 0.9829637, 0.9843126, 0.9844039, 0.9823034, 0.9832276, 0.9851837, 0.98292774, 0.984281, 0.984569, 0.9826984, 0.98416585, 0.98178285, 0.9825225]
prec1:  [0.31578946, 0.23860182, 0.40520447, 0.41877255, 0.31806615, 0.316, 0.29761904, 0.40462428, 0.38257575, 0.30718955, 0.14077164, 0.2993197, 0.249226, 0.28239202, 0.23956044, 0.27305606, 0.280543, 0.21525215, 0.34455958, 0.27360776, 0.39330545, 0.40597016, 0.23029046, 0.3653846, 0.25186568, 0.32843137, 0.35732648, 0.23108384, 0.3880597, 0.21837349, 0.4434251, 0.30113637, 0.2866521, 0.3041958, 0.13358779, 0.30414745, 0.23901099, 0.28101265, 0.22465754, 0.25421134, 0.28382838, 0.37777779, 0.2562418, 0.2676399, 0.2930403, 0.28452578, 0.17766498, 0.23372781, 0.24452555, 0.2546917]
rec0:  [0.99555254, 0.9857167, 0.99543846, 0.99540997, 0.99235946, 0.99024975, 0.99327177, 0.9970635, 0.9953529, 0.99395597, 0.97650814, 0.99119055, 0.98617285, 0.99384195, 0.9901357, 0.98853916, 0.995467, 0.9818109, 0.99278706, 0.99144715, 0.9958661, 0.9943266, 0.989423, 0.99435514, 0.9885677, 0.9921884, 0.9928726, 0.9892804, 0.99532443, 0.98520356, 0.9948113, 0.9929867, 0.9907059, 0.9943266, 0.98058504, 0.9913901, 0.99210286, 0.9919033, 0.9838636, 0.9861159, 0.99381346, 0.9944121, 0.9838636, 0.99141866, 0.9889953, 0.98774093, 0.98614436, 0.98523206, 0.99409854, 0.9920744]
rec1:  [0.10084034, 0.21988796, 0.15266107, 0.16246499, 0.17507003, 0.22128852, 0.14005603, 0.09803922, 0.14145659, 0.13165267, 0.18907563, 0.18487395, 0.2254902, 0.11904762, 0.15266107, 0.2114846, 0.086834736, 0.24509804, 0.18627451, 0.15826331, 0.13165267, 0.1904762, 0.15546219, 0.15966387, 0.18907563, 0.18767507, 0.19467787, 0.15826331, 0.14565827, 0.20308124, 0.20308124, 0.14845939, 0.1834734, 0.12184874, 0.14705883, 0.18487395, 0.12184874, 0.15546219, 0.22969188, 0.232493, 0.12044818, 0.16666667, 0.27310926, 0.15406163, 0.22408964, 0.2394958, 0.14705883, 0.22128852, 0.09383754, 0.13305323]
minpse:  [0.22911051212938005, 0.2381615598885794, 0.2647058823529412, 0.25555555555555554, 0.25761772853185594, 0.26666666666666666, 0.24649859943977592, 0.2541899441340782, 0.2550607287449393, 0.25874125874125875, 0.15987210231814547, 0.25630252100840334, 0.247682119205298, 0.24089635854341737, 0.25069252077562326, 0.2619047619047619, 0.2493150684931507, 0.2338935574229692, 0.2857142857142857, 0.2590529247910863, 0.26750700280112044, 0.29346314325452016, 0.20949720670391062, 0.26573426573426573, 0.23160762942779292, 0.23949579831932774, 0.24732620320855614, 0.22549019607843138, 0.2594142259414226, 0.2157622739018088, 0.2773109243697479, 0.2493112947658402, 0.26433566433566436, 0.2078853046594982, 0.1421161825726141, 0.2292817679558011, 0.20388349514563106, 0.26648721399730824, 0.22937062937062938, 0.26170798898071623, 0.2303448275862069, 0.25069637883008355, 0.2661064425770308, 0.21675531914893617, 0.2686762778505898, 0.266016713091922, 0.16022889842632332, 0.22758620689655173, 0.21988795518207283, 0.25139664804469275]
---
max_auprc:  0.21980513505773336
best_epoch:  21


###Results on test set:

Test loss = 12.9920


confusion matrix:
[[84691   553]
 [ 1155   296]]
accuracy = 0.9802987575531006
precision class 0 = 0.9865456819534302
precision class 1 = 0.34864547848701477
recall class 0 = 0.993512749671936
recall class 1 = 0.20399723947048187
AUC of ROC = 0.8375129247502309
AUC of PRC = 0.20880075444440566
min(+P, Se) = 0.27911784975878706

##############################################################################
#Experiment 3: Reproducing StageNet_II via Google Colab
# need to copy down: args, all results, epoch training times, data size and stats etc.
##############################################################################

###best saved model: trained_model_ablation207052023_21_11


###Results from data stats:

Total number of ICU stays: 6945
Total number of patients: 5594
--- max number of stays per patient: 17
--- max length of stay: 2391.0
--- min number of stays per patient: 1
--- min length of stay: 5.0
--- average number of stays per patient: 1.2415087593850553
--- average length of stay: 70.56573074154068

Total number of visits (loaded subset): 462289, Positive samples: 9544
--Train--
Number of stays: 5000, Number of visits: 329751, Number positive visits: 7307
--Validation--
Number of stays: 556, Number of visits: 38663, Number positive visits: 714
--Test--
Number of stays: 1389, Number of visits: 93875, Number positive visits: 1523


###Stats from training

args:  Namespace(test_mode=0, data_path='./data/', file_name='trained_model_ablation2', small_part=5000, batch_size=128, epochs=50, lr=0.001, input_dim=76, rnn_dim=384, output_dim=1, dropout_rate=0.5, dropconnect_rate=0.5, dropres_rate=0.3, K=10, chunk_level=3, f='/root/.local/share/jupyter/runtime/kernel-f5a5045e-e313-47bd-9e63-93722646dbfc.json')
my_epoch_times:  [43.79840974699982, 39.197561482999845, 39.05506006399992, 39.58472141999937, 38.52912911000021, 38.419916741000634, 39.370676523999464, 38.93925171400042, 39.26577687400004, 39.714313759000106, 40.51470758200048, 39.63377812999988, 39.56421609500012, 40.026871377000134, 40.01277177300017, 40.568097940999905, 39.15516745599962, 39.93256628099971, 40.81712327000059, 39.19181686800039, 39.70535662199927, 39.446075768999435, 40.271870542000215, 40.02906615699976, 39.70771315999991, 40.35813434300053, 40.62615116500001, 39.81542464799986, 39.70161929099959, 39.84622990200023, 38.85346423899955, 38.74908194300042, 39.38271613300003, 38.82662173800054, 39.38315835599951, 39.058757056000104, 39.53772448000018, 39.43012197600001, 38.31362704500043, 40.04544130500017, 39.30896649999977, 39.71275263200005, 40.123120401999586, 40.60093709800003, 40.345266017999165, 39.75426425200021, 39.87078901200039, 39.431430552000165, 39.97657315100059, 39.27112517600017]
train_loss:  [26.58382, 15.545555, 13.062505, 12.736682, 12.204922, 11.996259, 11.577826, 11.2663965, 11.205088, 11.002157, 10.965435, 10.645295, 10.509924, 10.677893, 10.422864, 10.71438, 10.147049, 10.02077, 9.936773, 9.915042, 9.802327, 9.530001, 9.818945, 9.398882, 9.234221, 9.028865, 8.818713, 8.850766, 8.530825, 8.405995, 8.292017, 8.236393, 8.080931, 7.832621, 8.176172, 8.008206, 7.7072344, 7.5678573, 7.9134417, 8.02585, 7.3978715, 7.364849, 7.0976777, 7.0967093, 6.799362, 7.1081038, 6.9030137, 6.9163923, 7.2493424, 6.9753847]
val_loss:  [18.878632, 17.079548, 10.506252, 9.926561, 9.765723, 11.272933, 9.999063, 10.186204, 13.511301, 9.595396, 9.577985, 9.883543, 11.348818, 11.50854, 14.928091, 10.945311, 12.399936, 12.339876, 10.171778, 11.055239, 13.90082, 16.69173, 12.352917, 11.852285, 11.8996315, 12.070448, 12.618045, 12.133313, 11.893258, 15.025993, 11.473372, 14.173106, 13.074384, 14.69355, 12.1242695, 15.631253, 18.05679, 23.011404, 18.418081, 15.475934, 23.379017, 18.570139, 22.865234, 22.533949, 21.207233, 21.60358, 22.335775, 15.198235, 14.278589, 25.139715]
val_auroc:  [0.7750624454366078, 0.7976602546595101, 0.836830800857234, 0.8504756618122217, 0.8561670249123712, 0.8530788527065519, 0.8729398476233918, 0.8711616360536687, 0.8605580303737415, 0.8822767361021271, 0.872911098525395, 0.8786413527664458, 0.8413628965099553, 0.8711071924493369, 0.8512859072241052, 0.8747159629047193, 0.8633778177709674, 0.8787452488122629, 0.8801675904710156, 0.8725163175088715, 0.8479511516090072, 0.8500649889331945, 0.8488442702887975, 0.8781455106846022, 0.8865073854835581, 0.8727024679184023, 0.854077863897298, 0.8669690792270838, 0.8707936076700038, 0.8623501573054813, 0.8894839552881251, 0.8609109455163066, 0.8458508702831116, 0.8669000015332852, 0.8561639303914063, 0.8326688698058764, 0.8271724215972167, 0.7765995838408348, 0.7769187587225561, 0.8210698266078013, 0.7627136297556997, 0.8302146950694977, 0.8121047597964947, 0.7886886993365028, 0.7865777169574638, 0.8169233881259197, 0.7883331688246059, 0.8241777638184935, 0.8505526654726208, 0.7519189823266518]
val_auprc:  [0.06189122145153009, 0.10342087841724547, 0.1575624726855076, 0.16703836268496658, 0.17584336531525824, 0.183881980566567, 0.18667668685712036, 0.18269978553269345, 0.18345691274254888, 0.17512033890111472, 0.15847522453711724, 0.16794933817284158, 0.13596789143050286, 0.15322222856927203, 0.15608567302587312, 0.1421712249996475, 0.14853444974942998, 0.15043305905117418, 0.15251681220829316, 0.15733239063682367, 0.17504848590097202, 0.15303854572307063, 0.14944864987084203, 0.16603572296311692, 0.1700481754179469, 0.15960045996305988, 0.1350289835023977, 0.1573616118120314, 0.16588937605739149, 0.1526799374116765, 0.16823130961574256, 0.1558822603682777, 0.17068515106548143, 0.13649256673390464, 0.14872573102114883, 0.16646907465313848, 0.13426444964951761, 0.12740005993087586, 0.12222815723622145, 0.16594916255960684, 0.12410581479007676, 0.18033169780691283, 0.11229343256644186, 0.09771148514123129, 0.15047496561868873, 0.119684221956395, 0.12562604119448045, 0.10791549329646662, 0.15151985310420787, 0.14562353571045805]
acc:  [0.98005027, 0.97633415, 0.97611064, 0.97432244, 0.977843, 0.97856945, 0.97761947, 0.9734283, 0.97904444, 0.9767533, 0.97440624, 0.9765856, 0.97694886, 0.977843, 0.9790165, 0.977843, 0.9764459, 0.9734004, 0.97639006, 0.9769768, 0.97904444, 0.9729813, 0.9740989, 0.978765, 0.97789884, 0.97563565, 0.9742665, 0.97678125, 0.97275776, 0.97809446, 0.9749092, 0.9747136, 0.9745739, 0.97541213, 0.9759989, 0.97725624, 0.9682872, 0.97834593, 0.9723666, 0.9773959, 0.97462976, 0.9736519, 0.97166806, 0.9729533, 0.9705504, 0.973568, 0.97460186, 0.96515787, 0.9721151, 0.97337246]
prec0:  [0.98005027, 0.9816509, 0.98322463, 0.98393357, 0.9833357, 0.98296684, 0.9836047, 0.98416615, 0.98237777, 0.982745, 0.9830317, 0.98296, 0.98231333, 0.9822478, 0.9816192, 0.98146224, 0.9819242, 0.9826318, 0.98214036, 0.9820695, 0.98194426, 0.98341787, 0.98223513, 0.9818039, 0.9821946, 0.9825348, 0.98237437, 0.9818489, 0.98349637, 0.9817375, 0.9829857, 0.98281854, 0.98371845, 0.9816338, 0.9822148, 0.9823732, 0.983256, 0.9816068, 0.98231316, 0.9824571, 0.98219025, 0.98397756, 0.98150975, 0.98175144, 0.98392695, 0.9825256, 0.9820809, 0.9820731, 0.98274595, 0.9839181]
prec1:  [nan, 0.24124514, 0.31496063, 0.2945892, 0.37846154, 0.40221402, 0.37677053, 0.284153, 0.41428572, 0.3167702, 0.26511627, 0.31976745, 0.30388692, 0.33877552, 0.38064516, 0.28877005, 0.26373628, 0.2283105, 0.2749141, 0.29166666, 0.3988764, 0.25244617, 0.22193211, 0.37078652, 0.33891213, 0.2729885, 0.23255815, 0.27058825, 0.25142857, 0.32323232, 0.27450982, 0.26302728, 0.29324895, 0.2137931, 0.26688102, 0.3188406, 0.1872214, 0.32960895, 0.196468, 0.32851985, 0.23055555, 0.28273246, 0.14285715, 0.17435898, 0.23354232, 0.2264151, 0.22379604, 0.120910384, 0.21255061, 0.27579737]
rec0:  [1.0, 0.9944406, 0.992559, 0.98996466, 0.99424106, 0.9953815, 0.9937279, 0.98879576, 0.99649334, 0.9937279, 0.990991, 0.99332875, 0.99438363, 0.9953815, 0.9972631, 0.99620825, 0.9942696, 0.9903638, 0.9939845, 0.9946687, 0.9969495, 0.98910934, 0.99150413, 0.9968069, 0.9954955, 0.99278706, 0.9915327, 0.9946972, 0.98879576, 0.9961797, 0.9915612, 0.9915327, 0.9904493, 0.9934998, 0.9934998, 0.99464023, 0.9844053, 0.9965789, 0.98962253, 0.9946972, 0.99210286, 0.9892234, 0.98973656, 0.99081993, 0.98605883, 0.99064887, 0.9921884, 0.9823811, 0.9889098, 0.9889953]
rec1:  [0.0, 0.086834736, 0.16806723, 0.20588236, 0.17226891, 0.15266107, 0.18627451, 0.2184874, 0.12184874, 0.14285715, 0.15966387, 0.15406163, 0.12044818, 0.1162465, 0.082633056, 0.075630255, 0.10084034, 0.14005603, 0.11204482, 0.10784314, 0.09943978, 0.18067227, 0.11904762, 0.09243698, 0.11344538, 0.13305323, 0.12605043, 0.09663866, 0.18487395, 0.08963586, 0.15686275, 0.14845939, 0.19467787, 0.086834736, 0.1162465, 0.1232493, 0.1764706, 0.082633056, 0.12464986, 0.12745099, 0.1162465, 0.20868348, 0.084033616, 0.0952381, 0.20868348, 0.13445379, 0.11064426, 0.11904762, 0.14705883, 0.20588236]
minpse:  [0.0836104513064133, 0.20868347338935575, 0.242296918767507, 0.24930747922437674, 0.2619047619047619, 0.2625, 0.26330532212885155, 0.2507002801120448, 0.25034965034965034, 0.21896792189679218, 0.21568627450980393, 0.22549019607843138, 0.19887955182072828, 0.21223958333333334, 0.22451790633608815, 0.20053475935828877, 0.20505992010652463, 0.2015855039637599, 0.21212121212121213, 0.20168067226890757, 0.23291492329149233, 0.24910607866507747, 0.2180365296803653, 0.24778761061946902, 0.2430167597765363, 0.21731123388581952, 0.21109770808202655, 0.23482849604221637, 0.2462787550744249, 0.21972132904608788, 0.25555555555555554, 0.22845953002610966, 0.2629370629370629, 0.19062259800153727, 0.2055984555984556, 0.26286509040333794, 0.18860244233378562, 0.18479685452162517, 0.17687074829931973, 0.26425591098748263, 0.206993006993007, 0.2748299319727891, 0.18133535660091046, 0.17510259917920656, 0.23342541436464087, 0.19747899159663865, 0.24089635854341737, 0.1533847472150814, 0.2383093525179856, 0.24200278164116829]
---
max_auprc:  0.18667668685712036
best_epoch:  6


###Results on test set:

Test loss = 10.1016


confusion matrix:
[[84884   360]
 [ 1181   270]]
accuracy = 0.9822250604629517
precision class 0 = 0.9862778186798096
precision class 1 = 0.4285714328289032
recall class 0 = 0.9957768321037292
recall class 1 = 0.18607856333255768
AUC of ROC = 0.8444596596607215
AUC of PRC = 0.21078365808954203
min(+P, Se) = 0.27980702963473464


##############################################################################
#Experiment 4: Using pre-trained models
##############################################################################

###Original StageNet:
######Full Sample:

Test loss = 9.8845


confusion matrix:
[[698273  10434]
 [  8948   6147]]
accuracy = 0.9732219576835632
precision class 0 = 0.9873476624488831
precision class 1 = 0.3707255423069
recall class 0 = 0.9852774143218994
recall class 1 = 0.4072209298610687
AUC of ROC = 0.9086531322195948
AUC of PRC = 0.34059397393949986
min(+P, Se) = 0.3899046104928458

######Subsample:
    
Test loss = 9.1922


confusion matrix:
[[84424   820]
 [  975   476]]
accuracy = 0.9792952537536621
precision class 0 = 0.9885830283164978
precision class 1 = 0.3672839403152466
recall class 0 = 0.9903805255889893
recall class 1 = 0.3280496299266815
AUC of ROC = 0.8896462931672431
AUC of PRC = 0.28926306571673305
min(+P, Se) = 0.34734665747760163
        
###Reproduced StageNet

######Full Sample:
    
 Test loss = 11.3214


 confusion matrix:
 [[701738   6969]
  [ 11410   3685]]
 accuracy = 0.9746077060699463
 precision class 0 = 0.9840005040168762
 precision class 1 = 0.3458794951438904
 recall class 0 = 0.9901666045188904
 recall class 1 = 0.24412056803703308
 AUC of ROC = 0.8737938079830776
 AUC of PRC = 0.22807719690483588
 min(+P, Se) = 0.2915811088295688
   
######Subsample:
    
Test loss = 10.0933


confusion matrix:
[[84732   512]
 [ 1122   329]]
accuracy = 0.9811522960662842
precision class 0 = 0.986931324005127
precision class 1 = 0.3912009596824646
recall class 0 = 0.9939936995506287
recall class 1 = 0.22674018144607544
AUC of ROC = 0.8416872839602512
AUC of PRC = 0.20553427954870135
min(+P, Se) = 0.28049620951068227
    
###StageNet_I
######Full Sample:
    
Test loss = 16.5680


confusion matrix:
[[702989   5718]
 [ 11568   3527]]
accuracy = 0.9761177897453308
precision class 0 = 0.9838109612464905
precision class 1 = 0.38150352239608765
recall class 0 = 0.9919317960739136
recall class 1 = 0.23365353047847748
AUC of ROC = 0.8500106953613311
AUC of PRC = 0.2258824534301621
min(+P, Se) = 0.3145881574594773

######Subsample:    
    
 Test loss = 12.9921


 confusion matrix:
 [[84691   553]
  [ 1155   296]]
 accuracy = 0.9802987575531006
 precision class 0 = 0.9865456819534302
 precision class 1 = 0.34864547848701477
 recall class 0 = 0.993512749671936
 recall class 1 = 0.20399723947048187
 AUC of ROC = 0.8375128398599314
 AUC of PRC = 0.20879984588125294
 min(+P, Se) = 0.27911784975878706   
    

###StageNet_II
######Full Sample:
    
Test loss = 11.3272


confusion matrix:
[[703099   5608]
 [ 11900   3195]]
accuracy = 0.9758110642433167
precision class 0 = 0.9833565950393677
precision class 1 = 0.36294445395469666
recall class 0 = 0.9920870065689087
recall class 1 = 0.2116594910621643
AUC of ROC = 0.8723744397569753
AUC of PRC = 0.21982961079830515
min(+P, Se) = 0.2870486916197416

######Subsample:

Test loss = 10.1016


confusion matrix:
[[84884   360]
 [ 1181   270]]
accuracy = 0.9822250604629517
precision class 0 = 0.9862778186798096
precision class 1 = 0.4285714328289032
recall class 0 = 0.9957768321037292
recall class 1 = 0.18607856333255768
AUC of ROC = 0.8444596637031166
AUC of PRC = 0.21078365722703682
min(+P, Se) = 0.27980702963473464

"""