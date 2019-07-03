from pathlib import Path
import os
import json
import shutil
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from arff2pandas import a2p


class NumpyToJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyToJsonEncoder, self).default(obj)


def csv_to_benchmark_data(input_file: str, data_set_name: str, target_name: str):
    path_input_file = Path(os.path.dirname(__file__)).parent.joinpath(input_file)
    with path_input_file.open(mode='r') as f:
        df = a2p.load(f)
    df.columns = [colname.split('@')[0] for colname in df.columns]

    path_output = Path(os.path.dirname(__file__)).parent.joinpath('output').joinpath(data_set_name)
    if not path_output.is_dir():
        path_output.mkdir()

    path_dataset = Path(os.path.dirname(__file__)).parent.joinpath('cdataset').joinpath('data').joinpath(data_set_name)
    if not path_dataset.is_dir():
        path_dataset.mkdir()

    type_df = pd.DataFrame([dtype.name for dtype in df.dtypes.values]).T
    type_df.to_csv(path_output.joinpath('dtype.csv'), index=False, header=False)
    df.to_csv(path_output.joinpath('all.csv'), index=False)

    with path_output.joinpath('target_name.txt').open(mode='w') as f:
        f.write(target_name)

    with path_dataset.joinpath('target_name.txt').open(mode='w') as f:
        f.write(target_name)

    shutil.copy(str(path_input_file), str(path_dataset.joinpath('df.arff')))

    kf = KFold(n_splits=4, random_state=71, shuffle=True)
    kf_index_list = [{"train": train_index, "test": test_index} for train_index, test_index in kf.split(df)]
    with path_output.joinpath('cv4_index.json').open(mode='w') as f:
        json.dump({"cv_list": kf_index_list}, f, cls=NumpyToJsonEncoder)

    kf = KFold(n_splits=8, random_state=71, shuffle=True)
    kf_index_list = [{"train": train_index, "test": test_index} for train_index, test_index in kf.split(df)]
    with path_output.joinpath('cv8_index.json').open(mode='w') as f:
        json.dump({"cv_list": kf_index_list}, f, cls=NumpyToJsonEncoder)


csv_to_benchmark_data('input/jungle_chess/jungle_chess_2pcs_raw_endgame_complete.arff',
                      'jungle_chess', 'class')

csv_to_benchmark_data('input/speed_dating/speeddating.arff',
                      'speed_dating', 'match')

csv_to_benchmark_data('input/mice_protein/phpO6CpB2.arff',
                      'mice_protein', 'class')

csv_to_benchmark_data('input/churn/churn.arff', 'churn',
                      'class')

csv_to_benchmark_data('input/climate_model_simulation_crashes/phpJem85A.arff',
                      'climate_model_simulation_crashes', 'outcome')

csv_to_benchmark_data('input/segment/phpyM5ND4.arff',
                      'segment', 'class')

csv_to_benchmark_data('input/wilt/phpcSeK3V.arff',
                      'wilt', 'class')

csv_to_benchmark_data('input/steel_plates_fault/php5s7Ep8.arff',
                      'steel_plates_fault', 'target')

csv_to_benchmark_data('input/australian/phpelnJ6y.arff',
                      'australian', 'A15')

csv_to_benchmark_data('input/internet_advertisements/phpPIHVvG.arff',
                      'internet_advertisements', 'class')

csv_to_benchmark_data('input/car/php2jDIhh.arff',
                      'car', 'class')

csv_to_benchmark_data('input/collins/php5OMDBD.arff',
                      'collins', 'Corp.Genre')

csv_to_benchmark_data('input/dna/dna.arff',
                      'dna', 'class')

csv_to_benchmark_data('input/connect_4/connect-4.arff',
                      'connect_4', 'class')

csv_to_benchmark_data('input/texture/phpBDgUyY.arff',
                      'texture', 'Class')

csv_to_benchmark_data('input/LED_display_domain_7digit/phpSj3fWL.arff',
                      'LED_display_domain_7digit', 'Class')

csv_to_benchmark_data('input/numerai28_6/phpg2t68G.arff',
                      'numerai28_6', 'attribute_21')

csv_to_benchmark_data('input/dresses_sales/phpcFPMhq.arff',
                      'dresses_sales', 'Class')

csv_to_benchmark_data('input/cjs/phpDAC5gS.arff',
                      'cjs', 'TR')

csv_to_benchmark_data('input/cylinder_bands/phpAz9Len.arff',
                      'cylinder_bands', 'band_type')

csv_to_benchmark_data('input/adult/phpMawTba.arff',
                      'adult', 'class')

csv_to_benchmark_data('input/micro_mass/phpHyLSNF.arff',
                      'micro_mass', 'Class')

csv_to_benchmark_data('input/wdbc/phpAmSP4g.arff',
                      'wdbc', 'Class')

csv_to_benchmark_data('input/wdbc/phpAmSP4g.arff',
                      'wdbc', 'Class')

csv_to_benchmark_data('input/nomao/phpDYCOet.arff',
                      'nomao', 'Class')

csv_to_benchmark_data('input/madelon/phpfLuQE4.arff',
                      'madelon', 'Class')

csv_to_benchmark_data('input/ilpd/phpOJxGL9.arff',
                      'ilpd', 'Class')

csv_to_benchmark_data('input/hill_valley/php3isjYz.arff',
                      'hill_valley', 'Class')

csv_to_benchmark_data('input/gas_drift/phpbL6t4U.arff',
                      'gas_drift', 'Class')

csv_to_benchmark_data('input/cnae_9/phpmcGu2X.arff',
                      'cnae_9', 'Class')

csv_to_benchmark_data('input/blood_transfusion_service_center/php0iVrYT.arff',
                      'blood_transfusion_service_center', 'Class')

csv_to_benchmark_data('input/banknote_authentication/php50jXam.arff',
                      'banknote_authentication', 'Class')

csv_to_benchmark_data('input/bank_marketing/phpkIxskf.arff',
                      'bank_marketing', 'Class')

csv_to_benchmark_data('input/artificial_characters/phpPQrHPH.arff',
                      'artificial_characters', 'Class')

csv_to_benchmark_data('input/kc2/kc2.arff',
                      'kc2', 'problems')

csv_to_benchmark_data('input/mozilla4/mozilla4.arff',
                      'mozilla4', 'state')

csv_to_benchmark_data('input/gina_agnostic/gina_agnostic.arff',
                      'gina_agnostic', 'label')

csv_to_benchmark_data('input/profb/profb.arff',
                      'profb', 'Home/Away')

csv_to_benchmark_data('input/analcatdata_dmft/analcatdata_dmft.arff',
                      'analcatdata_dmft', 'Prevention')

csv_to_benchmark_data('input/analcatdata_authorship/analcatdata_authorship.arff',
                      'analcatdata_authorship', 'Author')

csv_to_benchmark_data('input/irish/irish.arff',
                      'irish', 'Leaving_Certificate')

csv_to_benchmark_data('input/synthetic_control/synthetic_control.arff',
                      'synthetic_control', 'class')

csv_to_benchmark_data('input/japanese_vowels/JapaneseVowels.arff',
                      'japanese_vowels', 'speaker')

csv_to_benchmark_data('input/monks_problems_3/phphZierv.arff',
                      'monks_problems_3', 'class')

csv_to_benchmark_data('input/monks_problems_2/php4fATLZ.arff',
                      'monks_problems_2', 'class')

csv_to_benchmark_data('input/monks_problems_1/phpAyyBys.arff',
                      'monks_problems_1', 'class')

csv_to_benchmark_data('input/vowel/phpd8EoD9.arff',
                      'vowel', 'Class')

csv_to_benchmark_data('input/eucalyptus/dataset_194_eucalyptus.arff',
                      'eucalyptus', 'Utility')

csv_to_benchmark_data('input/electricity/electricity-normalized.arff',
                      'electricity', 'class')

csv_to_benchmark_data('input/vehicle/dataset_54_vehicle.arff',
                      'vehicle', 'Class')

csv_to_benchmark_data('input/tic_tac_toe/dataset_50_tic-tac-toe.arff',
                      'tic_tac_toe', 'Class')

csv_to_benchmark_data('input/soybean/dataset_42_soybean.arff',
                      'soybean', 'class')

csv_to_benchmark_data('input/diabetes/dataset_37_diabetes.arff',
                      'diabetes', 'class')

csv_to_benchmark_data('input/credit_approval/dataset_29_credit-a.arff',
                      'credit_approval', 'class')

csv_to_benchmark_data('input/breast_w/openml_phpJNxH0q.arff',
                      'breast_w', 'Class')

csv_to_benchmark_data('input/balance_scale/dataset_11_balance-scale.arff',
                      'balance_scale', 'class')
