from pathlib import Path
import os
from arff2pandas import a2p
import pandas as pd


class ClassificationDataSetName:
    ADULT = 'adult'
    ANALCATDATA_AUTHORSHIP = 'analcatdata_authorship'
    ANALCATDATA_DMFT = 'analcatdata_dmft'
    ARTIFICIAL_CHARACTERS = 'artificial_characters'
    AUSTRALIAN = 'australian'
    BALANCE_SCALE = 'balance_scale'
    BANK_MARKETING = 'bank_marketing'
    BANKNOTE_AUTHENTICATION = 'banknote_authentication'
    BLOOD_TRANSFUSION_SERVICE_CENTER = 'blood_transfusion_service_center'
    BREAST_W = 'breast_w'
    CAR = 'car'
    CHURN = 'churn'
    CJS = 'cjs'
    CLIMATE_MODEL_SIMULATION_CRASHES = 'climate_model_simulation_crashes'
    CNAE_9 = 'cnae_9'
    COLLINS = 'collins'
    CONNECT_4 = 'connect_4'
    CREDIT_APPROVAL = 'credit_approval'
    CYLINDER_BANDS = 'cylinder_bands'
    DIABETES = 'diabetes'
    DNA = 'dna'
    DRESSES_SALES = 'dresses_sales'
    ELECTRICITY = 'electricity'
    EUCALYPTUS = 'eucalyptus'
    GAS_DRIFT = 'gas_drift'
    GINA_AGNOSTIC = 'gina_agnostic'
    HILL_VALLEY = 'hill_valley'
    ILPD = 'ilpd'
    INTERNET_ADVERTISEMENTS = 'internet_advertisements'
    IRISH = 'irish'
    JAPANESE_VOWELS = 'japanese_vowels'
    JUNGLE_CHESS = 'jungle_chess'
    KC2 = 'kc2'
    LED_DISPLAY_DOMAIN_7DIGIT = 'LED_display_domain_7digit'
    MADELON = 'madelon'
    MICE_PROTEIN = 'mice_protein'
    MICRO_MASS = 'micro_mass'
    MONKS_PROBLEM_1 = 'monks_problems_1'
    MONKS_PROBLEM_2 = 'monks_problems_2'
    MONKS_PROBLEM_3 = 'monks_problems_3'
    MOZILLA4 = 'mozilla4'
    NOMAO = 'nomao'
    NUMERAI28_6 = 'numerai28_6'
    PROFB = 'profb'
    SEGMENT = 'segment'
    SOYBEAN = 'soybean'
    SPEED_DATING = 'speed_dating'
    STEEL_PLATES_FAULT = 'steel_plates_fault'
    SYNTHETIC_CONTROL = 'synthetic_control'
    TEXTURE = 'texture'
    TIC_TAC_TOE = 'tic_tac_toe'
    VEHICLE = 'vehicle'
    VOWEL = 'vowel'
    WDBC = 'wdbc'
    WILT = 'wilt'

    @staticmethod
    def get_all_dataset_names():
        return [
            ClassificationDataSetName.ADULT,
            ClassificationDataSetName.ANALCATDATA_AUTHORSHIP,
            ClassificationDataSetName.ANALCATDATA_DMFT,
            ClassificationDataSetName.ARTIFICIAL_CHARACTERS,
            ClassificationDataSetName.AUSTRALIAN,
            ClassificationDataSetName.BALANCE_SCALE,
            ClassificationDataSetName.BANK_MARKETING,
            ClassificationDataSetName.BANKNOTE_AUTHENTICATION,
            ClassificationDataSetName.BLOOD_TRANSFUSION_SERVICE_CENTER,
            ClassificationDataSetName.BREAST_W,
            ClassificationDataSetName.CAR,
            ClassificationDataSetName.CHURN,
            ClassificationDataSetName.CJS,
            ClassificationDataSetName.CLIMATE_MODEL_SIMULATION_CRASHES,
            ClassificationDataSetName.CNAE_9,
            ClassificationDataSetName.COLLINS,
            ClassificationDataSetName.CONNECT_4,
            ClassificationDataSetName.CREDIT_APPROVAL,
            ClassificationDataSetName.CYLINDER_BANDS,
            ClassificationDataSetName.DIABETES,
            ClassificationDataSetName.DNA,
            ClassificationDataSetName.DRESSES_SALES,
            ClassificationDataSetName.ELECTRICITY,
            ClassificationDataSetName.EUCALYPTUS,
            ClassificationDataSetName.GAS_DRIFT,
            ClassificationDataSetName.GINA_AGNOSTIC,
            ClassificationDataSetName.HILL_VALLEY,
            ClassificationDataSetName.ILPD,
            ClassificationDataSetName.INTERNET_ADVERTISEMENTS,
            ClassificationDataSetName.IRISH,
            ClassificationDataSetName.JAPANESE_VOWELS,
            ClassificationDataSetName.JUNGLE_CHESS,
            ClassificationDataSetName.KC2,
            ClassificationDataSetName.LED_DISPLAY_DOMAIN_7DIGIT,
            ClassificationDataSetName.MADELON,
            ClassificationDataSetName.MICE_PROTEIN,
            ClassificationDataSetName.MICRO_MASS,
            ClassificationDataSetName.MONKS_PROBLEM_1,
            ClassificationDataSetName.MONKS_PROBLEM_2,
            ClassificationDataSetName.MONKS_PROBLEM_3,
            ClassificationDataSetName.MOZILLA4,
            ClassificationDataSetName.NOMAO,
            ClassificationDataSetName.NUMERAI28_6,
            ClassificationDataSetName.PROFB,
            ClassificationDataSetName.SEGMENT,
            ClassificationDataSetName.SOYBEAN,
            ClassificationDataSetName.SPEED_DATING,
            ClassificationDataSetName.STEEL_PLATES_FAULT,
            ClassificationDataSetName.SYNTHETIC_CONTROL,
            ClassificationDataSetName.TEXTURE,
            ClassificationDataSetName.TIC_TAC_TOE,
            ClassificationDataSetName.VEHICLE,
            ClassificationDataSetName.VOWEL,
            ClassificationDataSetName.WDBC,
            ClassificationDataSetName.WILT
        ]


class ClassificationDataSet:

    @staticmethod
    def load_df(dataset_name: str) -> pd.DataFrame:
        with Path(os.path.dirname(__file__)).joinpath('data').joinpath(dataset_name).joinpath('df.arff').open(mode='r') as f:
            df = a2p.load(f)
        df.columns = [colname.split('@')[0] for colname in df.columns]
        return df

    @staticmethod
    def load_target_name(dataset_name: str) -> str:
        with Path(os.path.dirname(__file__)).joinpath('data').joinpath(dataset_name).joinpath('target_name.txt').open(mode='r') as f:
            target_name = f.read()

        return target_name

    @staticmethod
    def load_df_and_target_name(dataset_name: str) -> (pd.DataFrame, str):
        return ClassificationDataSet.load_df(dataset_name), ClassificationDataSet.load_target_name(dataset_name)
