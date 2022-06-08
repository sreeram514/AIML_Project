import datetime

import numpy as np
import pandas as pd
from math import sqrt
from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

import log_init as log
from log_init import initiate_logger

"""
Expects data has empty values as either blank or na , clean the data before giving it to the model.
"""


def get_prime_num_list():
    p_num = []

    def is_prime(n):
        prime_flag = 0
        if n > 1:
            for num in range(2, int(sqrt(n)) + 1):
                if n % num == 0:
                    prime_flag = 1
                if prime_flag == 0:
                    return True
                else:
                    return False
        else:
            return False

    for n in range(1, 500):
        if is_prime(n):
            p_num.append(n)
    return p_num


def calculate_outliers(component):
    quartile_1, quartile_3 = np.percentile(component, [25, 75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    outliers_tuple = np.where((component > upper_bound) | (component < lower_bound))
    return set(outliers_tuple[0])


class RegressionLearning:
    def __init__(self, input_data_path, output_component, discrete_comp):
        """
        Regression Learning Inputs
        :param input_data_path: Path of data tobe read
        :param output_component: Output/target Column
        :param discrete_comp: List of Discrete Columns of given data
        """
        self.input_data_path = input_data_path
        self.output_component = output_component
        self.discrete_columns = discrete_comp
        self.continuous_columns = None
        self.data = None
        self.cleaned_data = None
        self.trained_model = None
        self.in_test = None
        self.out_test = None
        self.r2_results = None
        self.pca_data = None
        self.prime_num_list = get_prime_num_list()

    def set_continuous_columns(self):
        """
        Sets continous columns of given self.data
        :return:
        """
        self.continuous_columns = [column for column in self.data.columns if column not in self.discrete_columns]
        log.logger.info(f"Continuous Columns are : {self.continuous_columns}")

    def prepare_data(self):
        """
        Reads data and finds out continuous columns
        :return:
        """
        log.logger.info(f"Reading data from {self.input_data_path}")
        self.read_data()
        self.set_continuous_columns()

    def get_null_columns(self):
        """
        Returns columns dict which has null values
        """
        null_count_series = self.data.isnull().sum()
        null_count_dict = null_count_series.to_dict()  # gets {"<column>" :  <null_count> int }
        return {column: null_count_dict[column] for column in null_count_dict.keys() if null_count_dict[column] != 0}

    def clean_data_with_mode_or_median(self):
        """
        Replaces empty data with either mode(for discrete data)  and median(for continuous data)
        :return:
        """
        log.logger.info("Replacing empty data with either mode / median")
        null_columns_dict = self.get_null_columns()
        self.cleaned_data = self.data.copy()
        column_data_types = self.cleaned_data.dtypes
        for column in null_columns_dict.keys():
            replaceable_value = self.cleaned_data[column].median() if column_data_types[column] != "O" else \
                self.cleaned_data[column].mode()[0]
            self.cleaned_data[column] = self.cleaned_data[column].fillna(replaceable_value)

    def clean_data_with_knnimputer(self):
        """
        Replaces empty data using KNN imputer method
        :return:
        """
        log.logger.info("Replacing empty data using KNN Imputer method")
        null_columns_dict = self.get_null_columns()
        knn_imp = KNNImputer(n_neighbors=4, weights="uniform")
        self.cleaned_data = pd.DataFrame()
        knn_cleaned_data = knn_imp.fit_transform(self.data[null_columns_dict.keys()])
        knn_cleaned_data = pd.DataFrame(knn_cleaned_data, columns=null_columns_dict.keys())
        for pos, column in enumerate(self.data.columns):
            column_data = self.data[column] if column not in knn_cleaned_data.columns else knn_cleaned_data[column]
            self.cleaned_data.insert(pos, column, column_data, True)

    def clean_data_with_normalisation(self):
        """
        Applying Normalisation technique on cleaned data
        """
        log.logger.info("Normalising all continuous columns ")
        self.clean_data_with_knnimputer()
        # self.cleaned_data = self.data.copy()
        for column in self.cleaned_data.columns:
            if column in self.continuous_columns:
                column_min = self.cleaned_data[column].min()
                column_max = self.cleaned_data[column].max()
                self.cleaned_data[column] = (self.cleaned_data[column] - column_min)/(column_max - column_min)

    def clean_data_with_standardisation(self):
        """
        Applying standardisation technique on cleaned data
        """
        log.logger.info("Standardising all continuous columns ")
        self.clean_data_with_knnimputer()
        for column in self.cleaned_data.columns:
            if column in self.continuous_columns:
                column_std = self.cleaned_data[column].std()
                column_mean = self.cleaned_data[column].mean()
                self.cleaned_data[column] = (self.cleaned_data[column] - column_mean) / (column_std)

    def remove_missing_columns(self):
        """
        Removes columns which has missing data > 10% of actual data
        """
        log.logger.info("Removing columns with missing data more than 10%")
        null_columns_dict = self.get_null_columns()
        num_of_rows = self.data.shape[0]
        for null_column in null_columns_dict.keys():
            if null_columns_dict[null_column]/num_of_rows > 0.10:
                self.data = self.data.drop(null_column, axis=1)

    def remove_outliers(self):
        """
        Calculates Outliers and removes respective rows,
         and also columns whose outliers count is > 5% of data
        """
        log.logger.info("Processing Outliers")
        outliers_dict = {}
        num_of_rows = self.data.shape[0]
        column_data_types = self.cleaned_data.dtypes
        for column in self.cleaned_data.columns:
            outliers_dict[column] = calculate_outliers(self.cleaned_data[column]) if column_data_types[
                                                                                         column] != "O" else set()
        # Sample output of outliers_dict {'Country': set(), 'Life expectancy': {2306, 2307, 2308}}
        rows_to_be_removed = set()
        for column in outliers_dict.keys():
            if len(outliers_dict[column])/num_of_rows > 0.05:
                log.logger.debug(f"Dropping column {column} as it has {len(outliers_dict[column])} outliers ie > 5%")
                self.cleaned_data = self.cleaned_data.drop(column, axis=1)
            elif len(outliers_dict[column]) > 0 and len(outliers_dict[column])/num_of_rows <= 0.05:
                log.logger.debug(f"Dropping {len(outliers_dict[column])} outliers from '{column}' ")
                rows_to_be_removed = rows_to_be_removed.union(outliers_dict[column])
        log.logger.info(f"Total outliers removed are {len(rows_to_be_removed)}")
        self.cleaned_data = self.cleaned_data.drop(rows_to_be_removed)

    def one_hot_encoding(self):
        """
        Encodes discrete variables
        """
        log.logger.info("One hot encoding discrete columns which has < 8 unique values")
        for discrete_column in self.discrete_columns:
            unique_val_count = len(self.cleaned_data[discrete_column].unique())
            log.logger.debug(f"Number of unique values of {discrete_column} are "
                            f"{unique_val_count}")
            if unique_val_count < 8:
                log.logger.info(f"Encoding '{discrete_column}' column")
                self.cleaned_data = pd.get_dummies(self.cleaned_data, columns=[discrete_column])
                self.discrete_columns.remove(discrete_column)

    def clean_data_with_pca(self, n_components=1):
        """
        Reduction of dimensions of data using PCA
        """
        log.logger.info("Reducing dimensionality using PCA")
        backup_discrete_col = self.discrete_columns.copy()
        self.remove_missing_columns()
        self.clean_data_with_knnimputer()
        self.remove_outliers()
        self.one_hot_encoding()
        pca = PCA(n_components=n_components)
        pca_data = self.cleaned_data.drop(self.discrete_columns, axis=1)  # PCA works with continuous columns
        pca_data = pca_data.drop([self.output_component], axis=1)  # PCA works only with input data
        self.pca_data = pca.fit_transform(pca_data)
        log.logger.info(f"Variance ratio : {pca.explained_variance_ratio_}")
        self.discrete_columns = backup_discrete_col.copy()

    def get_highly_correlated_columns(self):
        """
        Calculates correlation between columns and returns which are higher
        :return: highly correlated columns
        """
        correlation = self.cleaned_data.corr()
        highly_correlated_columns = {}
        for column1 in correlation.columns:
            for column2 in correlation.columns:
                if column1 == column2:
                    continue
                if correlation[column1][column2] >= 0.90 or correlation[column1][column2] <= -0.90:
                    if column2 in highly_correlated_columns.keys() and highly_correlated_columns[column2] == column1:
                        continue
                    highly_correlated_columns[column1] = column2
                    log.logger.debug(f"{column1} and {column2} are highly correlated:  {correlation[column1][column2]}")
        return highly_correlated_columns.keys()

    def drop_higly_correlated_columns(self):
        """
        Drops highly correlated columns on cleaned data
        :return:
        """
        log.logger.info("Calculating highly correlated columns")
        columns_tobe_dropped = self.get_highly_correlated_columns()
        log.logger.info(f"Dropping highly correlated columns {columns_tobe_dropped}")
        self.cleaned_data = self.cleaned_data.drop(columns_tobe_dropped, axis=1)

    def read_data(self):
        """
        Reads data from input data path
        :return:
        """
        self.data = pd.read_csv(self.input_data_path)

    def train_model(self, method_name, random_state=2):
        """
        Trains the model using given data manipulation method/technique
        :param method_name: data manipulation method
        :param random_state: Random state value
        :return:
        """
        if "PCA" in method_name:
            train_data = self.pca_data
        else:
            columns_tobe_dropped = self.get_highly_correlated_columns()
            train_columns = [column for column in self.cleaned_data.columns if column not in columns_tobe_dropped]
            train_columns.remove(self.output_component)
            [train_columns.remove(column) for column in self.discrete_columns]
            train_data = self.cleaned_data[train_columns]

        in_train, in_test, out_train, out_test = train_test_split(train_data,
                                                                  self.cleaned_data[self.output_component],
                                                                  test_size=0.1, random_state=random_state)
        self.trained_model = linear_model.LinearRegression()
        self.trained_model.fit(in_train, out_train)
        self.in_test, self.out_test = in_test, out_test

    def calculate_r2(self):
        """
        Calculates R2 for the trained model
        """

        predicted = self.trained_model.predict(self.in_test)
        return r2_score(self.out_test, predicted)

    def formulate_r2(self, method_name):
        """
        Trains the linear regression model and calculates R2
        :param method_name: Data manipulation technique
        """
        # for p_num in range(400, 500):
        log.logger.info(f"Formulating R2 using {method_name}")
        breakpoint()
        for p_num in self.prime_num_list:
            self.train_model(method_name, random_state=p_num)
            r2 = self.calculate_r2()
            self.r2_results[method_name].append(round(r2, 2))

    def calculate_max_r2(self):
        """
        Creates a Tabular for Max R2 value from r2_results data
        """
        max_r2_dict = {"Method Used": [], "Random State": [], "Max R2": []}
        for result_col in self.r2_results.keys():
            if result_col == "Random State":
                continue
            column_r2_data = self.r2_results[result_col].copy()
            column_r2_data.sort()
            max_r2 = column_r2_data[-1]
            row = self.r2_results[result_col].index(max_r2)
            random_state = self.r2_results["Random State"][row]
            max_r2_dict["Method Used"].append(result_col)
            max_r2_dict["Max R2"].append(max_r2)
            max_r2_dict["Random State"].append(random_state)

        max_r2_df = pd.DataFrame(data=max_r2_dict)
        log.logger.info("==================  Highest R2 Values ==================\n"
                        f"{max_r2_df}")
        # f"{results_df.max()[1:]} \n \n"
        # f"{results_df.idxmax()[1:]} \n \n"
        # f"{results_df.loc[results_df.idxmax()[1:]]}")

    def run(self):
        """
        Runs regression learning model using different techniques
        """
        log.logger.info("===== Starting Regression Learning =====")
        breakpoint()
        self.prepare_data()
        self.r2_results = {"Random State": self.prime_num_list, "Mode_or_Median": [], "KNNImputer": [],
                           "Normalisation": [], "Standardisation": [], "PCA1": [], "PCA2": [], "PCA3": [], "PCA4": [],
                           "PCA5": []}
        for result_col in self.r2_results.keys():
            if result_col == "Random State":
                continue
            log.logger.info(f"========= Method: {result_col} =========")
            if "PCA" in result_col:
                self.clean_data_with_pca(n_components=int(f"{result_col.split('PCA')[-1]}"))
            else:
                getattr(self, f"clean_data_with_{result_col.lower()}")()
                self.drop_higly_correlated_columns()
            self.formulate_r2(method_name=result_col)
        results_df = pd.DataFrame(data=self.r2_results)
        log.logger.info("==================  R2 Analysis Report  ================== \n"
                        # f"{results_df.to_string()}")
                        f"{results_df}")
        self.calculate_max_r2()

        breakpoint()


def main():
    """
    Main method
    """
    start_time = f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
    log_filename = f"regression_learning_{start_time}.log"
    initiate_logger("regression-learning", log_filename, level="DEBUG")
    rl = RegressionLearning(input_data_path="./gr3regression.csv",
                            output_component="selling_price",
                            discrete_comp=["name", "year", "fuel", "seller_type", "transmission", "owner", "seats"])
    rl.run()


if __name__ == '__main__':
    main()
