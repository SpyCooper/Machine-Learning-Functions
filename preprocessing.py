# import the necessary packages
import numpy as np
import pandas as pd

'''
    Remove duplicates function
    - This function removes duplicate rows from the data.
    - It uses a set to store the unique rows and checks if each row is unique.

    Parameters:
    - data: the input data (Pandas DataFrame)

    Returns:
    - unique_rows: the data without duplicate rows (Pandas DataFrame)

    Example:
    data = pd.DataFrame({
        'A': [1, 2, 3, 1, 2],
        'B': [4, 5, 6, 4, 5],
        'B': [4, 5, 6, 4, 5]
    })
    unique_data = remove_duplicates(data)
    print(unique_data)
    # Output:
    #    A  B
    # 0  1  4
    # 1  2  5
    # 2  3  6
    # 3  1  4
    # 4  2  5
'''
def remove_duplicates(data):
    # create an empty set to store the unique rows
    # NOTE: a set is an unordered collection of unique elements meaning that it cannot contain duplicates
    seen = set()
    unique_rows = []

    # iterate through the rows of the data
    for index, row in data.iterrows():
        # convert the row to a tuple
        # NOTE: it has to be a tuple because lists are mutable and cannot be added to a set
        # NOTE: tuple can have an unlimited number of elements
        row_tuple = tuple(row)

        # check if the row is unique
        if row_tuple not in seen:
            # if the row is unique, add it to the unique rows list
            seen.add(row_tuple)
            unique_rows.append(row)
    
    # convert the unique rows list to a DataFrame and return it
    return pd.DataFrame(unique_rows)


'''
    Drop NA function
    - This function removes rows that contain missing values (NaN) from the data.
    - It checks if each row contains any missing values and only keeps the rows that do not have any missing values.

    Parameters:
    - data: the input data (Pandas DataFrame)

    Returns:
    - non_missing_rows: the data without rows that contain missing values (Pandas DataFrame)

    Example:
    data = pd.DataFrame({
        'A': [1, 2, np.nan, 4],
        'B': [5, 6, 7, 8]
    })
    non_missing_data = drop_na(data)
    print(non_missing_data)
    # Output:
    #    A  B
    # 0  1  5
    # 1  2  6
    # 3  4  8
'''
def drop_na(data):
    # create an array to store the rows that are not missing
    non_missing_rows = []

    # iterate through the rows of the data
    for index, row in data.iterrows():
        # check if the row is missing
        if not row.isnull().any():
            # if the row is not missing, add it to the non_missing_rows list
            non_missing_rows.append(row)

    # convert the non_missing_rows list to a DataFrame and return it
    return pd.DataFrame(non_missing_rows)


'''
    Remove outliers function
    - This function removes outliers from the data.
    - It can use three different methods to remove outliers: IQR, standard deviation, and z-score.
    - The IQR method removes outliers that are below the lower bound and above the upper bound.
    - The standard deviation method removes outliers that are below the lower bound and above the upper bound.
    - The z-score method removes outliers that are above the threshold.

    Parameters:
    - data: the input data (Pandas DataFrame) (NOTE: only numeric columns are considered)
    - method: the method to use to remove outliers (str) (default: 'iqr')
    - quartile_1: the first quartile (float) (used for IQR method) (default: 0.25)
    - quartile_3: the third quartile (float) (used for IQR method) (default: 0.75)
    - deviations: the number of standard deviations to consider (int) (used for standard deviation method) (default: 2)
    - threshold: the threshold to consider for the z-score method (int) (used for z-score method) (default: 2)
'''
def remove_outliers(data, method='iqr', quartile_1=0.25, quartile_3=0.75, deviations=2, threshold=2):    
    # check if the method is IQR (or default)
    if method == 'iqr':
        for column in data.columns:
            # check if the column is not numeric
            if not pd.api.types.is_numeric_dtype(data[column]):
                # if the column is not numeric, skip it
                continue
            # assuming all data is numeric...

            # calculate the first and third quartiles
            q1 = data[column].quantile(quartile_1)
            q3 = data[column].quantile(quartile_3)

            # calculate the interquartile range
            iqr = q3 - q1

            # calculate the lower and upper bounds
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            # filter out the outliers that are below the lower bound and above the upper bound
            data = data[(data[column] > lower_bound) & (data[column] < upper_bound)]

    # check if the method is standard deviation
    elif method == 'standard_deviation':
        for column in data.columns:
            # check if the column is not numeric
            if not pd.api.types.is_numeric_dtype(data[column]):
                # if the column is not numeric, skip it
                continue
            # assuming all data is numeric...

            # calculate the mean and standard deviation
            mean = data[column].mean()
            std = data[column].std()

            # calculate the lower and upper bounds
            lower_bound = mean - 2 * std
            upper_bound = mean + 2 * std

            # filter out the outliers that are below the lower bound and above the upper bound
            data = data[(data[column] > lower_bound) & (data[column] < upper_bound)]
    
    # check if the method is z-score
    elif method == 'z-score':
        # calculate the z-scores
        z_scores = np.abs(stats.zscore(data.select_dtypes(include=[np.number])))
        
        # filter out the outliers that are above the threshold
        filtered_entries = (z_scores < threshold).all(axis=1)
        
        # filter the data
        data = data[filtered_entries]
    
    # return the data without the outliers (if any)
    return data


'''
    Feature scaling function
    - This function scales the features of the data to be within a specific range.
    - It can use four different methods to scale the features: min-max, normalization, standardization, and robust scaling.
    - The min-max method scales the features to be between 0 and 1.
    - The normalization method scales the features to have a mean of 0 and a standard deviation of 1.
        ***** NOTE: the data should only contain numeric columns and the x values should be removed before normalizing.
    - The standardization method scales the features to have a mean of 0 and a standard deviation of 1.
    - The robust scaling method scales the features to have a median of 0 and an interquartile range of 1.

    Parameters:
    - data: the input data (Pandas DataFrame)
    - method: the method to use to scale the features (str) (default: 'min-max')

    Returns:
    - scaled_data: the data with the scaled features (Pandas DataFrame)
'''
def feature_scaling(data, method='min-max'):
    # check if the method is absolute max
    if method == "absolute_max":
        # iterate through the columns of the data
        for column in data.columns:
            # check if the column is not numeric
            if not pd.api.types.is_numeric_dtype(data[column]):
                # if the column is not numeric, skip it
                continue
            # assuming all data is numeric...

            # calculate the maximum values
            max_value = np.max(np.abs(data[column]))

            # scale the values to be between -1 and 1
            data[column] = data[column] / max_value
    
    # check if the method is min-max (or default)
    elif method == 'min-max':
        # iterate through the columns of the data
        for column in data.columns:
            # check if the column is not numeric
            if not pd.api.types.is_numeric_dtype(data[column]):
                # if the column is not numeric, skip it
                continue
            # assuming all data is numeric...

            # calculate the minimum and maximum values
            min_value = data[column].min()
            max_value = data[column].max()

            # scale the values to be between 0 and 1
            data[column] = (data[column] - min_value) / (max_value - min_value)
    
    # check if the method is normalization
    elif method == 'normalization':
        # iterate through the columns of the data
        for column in data.columns:
            # check if the column is not numeric
            if not pd.api.types.is_numeric_dtype(data[column]):
                # if the column is not numeric, skip it
                continue
            # assuming all data is numeric...

            # calculate the mean, minimum, and maximum values
            mean = data[column].mean()
            min_value = data[column].min()
            max_value = data[column].max()
            
            # scale the values to have a mean of 0 and a standard deviation of 1
            data[column] = (data[column] - mean) / (max_value - min_value)
    
    # check if the method is standardization
    elif method == 'standardization':
        # iterate through the columns of the data
        for column in data.columns:
            # check if the column is not numeric
            if not pd.api.types.is_numeric_dtype(data[column]):
                # if the column is not numeric, skip it
                continue
            # assuming all data is numeric...

            # calculate the mean and standard deviation
            mean = data[column].mean()
            std = data[column].std()

            # scale the values to have a mean of 0 and a standard deviation of 1
            data[column] = (data[column] - mean) / std

    # check if the method is robust scaling
    elif method == 'robust':
        # iterate through the columns of the data
        for column in data.columns:
            # check if the column is not numeric
            if not pd.api.types.is_numeric_dtype(data[column]):
                # if the column is not numeric, skip it
                continue
            # assuming all data is numeric...

            # calculate the first and third quartiles
            q1 = data[column].quantile(0.25)
            q3 = data[column].quantile(0.75)

            # calculate the interquartile range
            iqr = q3 - q1

            # scale the values to have a median of 0 and an interquartile range of 1
            data[column] = (data[column] - data[column].median()) / iqr

    # return the data with the scaled features
    return data


# TODO: feature filtering functions
def feature_filtering(data, method='correlation', threshold=0.5):
    # check if the method is correlation coefficient (or default)
    if method == 'correlation':
        # calculate the correlation matrix
        correlation_matrix = data.corr().abs()

        # filter out the features that are highly correlated
        upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(np.bool))
        to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
        data = data.drop(columns=to_drop)
    
    # check if the method is information gain
    elif method == 'information_gain':
        pass

    # check if the method is chi-squared
    elif method == 'chi-squared':
        pass

    # check if the method is fisher score
    elif method == 'fisher-score':
        pass

    # check if the method is variance threshold
    elif method == 'variance_threshold':
        pass

    # check if the method is mean absolute difference
    elif method == 'mad':
        pass

    # check if the method is dispersion ratio
    elif method == 'dispersion-ratio':
        pass

    # return the data with the selected features
    return data

data = pd.read_csv('SampleFile.csv')
personalized_data = feature_filtering(data)

# print the difference between the two datasets
print(personalized_data)