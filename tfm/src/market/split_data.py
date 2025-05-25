def split_data(df, train_start, train_end, test_start, test_end):
    """
    Splits the DataFrame into training and testing sets based on date ranges.

    Args:
        df (pd.DataFrame): The DataFrame to split.
        train_start (str): Start date for the training set.
        train_end (str): End date for the training set.
        test_start (str): Start date for the testing set.
        test_end (str): End date for the testing set.

    Returns:
        tuple: Two DataFrames, one for training and one for testing.
    """
    train_df = df[(df["date"] >= train_start) & (df["date"] <= train_end)]
    test_df = df[(df["date"] >= test_start) & (df["date"] <= test_end)]
    return train_df, test_df