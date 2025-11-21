import pandas as pd

from typing import List


class DataShifter:
    """
    This class is responsible for shifting data in a DataFrame.

    Args:
        None

    Returns:
        None
    """

    def __init__(self):
        pass

    def shifts_single_column(
        self, df: pd.DataFrame, shifts: List, target_column: str
    ) -> List[pd.DataFrame]:
        """
        Creates shifted versions of the dataset for a single column.

        Args:
            df (pd.DataFrame): Input dataframe.
            target_column (str): Column to apply the shift on.

        Returns:
            List[pd.DataFrame]: List of shifted dataframes.
        """
        data_shifted = []
        for value in shifts:
            df1 = df.copy()
            df1[target_column] = df1[target_column].shift(value)
            df1 = df1.dropna()
            data_shifted.append(df1)
        return data_shifted

    def shift_multiple_columns(
        self, df: pd.DataFrame, columns: List, shifts: List
    ) -> pd.DataFrame:
        """
        Generates shifted datasets for all columns, used for multiprocessing.

        Returns:
            List[Tuple[str, pd.DataFrame]]: List of tuples containing
            the column name and its corresponding shifted dataframe.
        """
        data_shifted = []
        for column in columns:
            for value in shifts:
                df1 = df.copy()
                df1[column] = df1[column].shift(value)
                df1 = df1.dropna()
                data_shifted.append((column, df1))  # Saves column and dataframe
        return data_shifted
