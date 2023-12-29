"""A module to check and verify derived statistics."""
from io import StringIO

import numpy as np
import pandas as pd


class UsefulStatistics:
    """A class containing methods for verifying derived statistics."""

    @staticmethod
    def statistics_check(input_obj):  # noqa: PLR0911, PLR0912
        """Check the statistics of an object."""
        if isinstance(input_obj, tuple):
            return [UsefulStatistics.statistics_check(inp) for inp in input_obj]

        elif isinstance(input_obj, pd.core.frame.DataFrame):
            if len(input_obj.columns) == 0:
                return pd.DataFrame()
            try:
                num_stats = input_obj.select_dtypes(include=np.number).describe()
                num_stats.loc["skew"] = input_obj[num_stats.columns].skew(
                    axis=0, skipna=True
                )
                num_stats_not_exist = False
            except ValueError:
                num_stats_not_exist = True
                pass
            try:
                cat_stats = (
                    input_obj.select_dtypes(exclude=np.number).applymap(str).describe()
                )
            except ValueError:
                return num_stats
            out = (
                cat_stats
                if num_stats_not_exist
                else pd.concat([num_stats, cat_stats], axis=1)
            )
            out.loc["dtype"] = out.dtypes.astype(str)
            return out.replace({pd.NA: str(pd.NA)})

        elif isinstance(input_obj, pd.core.series.Series):
            out = input_obj.describe().to_frame()
            out.loc["dtype"] = out.dtypes.astype(str)
            return out.replace({pd.NA: str(pd.NA)})

        elif isinstance(input_obj, (np.ndarray, list)):
            out = UsefulStatistics.statistics_check(pd.DataFrame(input_obj))
            column_suffix = str(type(input_obj))[8:-2]
            if len(out.columns) == 1:
                out.columns = [column_suffix]
            else:
                out.columns = [f"{col}_{column_suffix}" for col in out.columns]
            return out

        elif isinstance(input_obj, dict):
            out = pd.DataFrame(
                {k: pd.Series(v) for k, v in input_obj.items()}
            ).T.reset_index()
            out.columns = [
                "dict_keys" if i == 0 else f"dict_values_{i-1}"
                for i in range(len(out.columns))
            ]
            return UsefulStatistics.statistics_check(out)

        elif isinstance(input_obj, str):
            return pd.DataFrame(
                {"string_length": len(input_obj), "value": input_obj[:1000]}, index=[0]
            )

        elif str(type(input_obj))[8:-2].startswith("pandas.core.indexes"):
            return UsefulStatistics.statistics_check(
                pd.DataFrame(list(input_obj), columns=[str(type(input_obj))[8:-2]])
            )

        elif isinstance(input_obj, (int, float)):
            return input_obj

    @staticmethod
    def convert_df_to_str(df):
        """Convert a pandas dataframe to a string."""
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=True)
        return csv_buffer.getvalue()
