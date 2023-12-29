"""Test suite for the Useful class."""

import ast
import pathlib

import numpy as np
import pandas as pd

from useful import Useful

# Toggle testing outputs or generating outputs
UPDATE_TEST_OUTPUTS = False
KEYS_TO_NOT_TEST = {"started_at", "finished_at", "job_timestamp", "line_start", "error"}


def format_dictionary(d):
    """Format a dictionary to be compared with another dictionary."""
    return {
        k: format_dictionary(v) if isinstance(v, dict) else v
        for k, v in d.items()
        if k not in KEYS_TO_NOT_TEST
    }


def validate_output(func):
    """Validates or generates useful's data log outputs."""

    def wrapper(*args, **kwargs):
        usf = func(*args, **kwargs)
        output = str(usf.logger.all_data)
        path = pathlib.Path(f"test/expected_outputs/{func.__name__}.txt")

        if UPDATE_TEST_OUTPUTS:
            path.write_text(output)
        else:
            for line, expected_line in zip(
                ast.literal_eval(output), ast.literal_eval(path.read_text())
            ):
                cur_line = format_dictionary(ast.literal_eval(line))
                exp_line = format_dictionary(ast.literal_eval(expected_line))
                assert str(cur_line) == str(
                    exp_line
                ), f"Current line: {cur_line}\nExpected line: {exp_line}"

    return wrapper


@validate_output
def test_case_1():
    """Test case for Useful class, this includes a more random case."""
    usf = Useful("random_test", block_uploads=True, ignore_execution_errors=False)

    @usf.check(verbose=1)
    def func_a(a=1, b=2):
        return a + b

    @usf.check(verbose=1)
    def func_b(df, a=1):
        return func_a(a) * df

    func_b(a=func_a(a=2), df=pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}))

    return usf


@validate_output
def test_process_numpy_arrays_and_dicts():
    """Test case for processing numpy arrays and dictionaries."""
    usf = Useful(
        "test_np_array_and_dicts",
        block_uploads=True,
        ignore_execution_errors=False,
    )

    @usf.check()
    def add_arrays(arr1, arr2):
        return np.add(arr1, arr2)

    @usf.check()
    def square_array(arr):
        return np.square(arr)

    @usf.check()
    def calculate_statistics(arr):
        mean = np.mean(arr)
        std_dev = np.std(arr)
        return {"mean": mean, "std_dev": std_dev}

    array1 = np.array([1, 2, 3, 4, 5])
    array2 = np.array([5, 4, 3, 2, 1])

    calculate_statistics(square_array(add_arrays(array1, array2)))

    return usf


@validate_output
def test_process_strings_and_lists():
    """Test case for processing strings and lists."""
    usf = Useful(
        "test_strings_and_lists", block_uploads=True, ignore_execution_errors=False
    )

    @usf.check()
    def reverse_string(s):
        """Reverse a given string."""
        return s[::-1]

    @usf.check()
    def concatenate_strings(strings):
        """Concatenate a list of strings into a single string."""
        return " ".join(strings)

    input_strings = ["Hello", "World", "Python", "Programming"]

    reverse_string(concatenate_strings(input_strings))

    return usf


@validate_output
def test_process_pandas_dataframe():
    """Test case for processing pandas dataframe."""
    usf = Useful("test_pandas", block_uploads=True, ignore_execution_errors=False)

    @usf.check()
    def filter_dataframe(df, column_name, threshold=0):
        """
        Filter a pandas dataframe based on a column and threshold.

        Args:
        ----
            df (pd.DataFrame): The input dataframe.
            column_name (str): The name of the column to filter on.
            threshold (float, optional): The threshold value. Defaults to 0.

        Returns:
        -------
            pd.DataFrame: The filtered dataframe.
        """
        filtered_df = df[df[column_name] > threshold]
        return filtered_df

    data = {"A": [1, 2, 3, 4, 5], "B": [10, 20, 30, 40, 50]}

    df = pd.DataFrame(data)

    filter_dataframe(df, column_name="B", threshold=25)

    return usf


@validate_output
def test_perform_complex_operations():
    """Test case for performing complex operations."""
    usf = Useful(
        "test_perform_complex_ops", block_uploads=True, ignore_execution_errors=False
    )

    @usf.check()
    def perform_operations(arr):
        arr_squared = np.square(arr)
        arr_sum = np.sum(arr_squared)
        return np.sqrt(arr_sum)

    input_list = [1, 2, 3, 4, 5]
    perform_operations(np.array(input_list))

    return usf


@validate_output
def test_process_nested_functions():
    """Test case for processing nested functions."""
    usf = Useful(
        "process_nested_functions_test",
        block_uploads=True,
        ignore_execution_errors=False,
    )

    @usf.check()
    def add(x, y):
        return x + y

    @usf.check()
    def subtract(x, y):
        return x - y

    @usf.check()
    def multiply(x, y):
        return x * y

    @usf.check()
    def divide(x, y):
        if y == 0:
            return "Cannot divide by zero"
        return x / y

    def calculate_expression(a, b, c):
        result1 = add(a, b)
        result2 = subtract(result1, c)
        result3 = multiply(result1, result2)
        result4 = divide(result3, 2)
        return result4

    a = 5
    b = 3
    c = 2

    calculate_expression(a, b, c)

    return usf


@validate_output
def test_process_keyword_arguments():
    """Test case for processing keyword arguments."""
    usf = Useful(
        "process_keyword_arguments_test",
        block_uploads=True,
        ignore_execution_errors=False,
    )

    @usf.check()
    def calculate_area(length=1, width=1, height=1):
        return length * width * height

    calculate_area()  # Use default values
    calculate_area(length=5, width=3, height=2)  # Provide custom values
    calculate_area(width=4)  # Provide some custom values

    return usf


@validate_output
def test_process_multiple_return_values():
    """Test case for processing multiple return values."""
    usf = Useful(
        "process_multiple_return_values_test",
        block_uploads=True,
        ignore_execution_errors=False,
    )

    @usf.check()
    def calculate_statistics(arr):
        mean = np.mean(arr)
        std_dev = np.std(arr)
        return mean, std_dev

    input_list = [1, 2, 3, 4, 5]
    mean, std_dev = calculate_statistics(input_list)

    return usf


@validate_output
def process_complex_data():
    """Process complex data."""
    usf = Useful(
        "process_complex_data_test", block_uploads=True, ignore_execution_errors=False
    )

    @usf.check()
    def normalize_array(arr):
        mean = np.mean(arr)
        std_dev = np.std(arr)
        return (arr - mean) / std_dev

    @usf.check()
    def calculate_stats(df):
        summary = df.describe()
        return summary

    def process_data(input_data):
        numpy_array = np.array(input_data)
        normalized_array = normalize_array(numpy_array)
        df = pd.DataFrame({"data": normalized_array})
        stats = calculate_stats(df)
        return stats

    input_data = [10, 15, 20, 25, 30]
    process_data(input_data)

    return usf


@validate_output
def test_process_recursive_function():
    """Process the recursive function."""
    usf = Useful(
        "process_recursive_function_test",
        block_uploads=True,
        ignore_execution_errors=False,
    )

    @usf.check()
    def factorial(n):
        if n == 0:
            return 1
        else:
            return n * factorial(n - 1)

    def compute_factorial_chain(start, end):
        result = 1
        for i in range(start, end + 1):
            result *= factorial(i)
        return result

    start_value = 1
    end_value = 5
    compute_factorial_chain(start_value, end_value)

    return usf


@validate_output
def test_process_complex_logic():
    """Process complex logic."""
    usf = Useful(
        "process_complex_logic_test", block_uploads=True, ignore_execution_errors=False
    )

    @usf.check()
    def compute_moving_average(data_series, window_size):
        moving_avg = data_series.rolling(window=window_size).mean()
        return moving_avg

    @usf.check()
    def analyze_data(data, threshold):
        data_series = pd.Series(data)
        moving_average = compute_moving_average(data_series, window_size=3)
        filtered_data = data_series[data_series > threshold]
        return {"moving_average": moving_average, "filtered_data": filtered_data}

    input_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    threshold_value = 4

    analyze_data(input_data, threshold_value)

    return usf


class Point:
    """A class to represent a point in 2D space."""

    def __init__(self, x, y):
        """Initialize Point object with x and y coordinates."""
        self.x = x
        self.y = y


@validate_output
def test_process_custom_classes_and_sets():
    """Process custom classes and sets."""
    usf = Useful(
        "process_custom_classes_and_sets_test",
        block_uploads=True,
        ignore_execution_errors=False,
    )

    @usf.check()
    def calculate_distance(point1, point2):
        return ((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2) ** 0.5

    @usf.check()
    def unique_elements(input_list):
        unique_set = set(input_list)
        return list(unique_set)

    point1 = Point(1, 2)
    point2 = Point(4, 6)
    calculate_distance(point1, point2)

    input_list = [1, 2, 2, 3, 4, 4, 5]
    unique_elements(input_list)

    return usf


@validate_output
def test_call_error():
    """Call a function that raises an error."""
    usf = Useful(
        "call_error_test",
        block_uploads=True,
        ignore_execution_errors=False,
    )

    @usf.check()
    def divide(x, y):
        return x / y

    has_failed = False
    try:
        divide("1", "0")
    except:  # noqa: E722
        has_failed = True
        pass

    assert has_failed

    return usf


@validate_output
def test_check_statistics_false():
    """Call a function that uses check_statistics=False."""
    usf = Useful(
        "check_statistics_false_test",
        block_uploads=True,
        ignore_execution_errors=False,
    )

    @usf.check(check_statistics=False)
    def factorial(n):
        if n == 0:
            return 1
        else:
            return n * factorial(n - 1)

    def compute_factorial_chain(start, end):
        result = 1
        for i in range(start, end + 1):
            result *= factorial(i)
        return result

    start_value = 1
    end_value = 5
    compute_factorial_chain(start_value, end_value)

    return usf
