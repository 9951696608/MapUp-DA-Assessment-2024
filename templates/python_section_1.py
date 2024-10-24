from typing import Dict, List

import pandas as pd


def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements.
    """
    # Your code goes here.
    result = []
    
    # Process the list in chunks of n
    for i in range(0, len(lst), n):
        # Get the current group
        group = lst[i:i+n]
        
        # Manually reverse the group
        reversed_group = []
        for j in range(len(group)):
            reversed_group.append(group[len(group) - 1 - j])
        
        # Append the reversed group to the result
        result.extend(reversed_group)
    
    return lst
print(reverse_by_n_elements([1, 2, 3, 4, 5, 6, 7, 8], 3))
print(reverse_by_n_elements([1, 2, 3, 4, 5], 2))
print(reverse_by_n_elements([10, 20, 30, 40, 50, 60, 70], 4))
    


def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.
    """
    # Your code here
    length_dict = {}

    # Iterate over each string in the list
    for string in lst:
        length = len(string)  # Get the length of the string
        if length not in length_dict:
            length_dict[length] = []  # Initialize a new list for this length
        length_dict[length].append(string)  # Append the string to the appropriate list

    # Sort the dictionary by keys (lengths)
    sorted_length_dict = dict(sorted(length_dict.items()))
    
    return sorted_length_dict
print(group_by_length(["apple", "bat", "car", "elephant", "dog", "bear"]))
print(group_by_length(["one", "two", "three", "four"]))

def flatten_dict(nested_dict: Dict, sep: str = '.') -> Dict:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.
    
    :param nested_dict: The dictionary object to flatten
    :param sep: The separator to use between parent and child keys (defaults to '.')
    :return: A flattened dictionary
    """
    # Your code here
    flat_dict = {}

    def flatten(current_dict, parent_key=''):
        for k, v in current_dict.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                flatten(v, new_key)
            elif isinstance(v, list):
                for i, item in enumerate(v):
                    flatten({f"{new_key}[{i}]": item})
            else:
                flat_dict[new_key] = v

    flatten(nested_dict)
    
    return flat_dict
nested_dict = {
    "road": {
        "name": "Highway 1",
        "length": 350,
        "sections": [
            {
                "id": 1,
                "condition": {
                    "pavement": "good",
                    "traffic": "moderate"
                }
            }
        ]
    }
}

# Call the function and print the output
flattened_output = flatten_dict(nested_dict)
print(flattened_output)

def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.
    
    :param nums: List of integers (may contain duplicates)
    :return: List of unique permutations
    """
    # Your code here
    def backtrack(start: int):
        if start == len(nums):
            result.append(nums[:])  # Append a copy of the current permutation
            return 
        seen = set()  # To track duplicates
        for i in range(start, len(nums)):
            if nums[i] not in seen:  # Check if the number has been used
                seen.add(nums[i])  # Mark the number as seen
                nums[start], nums[i] = nums[i], nums[start]  # Swap to create a new permutation
                backtrack(start + 1)  # Recurse with the next index
                nums[start], nums[i] = nums[i], nums[start]  # Swap back to backtrack

    nums.sort()  # Sort the array to handle duplicates
    result = []
    backtrack(0)
    return result

# Example usage:
input_nums = [1, 1, 2]
output = unique_permutations(input_nums)
print(output)


def find_all_dates(text: str) -> List[str]:
    """
    This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.
    
    Parameters:
    text (str): A string containing the dates in various formats.

    Returns:
    List[str]: A list of valid dates in the formats specified.
    """

    valid_dates = []
    length = len(text)
    
    for i in range(length):
        # Check for dd-mm-yyyy format
        if i + 9 < length and text[i + 2] == '-' and text[i + 5] == '-':
            day, month, year = '', '', ''
            for j in range(2):  # Get day
                day += text[i + j]
            for j in range(2):  # Get month
                month += text[i + 3 + j]
            for j in range(4):  # Get year
                year += text[i + 6 + j]
            if day.isdigit() and month.isdigit() and year.isdigit():
                valid_dates.append(text[i:i + 10])

        # Check for mm/dd/yyyy format
        if i + 9 < length and text[i + 2] == '/' and text[i + 5] == '/':
            month, day, year = '', '', ''
            for j in range(2):  # Get month
                month += text[i + j]
            for j in range(2):  # Get day
                day += text[i + 3 + j]
            for j in range(4):  # Get year
                year += text[i + 6 + j]
            if day.isdigit() and month.isdigit() and year.isdigit():
                valid_dates.append(text[i:i + 10])

        # Check for yyyy.mm.dd format
        if i + 9 < length and text[i + 4] == '.' and text[i + 7] == '.':
            year, month, day = '', '', ''
            for j in range(4):  # Get year
                year += text[i + j]
            for j in range(2):  # Get month
                month += text[i + 5 + j]
            for j in range(2):  # Get day
                day += text[i + 8 + j]
            if day.isdigit() and month.isdigit() and year.isdigit():
                valid_dates.append(text[i:i + 10])

    return valid_dates

# Example usage
text = "I was born on 23-08-1994, my friend on 08/23/1994, and another one on 1994.08.23."
found_dates = find_all_dates(text)
print(found_dates)
    

def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with latitude, longitude, and distance between consecutive points.
    
    Args:
        polyline_str (str): The encoded polyline string.

    Returns:
        pd.DataFrame: A DataFrame containing latitude, longitude, and distance in meters.
    """
    coordinates = decode_polyline(polyline_str)
    
    # Create lists for latitude, longitude, and distance
    latitudes = []
    longitudes = []
    distances = [0]  # First distance is 0
    
    # Calculate distances
    for i, (lat, lon) in enumerate(coordinates):
        latitudes.append(lat)
        longitudes.append(lon)
        if i > 0:  # Calculate distance only if there's a previous point
            dist = haversine(latitudes[i-1], longitudes[i-1], latitudes[i], longitudes[i])
            distances.append(dist)

    # Create DataFrame
    df = pd.DataFrame({
        'latitude': latitudes,
        'longitude': longitudes,
        'distance': distances
    })
    
    return df

# Example usage
polyline_str = "_p~iF~gs|U_rB`yU"  # Example polyline string
result_df = polyline_to_dataframe(polyline_str)
print(result_df)

def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Rotate the given matrix by 90 degrees clockwise, then multiply each element 
    by the sum of its original row and column index before rotation.
    
    Args:
    - matrix (List[List[int]]): 2D list representing the matrix to be transformed.
    
    Returns:
    - List[List[int]]: A new 2D list representing the transformed matrix.
    """
    # Your code here
    n = len(matrix)
    
    # Step 1: Rotate the matrix by 90 degrees clockwise
    rotated_matrix = [[matrix[n - j - 1][i] for j in range(n)] for i in range(n)]
    
    # Step 2: Calculate the new transformed matrix
    final_matrix = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            # Calculate the sum of the current row and column excluding the current element
            row_sum = sum(rotated_matrix[i])
            col_sum = sum(rotated_matrix[k][j] for k in range(n))
            final_matrix[i][j] = row_sum + col_sum - rotated_matrix[i][j]  # Exclude the current element
    return final_matrix
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
result = rotate_and_transform_matrix(matrix)
for row in result:
    print(row)


def time_check(df) -> pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
    # Write your logic here
    # Convert start and end timestamps to datetime
    df['start'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'])
    df['end'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'])

    # Group by (id, id_2)
    grouped = df.groupby(['id', 'id_2'])

    # Initialize a list to store results
    results = []

    for (id_val, id_2_val), group in grouped:
        # Create a full range of dates for the week
        all_days = pd.date_range(start=group['start'].min().normalize(), 
                                  end=group['end'].max().normalize(), freq='D')

        # Check for full 24-hour coverage for each day
        full_coverage = []
        for day in all_days:
            day_start = day + pd.Timedelta(hours=0)
            day_end = day + pd.Timedelta(hours=23, minutes=59, seconds=59)

            day_covered = any((group['start'] <= day_end) & (group['end'] >= day_start))
            full_coverage.append(day_covered)

        # Check if all 7 days are covered
        days_covered = len(full_coverage) == 7 and all(full_coverage)
        
        # Append the result as a tuple (id, id_2, result)
        results.append(((id_val, id_2_val), not days_covered))

    # Create a MultiIndex from results
    index = pd.MultiIndex.from_tuples([res[0] for res in results])
    boolean_series = pd.Series([res[1] for res in results], index=index)
    

    return boolean_series
# Example usage:
if __name__ == "__main__":
    # Sample DataFrame creation
    data = {
        'id': [1, 1, 1, 2, 2, 2],
        'id_2': ['A', 'A', 'A', 'B', 'B', 'B'],
        'startDay': ['2024-10-01', '2024-10-02', '2024-10-03', '2024-10-01', '2024-10-02', '2024-10-03'],
        'startTime': ['00:00:00', '00:00:00', '00:00:00', '00:00:00', '00:00:00', '00:00:00'],
        'endDay': ['2024-10-01', '2024-10-02', '2024-10-03', '2024-10-01', '2024-10-02', '2024-10-03'],
        'endTime': ['23:59:59', '23:59:59', '23:59:59', '23:59:59', '23:59:59', '23:59:59'],
    }

    df = pd.DataFrame(data)
    
    # Call the time_check function
    result = time_check(df)

    # Print the result
    print(result)
