import pandas as pd
import os

# Load the data from the CSV file
df = pd.read_csv('my_file.csv')

# Filter the DataFrame to get the rows where 'S/N (Arya)' is 'n/a'
df_na = df[df['S/N (Arya)'] == 'n/a']

# Get the list of numbers corresponding to 'n/a'
numbers_to_delete = df_na['430359'].tolist()

# Print the total number of files to delete
print(f'Number of files to delete: {len(numbers_to_delete)}')

# Loop through the list and delete the corresponding files
for number in numbers_to_delete:
    filename = f'AGC{number}.fits'  # using .fits as the file extension
    if os.path.exists(filename):
        try:
            os.remove(filename)
            print(f'Successfully deleted {filename}')
        except Exception as e:
            print(f'Error while trying to delete {filename}: {e}')
    else:
        print(f'{filename} not found')

