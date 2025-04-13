# Change all csv's to txts that are white space-seperated and remove the header
import os
import numpy as np

def convert_csv_to_txt(input_dir, output_dir):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate through all files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.csv'):
            input_file_path = os.path.join(input_dir, filename)
            output_file_path = os.path.join(output_dir, filename.replace('.csv', '.txt'))

            # Read the CSV file and skip the header
            data = np.loadtxt(input_file_path, delimiter=',', skiprows=1)

            # Save the data as a whitespace-separated text file with :.3e format
            np.savetxt(output_file_path, data, delimiter=' ', fmt='%.3e')
            print(f"Converted {input_file_path} to {output_file_path}")

if __name__ == "__main__":
    # convert_csv_to_txt('/data/azelakiewicz/Documents/exoEarth-HWO/data/surfaces', '/data/azelakiewicz/Documents/exoEarth-HWO/data/surfaces_txt')