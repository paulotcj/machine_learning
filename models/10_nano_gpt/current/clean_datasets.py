import os
import shutil
from datasets import list_datasets, load_dataset

#-------------------------------------------------------------------------
def delete_all_datasets():
    # Get the list of all available datasets
    all_datasets = list_datasets()

    for dataset_name in all_datasets:
        try:
            # Load the dataset to get its cache files
            dataset = load_dataset(dataset_name, split='train')
            cache_files = dataset.cache_files

            for cache_file in cache_files:
                file_path = cache_file['filename']
                
                # Delete the dataset file
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"Deleted file: {file_path}")
                else:
                    print(f"File not found: {file_path}")

                # Delete the dataset directory if it exists
                dataset_dir = os.path.dirname(file_path)
                if os.path.exists(dataset_dir):
                    shutil.rmtree(dataset_dir)
                    print(f"Deleted directory: {dataset_dir}")
                else:
                    print(f"Directory not found: {dataset_dir}")

        except Exception as e:
            print(f"Could not process dataset {dataset_name}: {e}")
#-------------------------------------------------------------------------

delete_all_datasets()