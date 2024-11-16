import os
import pandas as pd
import argparse
import sys

def list_columns(csv_path, show_dtypes=False, show_sample=False, sample_size=5, filter_type=None):
    """
    Lists column names from a CSV file and optionally displays data types and sample data.

    Args:
        csv_path (str): Path to the CSV file.
        show_dtypes (bool): Whether to display data types of columns.
        show_sample (bool): Whether to display a sample of the data.
        sample_size (int): Number of sample rows to display.
        filter_type (str): Data type to filter columns (e.g., 'object', 'int64', 'float64').

    Returns:
        None
    """
    if not os.path.exists(csv_path):
        print(f"Error: The file '{csv_path}' does not exist.")
        sys.exit(1)
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading '{csv_path}': {e}")
        sys.exit(1)
    
    print(f"\n📄 Inspecting CSV File: {csv_path}\n")
    
    # Filter columns by data type if specified
    if filter_type:
        filtered_columns = df.select_dtypes(include=[filter_type]).columns.tolist()
        if not filtered_columns:
            print(f"No columns of type '{filter_type}' found.")
            return
        print(f"🔍 Columns with data type '{filter_type}':")
        for col in filtered_columns:
            print(f" - {col}")
    else:
        print("📋 All Column Names:")
        for col in df.columns:
            print(f" - {col}")
    
    # Show data types if requested
    if show_dtypes:
        print("\n🔢 Column Data Types:")
        print(df.dtypes.to_string())
    
    # Show sample data if requested
    if show_sample:
        print(f"\n📝 Sample Data (First {sample_size} Rows):")
        print(df.head(sample_size).to_string(index=False))
    
    print("\n📈 Inspection Complete.\n")

def main():
    parser = argparse.ArgumentParser(description="Inspect a CSV file: list columns, data types, and view samples.")
    parser.add_argument('csv_path', type=str, help='Path to the CSV file.')
    parser.add_argument('--dtypes', action='store_true', help='Show data types of columns.')
    parser.add_argument('--sample', action='store_true', help='Show a sample of the data.')
    parser.add_argument('--sample_size', type=int, default=5, help='Number of sample rows to display (default: 5).')
    parser.add_argument('--filter_type', type=str, choices=['object', 'int64', 'float64', 'bool', 'datetime'], help='Filter columns by data type.')
    
    args = parser.parse_args()
    
    list_columns(
        csv_path=args.csv_path,
        show_dtypes=args.dtypes,
        show_sample=args.sample,
        sample_size=args.sample_size,
        filter_type=args.filter_type
    )

if __name__ == "__main__":
    main()

