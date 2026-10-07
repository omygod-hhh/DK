"""
Simplified Full Data Parser
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import sys

# Force UTF-8 output
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def parse_full_data_simple(csv_file='ph_ds_iot_data_all_202603181049.csv',
                           output_dir='./processed_data',
                           chunk_size=100000):
    """
    Simplified full data parser
    """
    start_time = datetime.now()
    print("="*60)
    print("Full Data Parser - Simplified Version")
    print("="*60)
    print(f"Input: {csv_file}")
    print(f"Chunk size: {chunk_size:,}")
    print(f"Start: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    sys.stdout.flush()

    csv_path = Path(csv_file)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Stats
    total_rows = 0
    parsed_rows = 0
    error_count = 0
    all_data = []

    # Process chunks
    chunk_id = 0
    print("\nProcessing data in chunks...")
    sys.stdout.flush()

    for chunk_df in pd.read_csv(csv_path, chunksize=chunk_size):
        chunk_id += 1
        total_rows += len(chunk_df)

        # Parse JSON
        json_data = []
        for idx, row in chunk_df.iterrows():
            try:
                data_dict = json.loads(row['address_value'])
                record = {
                    'log_id': row['log_id'],
                    'device_code': row['device_code'],
                    'timestamp': row['create_time'],
                    'start_address': row['start_address'],
                }
                record.update(data_dict)
                json_data.append(record)
            except Exception as e:
                error_count += 1

        if json_data:
            parsed_df = pd.DataFrame(json_data)
            parsed_df['timestamp'] = pd.to_datetime(parsed_df['timestamp'])
            all_data.append(parsed_df)
            parsed_rows += len(parsed_df)

        # Progress
        if chunk_id % 5 == 0:
            elapsed = (datetime.now() - start_time).total_seconds()
            speed = total_rows / elapsed if elapsed > 0 else 0
            print(f"  Chunk {chunk_id:4d} | Total: {total_rows:10,} | "
                  f"Parsed: {parsed_rows:10,} | Speed: {speed:10,.0f} rows/s")
            sys.stdout.flush()

    # Merge all
    print(f"\nMerging {len(all_data)} chunks...")
    sys.stdout.flush()

    full_df = pd.concat(all_data, ignore_index=True)

    print(f"Total: {total_rows:,} rows")
    print(f"Parsed: {parsed_rows:,} rows")
    print(f"Errors: {error_count:,}")
    sys.stdout.flush()

    # Clean data
    print(f"\nCleaning data...")
    sys.stdout.flush()

    meta_cols = {'log_id', 'device_code', 'timestamp', 'start_address'}
    data_cols = [col for col in full_df.columns if col not in meta_cols]

    # Convert types
    for col in data_cols:
        full_df[col] = pd.to_numeric(full_df[col], errors='coerce')

    # Sort
    full_df = full_df.sort_values('timestamp').reset_index(drop=True)

    # Forward fill
    full_df[data_cols] = full_df[data_cols].ffill()
    full_df[data_cols] = full_df[data_cols].bfill()

    print(f"Data shape: {full_df.shape}")
    sys.stdout.flush()

    # Save by address
    print(f"\nSaving by address type...")
    sys.stdout.flush()

    address_counts = full_df['start_address'].value_counts()

    for addr in sorted(full_df['start_address'].unique()):
        addr_df = full_df[full_df['start_address'] == addr].copy()
        addr_df = addr_df.drop(columns=['start_address'])

        output_file = output_path / f"data_address_{addr}.csv"
        addr_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"  Address {addr:6d}: {len(addr_df):8,} rows -> {output_file.name}")
        sys.stdout.flush()

    # Save merged
    print(f"\nSaving merged data...")
    sys.stdout.flush()

    merged_file = output_path / "data_merged.csv"
    full_df.to_csv(merged_file, index=False, encoding='utf-8-sig')
    print(f"  Merged: {len(full_df):,} rows -> data_merged.csv")
    sys.stdout.flush()

    # Save statistics
    print(f"\nGenerating statistics...")
    sys.stdout.flush()

    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()

    stats = {
        'start_time': str(start_time),
        'end_time': str(end_time),
        'duration_seconds': elapsed,
        'duration_minutes': elapsed / 60,
        'speed_rows_per_second': total_rows / elapsed if elapsed > 0 else 0,
        'total_rows': int(total_rows),
        'parsed_rows': int(parsed_rows),
        'error_rows': int(error_count),
        'success_rate': float(parsed_rows / total_rows * 100) if total_rows > 0 else 0,
        'total_fields': int(len(full_df.columns)),
        'time_range': {
            'start': str(full_df['timestamp'].min()),
            'end': str(full_df['timestamp'].max())
        },
        'address_distribution': {str(k): int(v) for k, v in address_counts.items()}
    }

    # JSON stats
    with open(output_path / 'full_data_summary.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2, default=str)

    # Text stats
    with open(output_path / 'full_data_statistics.txt', 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("Full Data Statistics Report\n")
        f.write("="*60 + "\n\n")
        f.write(f"Start time: {start_time}\n")
        f.write(f"End time: {end_time}\n")
        f.write(f"Duration: {elapsed/60:.1f} minutes\n")
        f.write(f"Speed: {total_rows/elapsed:,.0f} rows/s\n\n")
        f.write(f"Total rows: {total_rows:,}\n")
        f.write(f"Parsed rows: {parsed_rows:,}\n")
        f.write(f"Errors: {error_count:,}\n")
        f.write(f"Success rate: {parsed_rows/total_rows*100:.2f}%\n")
        f.write(f"Fields: {len(full_df.columns)}\n\n")
        f.write(f"Time range:\n")
        f.write(f"  Start: {full_df['timestamp'].min()}\n")
        f.write(f"  End: {full_df['timestamp'].max()}\n\n")
        f.write("Address distribution:\n")
        for addr, count in address_counts.items():
            f.write(f"  {addr:6d}: {count:8,} ({count/len(full_df)*100:5.2f}%)\n")
        f.write("\n")

    print(f"Statistics saved.")
    sys.stdout.flush()

    print(f"\n{'='*60}")
    print(f"Complete!")
    print(f"{'='*60}")
    print(f"Duration: {elapsed/60:.1f} minutes")
    print(f"Files saved to: {output_path.absolute()}")
    sys.stdout.flush()


if __name__ == "__main__":
    parse_full_data_simple()
