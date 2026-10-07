"""
快速数据解析脚本 - 处理前10000行用于测试
"""

import pandas as pd
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def quick_parse_test():
    """快速解析测试 - 处理前10000行"""

    print("="*60)
    print("Quick Data Parser Test (10,000 rows)")
    print("="*60)

    csv_file = Path("ph_ds_iot_data_all_202603181049.csv")

    # 1. 读取前10000行
    print(f"\nStep 1: Loading 10,000 rows...")
    df = pd.read_csv(csv_file, nrows=10000)
    print(f"Loaded: {len(df):,} rows x {len(df.columns)} columns")

    # 2. 探索数据结构
    print(f"\nStep 2: Exploring structure...")
    print(f"Devices: {df['device_code'].nunique()}")
    print(f"Address Types: {sorted(df['start_address'].unique())}")
    print(f"Time Range: {df['create_time'].min()} ~ {df['create_time'].max()}")

    # 3. 解析JSON
    print(f"\nStep 3: Parsing JSON fields...")
    json_data = []

    for idx, row in df.iterrows():
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

            if (idx + 1) % 1000 == 0:
                print(f"  Processed {idx + 1:,} rows...")

        except Exception as e:
            print(f"Error at row {idx}: {e}")

    # 4. 创建DataFrame
    print(f"\nStep 4: Creating DataFrame...")
    parsed_df = pd.DataFrame(json_data)
    parsed_df['timestamp'] = pd.to_datetime(parsed_df['timestamp'])

    print(f"Created: {len(parsed_df):,} rows x {len(parsed_df.columns)} columns")

    # 5. 数据类型转换
    print(f"\nStep 5: Converting data types...")
    data_cols = [col for col in parsed_df.columns
                 if col not in ['log_id', 'device_code', 'timestamp', 'start_address']]

    for col in data_cols:
        parsed_df[col] = pd.to_numeric(parsed_df[col], errors='coerce')

    # 6. 缺失值分析
    print(f"\nStep 6: Analyzing missing values...")
    missing = parsed_df[data_cols].isnull().sum()
    total_missing = missing.sum()

    print(f"Total missing: {total_missing:,}")
    print(f"\nTop 20 fields with most missing values:")
    missing_top = missing.sort_values(ascending=False).head(20)
    for col, count in missing_top.items():
        pct = count / len(parsed_df) * 100
        print(f"  {col:10s}: {count:6,} ({pct:5.2f}%)")

    # 7. 基本统计
    print(f"\nStep 7: Basic statistics...")
    print(f"\nNumeric field statistics:")
    for col in data_cols[:10]:
        if parsed_df[col].dtype in ['float64', 'int64']:
            print(f"\n  {col}:")
            print(f"    Min: {parsed_df[col].min():.2f}")
            print(f"    Max: {parsed_df[col].max():.2f}")
            print(f"    Mean: {parsed_df[col].mean():.2f}")
            print(f"    Std: {parsed_df[col].std():.2f}")

    # 8. 保存数据
    print(f"\nStep 8: Saving data...")
    output_dir = Path("./processed_data")
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "data_test_10000.csv"
    parsed_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"Saved to: {output_file}")

    # 9. 保存统计报告
    stats_file = output_dir / "test_statistics.txt"
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("Data Parser Test Statistics\n")
        f.write("="*60 + "\n\n")

        f.write(f"Total rows: {len(parsed_df):,}\n")
        f.write(f"Total columns: {len(parsed_df.columns)}\n")
        f.write(f"Total missing: {total_missing:,}\n\n")

        f.write("Field List:\n")
        f.write("-"*60 + "\n")
        for i, col in enumerate(parsed_df.columns, 1):
            dtype = str(parsed_df[col].dtype)
            non_null = parsed_df[col].notna().sum()
            f.write(f"{i:3d}. {col:25s} ({dtype:10s}) - {non_null:,} non-null\n")

        f.write("\n\nMissing Value Statistics (Top 50):\n")
        f.write("-"*60 + "\n")
        missing_sorted = missing.sort_values(ascending=False).head(50)
        for col, count in missing_sorted.items():
            pct = count / len(parsed_df) * 100
            f.write(f"{col:10s}: {count:6,} ({pct:5.2f}%)\n")

    print(f"Statistics saved to: {stats_file}")

    print(f"\n{'='*60}")
    print(f"Test Complete!")
    print(f"{'='*60}")
    print(f"\nData saved in: {output_dir.absolute()}")
    print(f"\nCheck statistics.txt for detailed field information")


if __name__ == "__main__":
    quick_parse_test()
