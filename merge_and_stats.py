"""
合并已解析的地址数据文件并生成统计报告
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')


def merge_address_files(data_dir='./processed_data', output_file='data_merged.csv'):
    """
    合并所有地址数据文件
    :param data_dir: 数据目录
    :param output_file: 输出文件名
    """
    start_time = datetime.now()
    print("="*60)
    print("Merging Address Data Files")
    print("="*60)
    print(f"Start: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    data_path = Path(data_dir)

    # 查找所有地址文件
    address_files = sorted(data_path.glob('data_address_*.csv'))

    print(f"Found {len(address_files)} address files\n")

    # 读取并合并 (使用分块以节省内存)
    all_dfs = []
    total_rows = 0

    for i, addr_file in enumerate(address_files, 1):
        # 检查文件大小
        file_size_mb = addr_file.stat().st_size / 1024**2

        print(f"  Reading {addr_file.name} ({file_size_mb:.1f} MB)...", end=' ')

        # 对于大文件，使用分块读取
        if file_size_mb > 500:  # 大于500MB
            print(f"[Large file, chunked read]")
            chunks = []
            for chunk in pd.read_csv(addr_file, chunksize=50000):
                chunks.append(chunk)
            df = pd.concat(chunks, ignore_index=True)
        else:
            df = pd.read_csv(addr_file)

        total_rows += len(df)
        all_dfs.append(df)

        print(f"{len(df):,} rows")

        if i % 10 == 0:
            print(f"  Progress: {i}/{len(address_files)} files, {total_rows:,} total rows\n")
            # 强制垃圾回收
            import gc
            gc.collect()

    # 合并
    print(f"\nMerging {len(all_dfs)} dataframes...")
    merged_df = pd.concat(all_dfs, ignore_index=True)

    print(f"Merged shape: {merged_df.shape}")
    print(f"Total rows: {len(merged_df):,}")

    # 排序 (使用更内存友好的方式)
    if 'timestamp' in merged_df.columns:
        print(f"\nSorting by timestamp...")
        try:
            # 方法1: 直接排序
            merged_df['timestamp'] = pd.to_datetime(merged_df['timestamp'])
            merged_df = merged_df.sort_values('timestamp').reset_index(drop=True)
        except MemoryError:
            print(f"  Warning: Memory error during sort, trying alternative method...")
            # 方法2: 使用argsort
            merged_df['timestamp'] = pd.to_datetime(merged_df['timestamp'])
            idx = np.argsort(merged_df['timestamp'].values)
            merged_df = merged_df.iloc[idx].reset_index(drop=True)
            print(f"  Sorted using alternative method")

    print(f"Sort complete")

    # 保存
    print(f"\nSaving to {output_file}...")
    merged_df.to_csv(data_path / output_file, index=False, encoding='utf-8-sig')

    file_size = (data_path / output_file).stat().st_size / 1024**2
    print(f"Saved: {len(merged_df):,} rows, {file_size:.1f} MB")

    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()

    print(f"\nElapsed: {elapsed/60:.1f} minutes")
    print(f"\nMerge complete!")

    return merged_df


def generate_statistics(data_dir='./processed_data', merged_file='data_merged.csv'):
    """
    生成统计报告
    :param data_dir: 数据目录
    :param merged_file: 合并后的数据文件
    """
    print("\n" + "="*60)
    print("Generating Statistics Report")
    print("="*60)

    data_path = Path(data_dir)
    merged_path = data_path / merged_file

    if not merged_path.exists():
        print(f"Error: {merged_path} not found!")
        return

    print(f"Reading {merged_path}...")
    df = pd.read_csv(merged_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    print(f"Data shape: {df.shape}")

    # 统计信息
    meta_cols = {'log_id', 'device_code', 'timestamp', 'start_address', 'create_date'}
    data_cols = [col for col in df.columns if col not in meta_cols]

    # 地址分布
    if 'start_address' in df.columns:
        address_counts = df['start_address'].value_counts()
    else:
        address_counts = pd.Series()

    # 字段统计
    field_stats = []
    for col in data_cols:
        values = pd.to_numeric(df[col], errors='coerce')
        non_null = values.notna().sum()
        valid_pct = non_null / len(df) * 100

        stat = {
            'field': col,
            'valid': int(non_null),
            'missing': int(values.isna().sum()),
            'valid_pct': float(valid_pct)
        }

        if values.dtype in ['float64', 'int64']:
            stat.update({
                'min': float(values.min()) if not pd.isna(values.min()) else None,
                'max': float(values.max()) if not pd.isna(values.max()) else None,
                'mean': float(values.mean()) if not pd.isna(values.mean()) else None,
                'std': float(values.std()) if not pd.isna(values.std()) else None
            })

        field_stats.append(stat)

    # 排序
    field_stats.sort(key=lambda x: x['valid_pct'], reverse=True)

    # 生成JSON报告
    summary = {
        'data_overview': {
            'total_rows': len(df),
            'total_fields': len(df.columns),
            'time_range': {
                'start': str(df['timestamp'].min()),
                'end': str(df['timestamp'].max())
            }
        },
        'address_distribution': {str(k): int(v) for k, v in address_counts.items()},
        'field_statistics': field_stats[:100]  # Top 100 fields
    }

    summary_file = data_path / 'full_data_summary.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Summary JSON saved: {summary_file.name}")

    # 生成文本报告
    stats_file = data_path / 'full_data_statistics.txt'
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("Full Data Statistics Report\n")
        f.write("="*60 + "\n\n")

        f.write("1. Data Overview\n")
        f.write("-"*60 + "\n")
        f.write(f"  Total rows: {len(df):,}\n")
        f.write(f"  Total fields: {len(df.columns)}\n")
        f.write(f"  Time range: {df['timestamp'].min()} ~ {df['timestamp'].max()}\n\n")

        f.write("2. Address Distribution\n")
        f.write("-"*60 + "\n")
        for addr, count in address_counts.items():
            f.write(f"  {addr:6d}: {count:8,} ({count/len(df)*100:5.2f}%)\n")
        f.write("\n")

        f.write("3. Field Statistics (Top 100 by data availability)\n")
        f.write("-"*60 + "\n")
        f.write(f"{'Field':<12} {'Valid':>10} {'Missing':>10} {'Valid%':>8} "
                f"{'Min':>12} {'Max':>12} {'Mean':>12} {'Std':>12}\n")
        f.write("-"*60 + "\n")

        for stat in field_stats[:100]:
            min_val = f"{stat['min']:>12.2f}" if stat.get('min') is not None else f"{'N/A':>12}"
            max_val = f"{stat['max']:>12.2f}" if stat.get('max') is not None else f"{'N/A':>12}"
            mean_val = f"{stat['mean']:>12.2f}" if stat.get('mean') is not None else f"{'N/A':>12}"
            std_val = f"{stat['std']:>12.2f}" if stat.get('std') is not None else f"{'N/A':>12}"

            f.write(f"{stat['field']:<12} "
                    f"{stat['valid']:>10,} "
                    f"{stat['missing']:>10,} "
                    f"{stat['valid_pct']:>7.1f}% "
                    f"{min_val} "
                    f"{max_val} "
                    f"{mean_val} "
                    f"{std_val}\n")

    print(f"Statistics report saved: {stats_file.name}")

    print(f"\nStatistics generation complete!")


def main():
    """主函数"""
    print("\n" + "="*60)
    print("Data Merger & Statistics Generator")
    print("="*60)

    try:
        # 合并数据
        merged_df = merge_address_files()

        # 生成统计
        generate_statistics()

        print("\n" + "="*60)
        print("All Complete!")
        print("="*60)
        print(f"\nOutput files:")
        print(f"  - data_merged.csv")
        print(f"  - full_data_summary.json")
        print(f"  - full_data_statistics.txt")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
