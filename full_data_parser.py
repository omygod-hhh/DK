"""
完整数据解析脚本 - 处理全部2.12GB数据
优化内存使用和性能
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import time


class FullDataParser:
    """完整数据解析器 - 优化内存使用"""

    def __init__(self, csv_path, chunk_size=50000, output_dir='./processed_data'):
        """
        初始化
        :param csv_path: CSV文件路径
        :param chunk_size: 分块大小(默认50000)
        :param output_dir: 输出目录
        """
        self.csv_path = Path(csv_path)
        self.chunk_size = chunk_size
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.stats = {
            'total_rows': 0,
            'parsed_rows': 0,
            'error_rows': 0,
            'start_time': None,
            'end_time': None
        }

        print("="*60)
        print("Full Data Parser")
        print("="*60)
        print(f"Input file: {self.csv_path}")
        print(f"File size: {self.csv_path.stat().st_size / 1024**3:.2f} GB")
        print(f"Chunk size: {self.chunk_size:,}")
        print(f"Output dir: {self.output_dir.absolute()}")

    def parse_chunk(self, chunk_df, chunk_id):
        """
        解析单个数据块
        :param chunk_df: 数据块DataFrame
        :param chunk_id: 块ID
        :return: 解析后的DataFrame
        """
        json_data = []
        error_count = 0

        for idx, row in chunk_df.iterrows():
            try:
                data_dict = json.loads(row['address_value'])
                record = {
                    'log_id': row['log_id'],
                    'device_code': row['device_code'],
                    'timestamp': row['create_time'],
                    'start_address': row['start_address'],
                    'create_date': row['create_date']
                }
                record.update(data_dict)
                json_data.append(record)
            except Exception as e:
                error_count += 1

        # 创建DataFrame
        if json_data:
            parsed_df = pd.DataFrame(json_data)
            parsed_df['timestamp'] = pd.to_datetime(parsed_df['timestamp'])
            return parsed_df, error_count
        else:
            return pd.DataFrame(), error_count

    def process_full_data(self):
        """处理完整数据 - 分块解析"""
        print(f"\n{'='*60}")
        print(f"Step 1: Processing Full Data")
        print(f"{'='*60}")

        self.stats['start_time'] = datetime.now()
        print(f"Start time: {self.stats['start_time'].strftime('%Y-%m-%d %H:%M:%S')}")

        # 估算总块数
        total_rows = sum(1 for _ in open(self.csv_path, 'r', encoding='utf-8')) - 1
        total_chunks = (total_rows // self.chunk_size) + 1

        print(f"Estimated rows: {total_rows:,}")
        print(f"Estimated chunks: {total_chunks:,}")
        print(f"\nProcessing...\n")

        # 分块读取和处理
        chunk_id = 0
        all_parsed_chunks = []

        for chunk_df in pd.read_csv(self.csv_path, chunksize=self.chunk_size):
            chunk_id += 1
            self.stats['total_rows'] += len(chunk_df)

            # 解析该块
            parsed_chunk, errors = self.parse_chunk(chunk_df, chunk_id)

            if not parsed_chunk.empty:
                all_parsed_chunks.append(parsed_chunk)
                self.stats['parsed_rows'] += len(parsed_chunk)

            self.stats['error_rows'] += errors

            # 显示进度
            if chunk_id % 10 == 0 or chunk_id == total_chunks:
                elapsed = (datetime.now() - self.stats['start_time']).total_seconds()
                progress = chunk_id / total_chunks * 100
                speed = self.stats['total_rows'] / elapsed if elapsed > 0 else 0

                print(f"  Chunk {chunk_id:5d}/{total_chunks} "
                      f"| Progress: {progress:5.1f}% "
                      f"| Speed: {speed:,.0f} rows/s "
                      f"| Total rows: {self.stats['total_rows']:,} "
                      f"| Parsed: {self.stats['parsed_rows']:,} "
                      f"| Errors: {self.stats['error_rows']:,}")

        # 合并所有块
        print(f"\nMerging {len(all_parsed_chunks)} chunks...")
        self.parsed_df = pd.concat(all_parsed_chunks, ignore_index=True)

        self.stats['end_time'] = datetime.now()
        elapsed = (self.stats['end_time'] - self.stats['start_time']).total_seconds()

        print(f"\nProcessing Complete:")
        print(f"  Total rows: {self.stats['total_rows']:,}")
        print(f"  Parsed rows: {self.stats['parsed_rows']:,}")
        print(f"  Error rows: {self.stats['error_rows']:,}")
        print(f"  Columns: {len(self.parsed_df.columns)}")
        print(f"  Time elapsed: {elapsed/60:.1f} minutes")
        print(f"  Average speed: {self.stats['total_rows']/elapsed:,.0f} rows/s")

        return self.parsed_df

    def clean_and_optimize(self):
        """数据清洗和优化"""
        print(f"\n{'='*60}")
        print(f"Step 2: Data Cleaning & Optimization")
        print(f"{'='*60}")

        initial_rows = len(self.parsed_df)

        # 1. 排序
        print(f"  Sorting by timestamp...")
        self.parsed_df = self.parsed_df.sort_values('timestamp').reset_index(drop=True)

        # 2. 识别数据列
        meta_cols = {'log_id', 'device_code', 'timestamp', 'start_address', 'create_date'}
        data_cols = [col for col in self.parsed_df.columns if col not in meta_cols]

        # 3. 转换数据类型 (优化内存)
        print(f"  Converting data types...")
        for col in data_cols:
            self.parsed_df[col] = pd.to_numeric(self.parsed_df[col], errors='coerce')

        # 4. 优化整数列的内存
        for col in self.parsed_df.select_dtypes(include=['float64']).columns:
            # 检查是否可以转为int
            if self.parsed_df[col].dropna().apply(float.is_integer).all():
                self.parsed_df[col] = self.parsed_df[col].astype('Int64')

        # 5. 缺失值统计
        print(f"  Analyzing missing values...")
        missing_stats = self.parsed_df[data_cols].isnull().sum()
        total_missing = missing_stats.sum()
        print(f"    Total missing: {total_missing:,}")

        # 6. 前向填充(基于时间序列)
        print(f"  Forward filling missing values...")
        self.parsed_df[data_cols] = self.parsed_df[data_cols].ffill()

        # 7. 后向填充(处理开头缺失)
        self.parsed_df[data_cols] = self.parsed_df[data_cols].bfill()

        # 8. 异常检测统计
        print(f"  Detecting outliers...")
        anomaly_count = 0
        for col in data_cols[:50]:  # 只检测前50个字段以节省时间
            if self.parsed_df[col].dtype in ['float64', 'Int64']:
                mean = self.parsed_df[col].mean()
                std = self.parsed_df[col].std()
                if std > 0:
                    anomalies = abs(self.parsed_df[col] - mean) > 5 * std
                    anomaly_count += anomalies.sum()

        print(f"    Detected outliers: {anomaly_count:,} (5σ rule)")

        final_rows = len(self.parsed_df)
        print(f"\nCleaning Complete:")
        print(f"  Initial rows: {initial_rows:,}")
        print(f"  Final rows: {final_rows:,}")
        print(f"  Missing values: {total_missing:,}")

        self.stats['cleaning'] = {
            'initial_rows': initial_rows,
            'final_rows': final_rows,
            'missing_values': int(total_missing),
            'anomalies': int(anomaly_count)
        }

    def save_by_address(self):
        """按地址类型分块保存"""
        print(f"\n{'='*60}")
        print(f"Step 3: Saving Data by Address Type")
        print(f"{'='*60}")

        # 按地址类型分组保存
        addr_counts = self.parsed_df['start_address'].value_counts()
        total_addresses = len(addr_counts)

        print(f"Found {total_addresses} address types")
        print(f"Processing...\n")

        saved_files = []

        for addr in sorted(self.parsed_df['start_address'].unique()):
            addr_df = self.parsed_df[self.parsed_df['start_address'] == addr].copy()
            addr_df = addr_df.drop(columns=['start_address'])

            # 优化内存
            for col in addr_df.select_dtypes(include=['float64']).columns:
                col_mean = addr_df[col].mean()
                if abs(col_mean - int(col_mean)) < 0.01:
                    addr_df[col] = addr_df[col].fillna(0).astype(int)

            output_file = self.output_dir / f"data_address_{addr}.csv"
            addr_df.to_csv(output_file, index=False, encoding='utf-8-sig', compression='infer')
            saved_files.append(output_file.name)

            print(f"  Address {addr:6d}: {len(addr_df):8,} rows -> {output_file.name}")

        # 保存完整合并数据 (使用压缩格式)
        print(f"\n  Saving merged data (this may take a while)...")
        merged_file = self.output_dir / "data_merged.csv"
        self.parsed_df.to_csv(merged_file, index=False, encoding='utf-8-sig')
        saved_files.append(merged_file.name)

        print(f"  Merged: {len(self.parsed_df):,} rows -> data_merged.csv")

        return saved_files

    def save_statistics(self):
        """保存统计报告"""
        print(f"\n{'='*60}")
        print(f"Step 4: Saving Statistics")
        print(f"{'='*60}")

        # 1. 详细统计报告
        stats_file = self.output_dir / "full_data_statistics.txt"
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("Full Data Statistics Report\n")
            f.write("="*60 + "\n\n")

            f.write("1. Processing Information\n")
            f.write("-"*60 + "\n")
            f.write(f"  Start time: {self.stats['start_time'].strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"  End time: {self.stats['end_time'].strftime('%Y-%m-%d %H:%M:%S')}\n")
            elapsed = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
            f.write(f"  Duration: {elapsed/60:.1f} minutes\n")
            f.write(f"  Speed: {self.stats['total_rows']/elapsed:,.0f} rows/s\n\n")

            f.write("2. Data Overview\n")
            f.write("-"*60 + "\n")
            f.write(f"  Total raw rows: {self.stats['total_rows']:,}\n")
            f.write(f"  Successfully parsed: {self.stats['parsed_rows']:,}\n")
            f.write(f"  Parse errors: {self.stats['error_rows']:,}\n")
            f.write(f"  Success rate: {self.stats['parsed_rows']/self.stats['total_rows']*100:.2f}%\n")
            f.write(f"  Total fields: {len(self.parsed_df.columns)}\n\n")

            f.write("3. Data Cleaning\n")
            f.write("-"*60 + "\n")
            f.write(f"  Rows before cleaning: {self.stats['cleaning']['initial_rows']:,}\n")
            f.write(f"  Rows after cleaning: {self.stats['cleaning']['final_rows']:,}\n")
            f.write(f"  Missing values filled: {self.stats['cleaning']['missing_values']:,}\n")
            f.write(f"  Outliers detected: {self.stats['cleaning']['anomalies']:,}\n\n")

            f.write("4. Time Range\n")
            f.write("-"*60 + "\n")
            f.write(f"  Start: {self.parsed_df['timestamp'].min()}\n")
            f.write(f"  End: {self.parsed_df['timestamp'].max()}\n")
            duration = self.parsed_df['timestamp'].max() - self.parsed_df['timestamp'].min()
            f.write(f"  Duration: {duration}\n\n")

            f.write("5. Data Distribution by Address Type\n")
            f.write("-"*60 + "\n")
            addr_counts = self.parsed_df['start_address'].value_counts()
            for addr, count in addr_counts.items():
                f.write(f"  Address {addr:6d}: {count:8,} rows ({count/len(self.parsed_df)*100:5.2f}%)\n")
            f.write("\n")

            f.write("6. Field List\n")
            f.write("-"*60 + "\n")
            meta_cols = {'log_id', 'device_code', 'timestamp', 'start_address', 'create_date'}
            data_cols = [col for col in self.parsed_df.columns if col not in meta_cols]

            for i, col in enumerate(self.parsed_df.columns, 1):
                if col in meta_cols:
                    f.write(f"  {i:3d}. [META] {col:25s}\n")
                else:
                    dtype = str(self.parsed_df[col].dtype)
                    non_null = self.parsed_df[col].notna().sum()
                    f.write(f"  {i:3d}. {col:25s} ({dtype:10s}) - {non_null:,} non-null\n")

            f.write("\n7. Field Statistics (Top 50 data fields)\n")
            f.write("-"*60 + "\n")
            f.write(f"{'Field':<10} {'Valid':>10} {'Missing':>10} {'Valid%':>8} "
                    f"{'Min':>12} {'Max':>12} {'Mean':>12} {'Std':>12}\n")
            f.write("-"*60 + "\n")

            field_stats = []
            for col in data_cols:
                non_null = self.parsed_df[col].notna().sum()
                missing = self.parsed_df[col].isna().sum()
                valid_pct = non_null / len(self.parsed_df) * 100

                if self.parsed_df[col].dtype in ['float64', 'Int64', 'int64']:
                    field_stats.append({
                        'field': col,
                        'valid': non_null,
                        'missing': missing,
                        'valid_pct': valid_pct,
                        'min': self.parsed_df[col].min(),
                        'max': self.parsed_df[col].max(),
                        'mean': self.parsed_df[col].mean(),
                        'std': self.parsed_df[col].std()
                    })

            # 按数据可用性排序
            field_stats.sort(key=lambda x: x['valid_pct'], reverse=True)

            for stat in field_stats[:50]:
                f.write(f"{stat['field']:<10} "
                        f"{stat['valid']:>10,} "
                        f"{stat['missing']:>10,} "
                        f"{stat['valid_pct']:>7.1f}% "
                        f"{stat['min']:>12.2f} "
                        f"{stat['max']:>12.2f} "
                        f"{stat['mean']:>12.2f} "
                        f"{stat['std']:>12.2f}\n")

        print(f"  Statistics report: full_data_statistics.txt")

        # 2. JSON格式统计
        summary = {
            'processing': {
                'start_time': str(self.stats['start_time']),
                'end_time': str(self.stats['end_time']),
                'duration_seconds': (self.stats['end_time'] - self.stats['start_time']).total_seconds(),
                'speed_rows_per_second': self.stats['total_rows'] / (self.stats['end_time'] - self.stats['start_time']).total_seconds()
            },
            'data': {
                'total_rows': self.stats['total_rows'],
                'parsed_rows': self.stats['parsed_rows'],
                'error_rows': self.stats['error_rows'],
                'success_rate': self.stats['parsed_rows'] / self.stats['total_rows'] * 100,
                'total_fields': len(self.parsed_df.columns),
                'time_range': {
                    'start': str(self.parsed_df['timestamp'].min()),
                    'end': str(self.parsed_df['timestamp'].max())
                }
            },
            'cleaning': self.stats['cleaning'],
            'address_distribution': self.parsed_df['start_address'].value_counts().to_dict()
        }

        summary_file = self.output_dir / "full_data_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

        print(f"  Summary JSON: full_data_summary.json")

    def generate_summary(self):
        """生成最终摘要"""
        print(f"\n{'='*60}")
        print(f"Full Data Parsing Complete!")
        print(f"{'='*60}")

        elapsed = (self.stats['end_time'] - self.stats['start_time']).total_seconds()

        print(f"\nProcessing Summary:")
        print(f"  Start: {self.stats['start_time'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  End: {self.stats['end_time'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Duration: {elapsed/60:.1f} minutes")
        print(f"  Speed: {self.stats['total_rows']/elapsed:,.0f} rows/s")
        print(f"\nData Summary:")
        print(f"  Total rows: {self.stats['total_rows']:,}")
        print(f"  Parsed rows: {self.stats['parsed_rows']:,}")
        print(f"  Errors: {self.stats['error_rows']:,}")
        print(f"  Success rate: {self.stats['parsed_rows']/self.stats['total_rows']*100:.2f}%")
        print(f"  Fields: {len(self.parsed_df.columns)}")
        print(f"  Time range: {self.parsed_df['timestamp'].min()} ~ {self.parsed_df['timestamp'].max()}")

        print(f"\nOutput Files:")
        print(f"  - data_merged.csv (complete data)")
        print(f"  - data_address_*.csv (by address type)")
        print(f"  - full_data_statistics.txt (detailed report)")
        print(f"  - full_data_summary.json (machine-readable)")

        print(f"\nNext Steps:")
        print(f"  1. Review full_data_statistics.txt")
        print(f"  2. Run field_mapping.py for field definitions")
        print(f"  3. Start feature engineering or modeling")


def main():
    """主函数"""
    print("\n" + "="*60)
    print("Full IoT Data Parser - Building Load Prediction")
    print("="*60)

    CSV_FILE = "ph_ds_iot_data_all_202603181049.csv"
    OUTPUT_DIR = "./processed_data"

    try:
        parser = FullDataParser(CSV_FILE, chunk_size=50000, output_dir=OUTPUT_DIR)

        # 步骤1: 处理完整数据
        parser.process_full_data()

        # 步骤2: 数据清洗
        parser.clean_and_optimize()

        # 步骤3: 保存数据
        parser.save_by_address()

        # 步骤4: 保存统计
        parser.save_statistics()

        # 生成摘要
        parser.generate_summary()

        print(f"\n{'='*60}")
        print(f"SUCCESS: Full data parsing complete!")
        print(f"{'='*60}\n")

    except KeyboardInterrupt:
        print(f"\n\nInterrupted by user. Partial results may be saved.")
    except Exception as e:
        print(f"\n\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
