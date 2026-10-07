"""
数据解析脚本 - 读取并解析IoT CSV数据
功能：
1. 读取原始CSV数据
2. 解析address_value JSON字段
3. 展开数据点为单独列
4. 处理时间戳
5. 数据清洗和质量检查
6. 保存为结构化格式
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class IoTDataParser:
    """IoT数据解析器"""

    def __init__(self, csv_path, chunk_size=10000):
        """
        初始化解析器
        :param csv_path: CSV文件路径
        :param chunk_size: 分块读取大小(用于大文件)
        """
        self.csv_path = Path(csv_path)
        self.chunk_size = chunk_size
        self.raw_df = None
        self.parsed_df = None
        self.stats = {}

        print(f"  File: {self.csv_path}")
        print(f"  Size: {self.csv_path.stat().st_size / 1024**3:.2f} GB")

    def load_raw_data(self, nrows=None, use_chunks=False):
        """
        加载原始CSV数据
        :param nrows: 读取行数(None=全部)
        :param use_chunks: 是否分块读取
        """
        print(f"\n{'='*60}")
        print(f"Step 1: Load Raw Data")
        print(f"{'='*60}")

        if use_chunks and nrows is None:
            # 分块读取大文件
            chunks = []
            for i, chunk in enumerate(pd.read_csv(self.csv_path, chunksize=self.chunk_size)):
                chunks.append(chunk)
                if i % 10 == 0:
                    print(f"  已读取 {i * self.chunk_size:,} 行...")
            self.raw_df = pd.concat(chunks, ignore_index=True)
        else:
            self.raw_df = pd.read_csv(self.csv_path, nrows=nrows)

        print(f"  Loaded: {len(self.raw_df):,} rows x {len(self.raw_df.columns)} columns")
        print(f"  Columns: {self.raw_df.columns.tolist()}")

        # 保存统计信息
        self.stats['total_rows'] = len(self.raw_df)
        self.stats['columns'] = self.raw_df.columns.tolist()

        return self.raw_df

    def explore_raw_data(self):
        """探索原始数据结构"""
        print(f"\n{'='*60}")
        print(f"Step 2: Explore Data Structure")
        print(f"{'='*60}")

        # 基本统计
        print(f"\nBasic Info:")
        print(f"  - Device: {self.raw_df['device_code'].unique()}")
        print(f"  - Address Types: {sorted(self.raw_df['start_address'].unique())}")
        print(f"  - Time Range: {self.raw_df['create_time'].min()} ~ {self.raw_df['create_time'].max()}")

        # 每种地址类型的数量
        print(f"\nData Distribution:")
        addr_counts = self.raw_df['start_address'].value_counts()
        for addr, count in addr_counts.items():
            print(f"  - Address {addr}: {count:,} records ({count/len(self.raw_df)*100:.1f}%)")

        # 时间戳分析
        self.raw_df['create_time'] = pd.to_datetime(self.raw_df['create_time'])
        print(f"\nTime Analysis:")
        print(f"  - Earliest: {self.raw_df['create_time'].min()}")
        print(f"  - Latest: {self.raw_df['create_time'].max()}")
        print(f"  - Duration: {self.raw_df['create_time'].max() - self.raw_df['create_time'].min()}")

        # 采样一个数据点查看JSON结构
        print(f"\nJSON Data Structure Example:")
        sample_json = self.raw_df['address_value'].iloc[0]
        sample_dict = json.loads(sample_json)
        print(f"  - Data Points: {len(sample_dict)}")
        print(f"  - First 10 keys: {list(sample_dict.keys())[:10]}")

        self.stats['data_distribution'] = addr_counts.to_dict()
        self.stats['time_range'] = {
            'start': str(self.raw_df['create_time'].min()),
            'end': str(self.raw_df['create_time'].max())
        }

    def parse_json_fields(self, sample_only=False):
        """
        解析JSON字段
        :param sample_only: 仅解析前100行用于测试
        """
        print(f"\n{'='*60}")
        print(f"Step 3: Parse JSON Fields")
        print(f"{'='*60}")

        df = self.raw_df if not sample_only else self.raw_df.head(100)

        print(f"  Parsing JSON fields for {len(df):,} records...")

        # 解析JSON并展开为DataFrame
        json_data = []
        error_count = 0

        for idx, row in df.iterrows():
            try:
                data_dict = json.loads(row['address_value'])
                # 保留原始数据用于对齐
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
                if error_count <= 5:
                    print(f"  ⚠️  解析错误 (行{idx}): {str(e)[:100]}")

        if error_count > 5:
            print(f"  ⚠️  共有 {error_count:,} 条记录解析失败 (前5条已显示)")

        # 创建新的DataFrame
        self.parsed_df = pd.DataFrame(json_data)

        # 转换时间戳
        self.parsed_df['timestamp'] = pd.to_datetime(self.parsed_df['timestamp'])

        print(f"JSON Parsing Complete:")
        print(f"  - Success: {len(self.parsed_df):,} records")
        print(f"  - Failed: {error_count:,} records")
        print(f"  - Fields: {len(self.parsed_df.columns)} columns")
        print(f"  - Time Range: {self.parsed_df['timestamp'].min()} ~ {self.parsed_df['timestamp'].max()}")

        self.stats['parsed_rows'] = len(self.parsed_df)
        self.stats['parsed_columns'] = len(self.parsed_df.columns)
        self.stats['json_parse_errors'] = error_count

        return self.parsed_df

    def clean_data(self):
        """数据清洗"""
        print(f"\n{'='*60}")
        print(f"Step 4: Data Cleaning")
        print(f"{'='*60}")

        initial_rows = len(self.parsed_df)

        # 转换数值字段
        print(f"  Converting data types...")
        data_columns = [col for col in self.parsed_df.columns
                       if col not in ['log_id', 'device_code', 'timestamp', 'start_address', 'create_date']]

        for col in data_columns:
            self.parsed_df[col] = pd.to_numeric(self.parsed_df[col], errors='coerce')

        # 处理缺失值
        print(f"  Checking missing values...")
        missing_stats = self.parsed_df[data_columns].isnull().sum()
        total_missing = missing_stats.sum()

        if total_missing > 0:
            print(f"  Found {total_missing:,} missing values")
            print(f"  Top missing fields:")
            missing_top = missing_stats[missing_stats > 0].sort_values(ascending=False).head(10)
            for col, count in missing_top.items():
                print(f"    - {col}: {count:,} ({count/len(self.parsed_df)*100:.2f}%)")

            # 填充策略: 前向填充(基于时间序列)
            self.parsed_df[data_columns] = self.parsed_df[data_columns].ffill()
            print(f"  Filled missing values using forward fill")
        else:
            print(f"  No missing values found")

        # 检测异常值 (简单3σ原则)
        print(f"  Detecting outliers (3-sigma rule)...")
        anomaly_count = 0
        for col in data_columns:
            if self.parsed_df[col].dtype in ['float64', 'int64']:
                mean = self.parsed_df[col].mean()
                std = self.parsed_df[col].std()
                if std > 0:
                    anomalies = abs(self.parsed_df[col] - mean) > 3 * std
                    anomaly_count += anomalies.sum()

        if anomaly_count > 0:
            print(f"  Found {anomaly_count:,} potential outliers (3-sigma rule)")
        else:
            print(f"  No obvious outliers detected")

        # 排序
        self.parsed_df = self.parsed_df.sort_values('timestamp').reset_index(drop=True)

        final_rows = len(self.parsed_df)
        removed_rows = initial_rows - final_rows
        print(f"\nData Cleaning Complete:")
        print(f"  - Original rows: {initial_rows:,}")
        print(f"  - Final rows: {final_rows:,}")
        print(f"  - Removed/Filled: {removed_rows:,}")

        self.stats['cleaning'] = {
            'initial_rows': initial_rows,
            'final_rows': final_rows,
            'missing_values_filled': int(total_missing),
            'anomalies_detected': int(anomaly_count)
        }

        return self.parsed_df

    def save_data(self, output_dir='./processed_data'):
        """
        保存处理后的数据
        :param output_dir: 输出目录
        """
        print(f"\n{'='*60}")
        print(f"Step 5: Save Processed Data")
        print(f"{'='*60}")

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # 按起始地址分开保存
        for addr in sorted(self.parsed_df['start_address'].unique()):
            addr_df = self.parsed_df[self.parsed_df['start_address'] == addr].copy()
            addr_df = addr_df.drop(columns=['start_address'])

            output_file = output_path / f"data_address_{addr}.csv"
            addr_df.to_csv(output_file, index=False, encoding='utf-8-sig')

            print(f"  Address {addr}: {len(addr_df):,} rows -> {output_file.name}")

        # 保存合并数据
        merged_file = output_path / "data_merged.csv"
        self.parsed_df.to_csv(merged_file, index=False, encoding='utf-8-sig')
        print(f"  Merged data: {len(self.parsed_df):,} rows -> {merged_file.name}")

        # 保存统计报告
        stats_file = output_path / "data_statistics.txt"
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("Data Statistics Report\n")
            f.write("=" * 60 + "\n\n")

            f.write("1. Basic Information\n")
            f.write("-" * 60 + "\n")
            f.write(f"  Raw data file: {self.csv_path.name}\n")
            f.write(f"  Time range: {self.stats['time_range']['start']} ~ {self.stats['time_range']['end']}\n")
            f.write(f"  Total records: {self.stats['total_rows']:,}\n")
            f.write(f"  Parsed rows: {self.stats['parsed_rows']:,}\n")
            f.write(f"  Expanded fields: {self.stats['parsed_columns']}\n\n")

            f.write("2. Data Distribution\n")
            f.write("-" * 60 + "\n")
            for addr, count in self.stats['data_distribution'].items():
                f.write(f"  Address {addr}: {count:,} records ({count/self.stats['total_rows']*100:.1f}%)\n")
            f.write("\n")

            f.write("3. Data Quality\n")
            f.write("-" * 60 + "\n")
            f.write(f"  JSON parse errors: {self.stats['json_parse_errors']:,}\n")
            f.write(f"  Missing values filled: {self.stats['cleaning']['missing_values_filled']:,}\n")
            f.write(f"  Anomalies detected: {self.stats['cleaning']['anomalies_detected']:,}\n")
            f.write(f"  Final data rows: {self.stats['cleaning']['final_rows']:,}\n\n")

            f.write("4. Field List\n")
            f.write("-" * 60 + "\n")
            for i, col in enumerate(self.parsed_df.columns, 1):
                dtype = str(self.parsed_df[col].dtype)
                non_null = self.parsed_df[col].notna().sum()
                f.write(f"  {i:3d}. {col:25s} ({dtype:10s}) - {non_null:,} non-null\n")

        print(f"  Statistics report: {stats_file.name}")

        print(f"\nAll data saved to: {output_path.absolute()}")

    def generate_summary_report(self):
        """生成数据摘要报告"""
        print(f"\n{'='*60}")
        print(f"Data Parsing Summary")
        print(f"{'='*60}")

        print(f"Data Loading: {self.stats['total_rows']:,} raw records")
        print(f"JSON Parsing: {self.stats['parsed_rows']:,} success ({self.stats['json_parse_errors']} errors)")
        print(f"Fields: {self.stats['parsed_columns']} columns")
        print(f"Data Cleaning: {self.stats['cleaning']['final_rows']:,} final rows")

        print(f"\nNext Steps:")
        print(f"  1. Check ./processed_data/data_statistics.txt for details")
        print(f"  2. View ./processed_data/data_merged.csv for complete data")
        print(f"  3. Run field_mapping.py to see field definitions")
        print(f"  4. Run data_explorer.py for visualization")


def main():
    """主函数"""
    print("="*60)
    print("IoT Building Data Parser")
    print("="*60)

    # 配置
    CSV_FILE = "ph_ds_iot_data_all_202603181049.csv"
    OUTPUT_DIR = "./processed_data"

    # 创建解析器
    parser = IoTDataParser(CSV_FILE, chunk_size=10000)

    # 执行解析流程
    try:
        # 步骤1: 加载数据
        parser.load_raw_data(use_chunks=True)

        # 步骤2: 探索数据
        parser.explore_raw_data()

        # 步骤3: 解析JSON字段
        parser.parse_json_fields()

        # 步骤4: 数据清洗
        parser.clean_data()

        # 步骤5: 保存数据
        parser.save_data(OUTPUT_DIR)

        # 生成摘要报告
        parser.generate_summary_report()

        print(f"\n{'='*60}")
        print(f"Data Parsing Complete!")
        print(f"{'='*60}\n")

    except Exception as e:
        print(f"\nError during parsing: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
