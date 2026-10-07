"""
数据探索可视化脚本 - 分析关键字段的分布和趋势
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = ['DejaVu Sans']
import seaborn as sns
from pathlib import Path
import numpy as np
import warnings
warnings.filterwarnings('ignore')


class DataExplorer:
    """数据探索工具"""

    def __init__(self, data_file='processed_data/data_test_10000.csv'):
        """
        初始化数据探索器
        :param data_file: 数据文件路径
        """
        self.data_file = Path(data_file)
        self.df = None
        self.output_dir = Path('./reports/figures')
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Data Explorer Initialized")
        print(f"Data file: {self.data_file}")

    def load_data(self):
        """加载数据"""
        print(f"\n{'='*60}")
        print(f"Loading Data")
        print(f"{'='*60}")

        self.df = pd.read_csv(self.data_file)
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])

        print(f"Loaded: {len(self.df):,} rows x {len(self.df.columns)} columns")
        print(f"Time range: {self.df['timestamp'].min()} ~ {self.df['timestamp'].max()}")

    def identify_key_fields(self):
        """识别关键字段"""
        # 根据协议文件识别的关键字段
        self.key_fields = {
            # Temperature fields
            '2028': {'name': 'Ambient Temp', 'unit': '°C', 'category': 'Temperature'},
            '2029': {'name': 'Target Temp', 'unit': '°C', 'category': 'Temperature'},
            '2030': {'name': 'Display Temp', 'unit': '°C', 'category': 'Temperature'},
            '2054': {'name': 'Total In Water Temp', 'unit': '°C', 'category': 'Temperature'},
            '2055': {'name': 'Total Out Water Temp', 'unit': '°C', 'category': 'Temperature'},
            '2056': {'name': 'Ambient Temp (alt)', 'unit': '°C', 'category': 'Temperature'},

            # Valve fields
            '2033': {'name': 'Valve 1 Position', 'unit': 'N', 'category': 'Valve'},
            '2034': {'name': 'Valve 2 Position', 'unit': 'N', 'category': 'Valve'},

            # Power/Compressor fields
            '2031': {'name': 'Bus Voltage', 'unit': 'V', 'category': 'Power'},
            '2032': {'name': 'Compressor 1 Speed', 'unit': 'RPM', 'category': 'Power'},
        }

        print(f"\n{'='*60}")
        print(f"Key Fields Identified")
        print(f"{'='*60}")

        available = []
        for field_id, info in self.key_fields.items():
            if field_id in self.df.columns:
                valid_pct = self.df[field_id].notna().sum() / len(self.df) * 100
                info['valid_pct'] = valid_pct
                available.append((field_id, info))
                print(f"  {field_id}: {info['name']:30s} [{info['unit']}] - {valid_pct:.1f}% valid")
            else:
                print(f"  {field_id}: {info['name']:30s} - NOT AVAILABLE")

        print(f"\nAvailable key fields: {len(available)}/{len(self.key_fields)}")

    def plot_time_series(self):
        """绘制时间序列图"""
        print(f"\n{'='*60}")
        print(f"Creating Time Series Plots")
        print(f"{'='*60}")

        # 只绘制前1000个点以便查看
        df_sample = self.df.head(1000).copy()

        # Temperature fields
        temp_fields = ['2028', '2029', '2030', '2054', '2055', '2056']

        plt.figure(figsize=(15, 10))

        for i, field_id in enumerate(temp_fields, 1):
            if field_id in df_sample.columns:
                plt.subplot(2, 3, i)
                values = pd.to_numeric(df_sample[field_id], errors='coerce')
                valid = values.dropna()

                if len(valid) > 0:
                    info = self.key_fields.get(field_id, {'name': field_id, 'unit': ''})
                    plt.plot(valid.index, valid.values, linewidth=2)
                    plt.title(f"{info['name']} ({field_id})", fontsize=10)
                    plt.xlabel('Sample Index')
                    plt.ylabel(f"Value [{info['unit']}]")
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()

        plt.savefig(self.output_dir / '01_temperature_time_series.png', dpi=150, bbox_inches='tight')
        print(f"  Saved: 01_temperature_time_series.png")
        plt.close()

        # Valve and power fields
        other_fields = ['2033', '2034', '2031', '2032']

        plt.figure(figsize=(15, 8))

        for i, field_id in enumerate(other_fields, 1):
            if field_id in df_sample.columns:
                plt.subplot(2, 2, i)
                values = pd.to_numeric(df_sample[field_id], errors='coerce')
                valid = values.dropna()

                if len(valid) > 0:
                    info = self.key_fields.get(field_id, {'name': field_id, 'unit': ''})
                    plt.plot(valid.index, valid.values, linewidth=2, color='orange')
                    plt.title(f"{info['name']} ({field_id})", fontsize=10)
                    plt.xlabel('Sample Index')
                    plt.ylabel(f"Value [{info['unit']}]")
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()

        plt.savefig(self.output_dir / '02_valve_power_time_series.png', dpi=150, bbox_inches='tight')
        print(f"  Saved: 02_valve_power_time_series.png")
        plt.close()

    def plot_distributions(self):
        """绘制分布图"""
        print(f"\n{'='*60}")
        print(f"Creating Distribution Plots")
        print(f"{'='*60}")

        all_fields = list(self.key_fields.keys())

        plt.figure(figsize=(15, 12))

        for i, field_id in enumerate(all_fields, 1):
            if field_id in self.df.columns and i <= 12:
                plt.subplot(3, 4, i)
                values = pd.to_numeric(self.df[field_id], errors='coerce').dropna()

                if len(values) > 0:
                    info = self.key_fields.get(field_id, {'name': field_id, 'unit': ''})

                    # 绘制直方图和核密度估计
                    plt.hist(values, bins=50, alpha=0.7, density=True, color='steelblue')
                    # 不使用KDE，因为需要scipy
                    # values.plot(kind='kde', color='red', linewidth=2)

                    plt.title(f"{info['name']}\n({field_id})", fontsize=9)
                    plt.xlabel(f"Value [{info['unit']}]")
                    plt.ylabel('Density')
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()

        plt.savefig(self.output_dir / '03_distributions.png', dpi=150, bbox_inches='tight')
        print(f"  Saved: 03_distributions.png")
        plt.close()

    def plot_correlation(self):
        """绘制相关性矩阵"""
        print(f"\n{'='*60}")
        print(f"Creating Correlation Matrix")
        print(f"{'='*60}")

        # 选择数值型关键字段
        numeric_fields = [f for f in self.key_fields.keys() if f in self.df.columns]

        if len(numeric_fields) >= 2:
            corr_data = pd.DataFrame()

            for field_id in numeric_fields:
                values = pd.to_numeric(self.df[field_id], errors='coerce')
                info = self.key_fields.get(field_id, {'name': field_id})
                corr_data[info['name']] = values

            # 计算相关性矩阵
            corr_matrix = corr_data.corr()

            plt.figure(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                       center=0, square=True, linewidths=1,
                       xticklabels=corr_matrix.columns,
                       yticklabels=corr_matrix.columns)
            plt.title('Correlation Matrix of Key Fields', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(self.output_dir / '04_correlation_matrix.png', dpi=150, bbox_inches='tight')
            print(f"  Saved: 04_correlation_matrix.png")
            plt.close()

            # 保存相关性矩阵到CSV
            corr_matrix.to_csv(self.output_dir / 'correlation_matrix.csv')
            print(f"  Saved: correlation_matrix.csv")

    def plot_boxplots(self):
        """绘制箱线图"""
        print(f"\n{'='*60}")
        print(f"Creating Box Plots")
        print(f"{'='*60}")

        all_fields = list(self.key_fields.keys())
        data_list = []
        labels = []

        for field_id in all_fields:
            if field_id in self.df.columns:
                values = pd.to_numeric(self.df[field_id], errors='coerce').dropna()
                if len(values) > 0:
                    info = self.key_fields.get(field_id, {'name': field_id})
                    data_list.append(values)
                    labels.append(f"{info['name']}\n({field_id})")

        if data_list:
            plt.figure(figsize=(15, 6))
            bp = plt.boxplot(data_list, patch_artist=True)
            plt.xticks(range(1, len(labels)+1), labels, rotation=45, ha='right')

            # 设置颜色
            colors = plt.cm.Set3(np.linspace(0, 1, len(data_list)))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)

            plt.title('Box Plot of Key Fields', fontsize=14, fontweight='bold')
            plt.ylabel('Value')
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig(self.output_dir / '05_boxplots.png', dpi=150, bbox_inches='tight')
            print(f"  Saved: 05_boxplots.png")
            plt.close()

    def generate_statistics_report(self):
        """生成统计报告"""
        print(f"\n{'='*60}")
        print(f"Generating Statistics Report")
        print(f"{'='*60}")

        report_file = self.output_dir / '../data_exploration_report.txt'
        report_file.parent.mkdir(parents=True, exist_ok=True)

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("Data Exploration Report\n")
            f.write("="*60 + "\n\n")

            f.write(f"Data File: {self.data_file}\n")
            f.write(f"Total Records: {len(self.df):,}\n")
            f.write(f"Total Fields: {len(self.df.columns)}\n\n")

            f.write("Key Fields Statistics:\n")
            f.write("-"*60 + "\n\n")

            for field_id, info in self.key_fields.items():
                if field_id in self.df.columns:
                    values = pd.to_numeric(self.df[field_id], errors='coerce').dropna()

                    if len(values) > 0:
                        f.write(f"{info['name']} ({field_id}):\n")
                        f.write(f"  Unit: {info['unit']}\n")
                        f.write(f"  Valid: {len(values):,} ({len(values)/len(self.df)*100:.1f}%)\n")
                        f.write(f"  Min: {values.min():.2f}\n")
                        f.write(f"  Max: {values.max():.2f}\n")
                        f.write(f"  Mean: {values.mean():.2f}\n")
                        f.write(f"  Std: {values.std():.2f}\n")
                        f.write(f"  Median: {values.median():.2f}\n\n")

        print(f"  Saved: {report_file}")

    def run(self):
        """运行数据探索"""
        print("="*60)
        print("Data Explorer")
        print("="*60)

        try:
            # 1. 加载数据
            self.load_data()

            # 2. 识别关键字段
            self.identify_key_fields()

            # 3. 绘制时间序列
            self.plot_time_series()

            # 4. 绘制分布图
            self.plot_distributions()

            # 5. 绘制相关性矩阵
            self.plot_correlation()

            # 6. 绘制箱线图
            self.plot_boxplots()

            # 7. 生成统计报告
            self.generate_statistics_report()

            print(f"\n{'='*60}")
            print(f"Data Exploration Complete!")
            print(f"{'='*60}")
            print(f"\nOutputs saved to: {self.output_dir.absolute()}")
            print(f"\nGenerated files:")
            for f in sorted(self.output_dir.glob('*')):
                print(f"  - {f.name}")

        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    explorer = DataExplorer()
    explorer.run()
