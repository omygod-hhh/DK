"""
特征工程模块 - 构建预测模型的特征集
创建温度预测、负荷估算等任务的特征
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """特征工程器"""

    def __init__(self, data_file='processed_data/data_merged.csv', output_dir='./features'):
        """
        初始化特征工程器
        :param data_file: 输入数据文件
        :param output_dir: 输出目录
        """
        self.data_file = Path(data_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.df = None
        self.features_df = None

        print("="*60)
        print("Feature Engineering Module")
        print("="*60)
        print(f"Data file: {self.data_file}")
        print(f"Output dir: {self.output_dir}")

    def load_data(self, nrows=None):
        """
        加载数据
        :param nrows: 读取行数(None=全部)
        """
        print(f"\n{'='*60}")
        print(f"Step 1: Loading Data")
        print(f"{'='*60}")

        print(f"Reading {self.data_file}...")

        if nrows:
            self.df = pd.read_csv(self.data_file, nrows=nrows)
            print(f"Loaded {nrows:,} rows")
        else:
            self.df = pd.read_csv(self.data_file)
            print(f"Loaded all {len(self.df):,} rows")

        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])

        print(f"Shape: {self.df.shape}")
        print(f"Time range: {self.df['timestamp'].min()} ~ {self.df['timestamp'].max()}")

    def create_temperature_features(self):
        """创建温度相关特征"""
        print(f"\n{'='*60}")
        print(f"Step 2: Creating Temperature Features")
        print(f"{'='*60}")

        features = pd.DataFrame()
        features['timestamp'] = self.df['timestamp']

        # 1. 原始温度字段
        print(f"  Processing temperature fields...")
        temp_fields = {
            '2028': 'ambient_temp',
            '2029': 'target_temp',
            '2030': 'display_temp',
            '2054': 'water_in_temp',
            '2055': 'water_out_temp',
            '2056': 'ambient_temp_alt'
        }

        for field, name in temp_fields.items():
            if field in self.df.columns:
                features[name] = pd.to_numeric(self.df[field], errors='coerce')
                print(f"    - {field} -> {name}")

        # 2. 水温差 (关键特征)
        if '2054' in self.df.columns and '2055' in self.df.columns:
            features['water_temp_diff'] = self.df['2055'] - self.df['2054']
            print(f"    - water_temp_diff = 2055 - 2054")

        # 3. 温度偏差 (实际 - 目标)
        if '2028' in self.df.columns and '2029' in self.df.columns:
            features['temp_error'] = self.df['2028'] - self.df['2029']
            print(f"    - temp_error = 2028 - 2029")

        # 4. 温度一阶差分 (变化率)
        if '2030' in self.df.columns:
            features['temp_change'] = self.df['2030'].diff()
            print(f"    - temp_change = diff(2030)")

        # 5. 温度二阶差分 (加速度)
        if '2030' in self.df.columns:
            features['temp_acceleration'] = self.df['2030'].diff().diff()
            print(f"    - temp_acceleration = diff(diff(2030))")

        # 6. 水温变化率
        if '2054' in self.df.columns and '2055' in self.df.columns:
            features['water_temp_avg'] = (self.df['2054'] + self.df['2055']) / 2
            features['water_temp_change'] = features['water_temp_avg'].diff()
            print(f"    - water_temp_avg, water_temp_change")

        # 7. 温度异常检测 (基于3σ)
        if '2030' in self.df.columns:
            temp_mean = features['display_temp'].mean()
            temp_std = features['display_temp'].std()
            features['temp_is_outlier'] = (
                (features['display_temp'] < temp_mean - 3*temp_std) |
                (features['display_temp'] > temp_mean + 3*temp_std)
            ).astype(int)
            print(f"    - temp_is_outlier (3σ rule)")

        print(f"\n  Temperature features created: {len(features.columns) - 1}")

        return features

    def create_time_features(self, features):
        """创建时间相关特征"""
        print(f"\n{'='*60}")
        print(f"Step 3: Creating Time Features")
        print(f"{'='*60}")

        print(f"  Extracting time components...")

        # 基础时间特征
        features['hour'] = features['timestamp'].dt.hour
        features['day_of_week'] = features['timestamp'].dt.dayofweek  # 0=Mon, 6=Sun
        features['day_of_month'] = features['timestamp'].dt.day
        features['month'] = features['timestamp'].dt.month
        features['quarter'] = features['timestamp'].dt.quarter

        print(f"    - hour, day_of_week, day_of_month, month, quarter")

        # 是否周末
        features['is_weekend'] = (features['day_of_week'] >= 5).astype(int)
        print(f"    - is_weekend")

        # 是否工作时间 (8:00-18:00)
        features['is_working_hour'] = (
            (features['hour'] >= 8) & (features['hour'] < 18)
        ).astype(int)
        print(f"    - is_working_hour")

        # 是否白天 (6:00-22:00)
        features['is_daytime'] = (
            (features['hour'] >= 6) & (features['hour'] < 22)
        ).astype(int)
        print(f"    - is_daytime")

        # 小时的正弦/余弦编码 (捕捉周期性)
        features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
        features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
        features['day_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
        features['day_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)
        features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
        features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)
        print(f"    - Cyclic encoding: hour_sin/cos, day_sin/cos, month_sin/cos")

        print(f"\n  Time features created: {features.shape[1] - 1}")

        return features

    def create_rolling_features(self, features, windows=[1, 3, 6]):
        """
        创建滑动窗口特征
        :param features: 特征DataFrame
        :param windows: 窗口大小(小时)
        """
        print(f"\n{'='*60}")
        print(f"Step 4: Creating Rolling Window Features")
        print(f"{'='*60}")

        # 估计采样间隔
        time_diffs = features['timestamp'].diff().dropna()
        median_interval_minutes = time_diffs.median().total_seconds() / 60

        print(f"  Median sampling interval: {median_interval_minutes:.1f} minutes")

        for window_hours in windows:
            window_periods = int(window_hours * 60 / median_interval_minutes)
            print(f"\n  Window: {window_hours} hours ({window_periods} samples)")

            # 显示温度的滑动窗口
            if 'display_temp' in features.columns:
                col_name = f'display_temp_rolling_{window_hours}h'
                features[col_name + '_mean'] = features['display_temp'].rolling(
                    window=window_periods, min_periods=1
                ).mean()
                features[col_name + '_std'] = features['display_temp'].rolling(
                    window=window_periods, min_periods=1
                ).std()
                features[col_name + '_min'] = features['display_temp'].rolling(
                    window=window_periods, min_periods=1
                ).min()
                features[col_name + '_max'] = features['display_temp'].rolling(
                    window=window_periods, min_periods=1
                ).max()

                print(f"    - {col_name}_mean/std/min/max")

            # 水温差的滑动窗口
            if 'water_temp_diff' in features.columns:
                col_name = f'water_temp_diff_rolling_{window_hours}h'
                features[col_name + '_mean'] = features['water_temp_diff'].rolling(
                    window=window_periods, min_periods=1
                ).mean()
                features[col_name + '_std'] = features['water_temp_diff'].rolling(
                    window=window_periods, min_periods=1
                ).std()

                print(f"    - {col_name}_mean/std")

            # 阀门位置的滑动窗口
            if 'valve_position' in features.columns:
                col_name = f'valve_rolling_{window_hours}h'
                features[col_name + '_mean'] = features['valve_position'].rolling(
                    window=window_periods, min_periods=1
                ).mean()
                features[col_name + '_std'] = features['valve_position'].rolling(
                    window=window_periods, min_periods=1
                ).std()

                print(f"    - {col_name}_mean/std")

        print(f"\n  Rolling features added")
        return features

    def create_control_features(self):
        """创建控制相关特征"""
        print(f"\n{'='*60}")
        print(f"Step 5: Creating Control Features")
        print(f"{'='*60}")

        # 创建新DataFrame避免修改原始features
        features = self.features_df.copy()

        # 1. 阀门开度
        print(f"  Processing control fields...")
        if '2034' in self.df.columns:
            features['valve_position'] = pd.to_numeric(self.df['2034'], errors='coerce')
            print(f"    - valve_position (from 2034)")

        # 2. 阀门变化率
        if 'valve_position' in features.columns:
            features['valve_change_rate'] = features['valve_position'].diff() / 5  # per 5 min
            print(f"    - valve_change_rate")

        # 3. 压缩机转速
        if '2032' in self.df.columns:
            features['compressor_speed'] = pd.to_numeric(self.df['2032'], errors='coerce')
            print(f"    - compressor_speed (from 2032)")

        # 4. 设备运行状态 (基于转速)
        if 'compressor_speed' in features.columns:
            features['is_compressor_running'] = (features['compressor_speed'] > 0).astype(int)
            print(f"    - is_compressor_running")

        # 5. 阀门饱和度 (0或满量程)
        if 'valve_position' in features.columns:
            valve_min = features['valve_position'].min()
            valve_max = features['valve_position'].max()
            features['valve_saturated'] = (
                (features['valve_position'] <= valve_min + 5) |
                (features['valve_position'] >= valve_max - 5)
            ).astype(int)
            print(f"    - valve_saturated")

        self.features_df = features
        return features

    def create_lag_features(self, features, lags=[1, 2, 3, 6, 12]):
        """
        创建滞后特征
        :param features: 特征DataFrame
        :param lags: 滞后步数列表
        """
        print(f"\n{'='*60}")
        print(f"Step 6: Creating Lag Features")
        print(f"{'='*60}")

        print(f"  Lags: {lags} steps (~{5*np.mean(lags):.0f} minutes)")

        # 对关键字段创建滞后特征
        lag_fields = ['display_temp', 'water_temp_diff', 'valve_position',
                      'ambient_temp', 'target_temp']

        for field in lag_fields:
            if field in features.columns:
                for lag in lags:
                    features[f'{field}_lag{lag}'] = features[field].shift(lag)

                print(f"    - {field}_lag{lags}")

        print(f"\n  Lag features added")

        return features

    def create_target_variables(self):
        """创建目标变量 (用于监督学习)"""
        print(f"\n{'='*60}")
        print(f"Step 7: Creating Target Variables")
        print(f"{'='*60}")

        features = self.features_df.copy()

        # 预测未来1小时、3小时、6小时的显示温度
        # 基于5分钟采样间隔
        print(f"  Creating multi-horizon targets...")

        # 1小时 = 12步 (5分钟*12=60分钟)
        features['target_temp_1h'] = features['display_temp'].shift(-12)
        print(f"    - target_temp_1h (12 steps ahead)")

        # 3小时 = 36步
        features['target_temp_3h'] = features['display_temp'].shift(-36)
        print(f"    - target_temp_3h (36 steps ahead)")

        # 6小时 = 72步
        features['target_temp_6h'] = features['display_temp'].shift(-72)
        print(f"    - target_temp_6h (72 steps ahead)")

        # 未来阀门位置 (可选)
        if 'valve_position' in features.columns:
            features['target_valve_1h'] = features['valve_position'].shift(-12)
            features['target_valve_3h'] = features['valve_position'].shift(-36)
            print(f"    - target_valve_1h, target_valve_3h")

        # 未来温度变化 (分类任务)
        features['temp_will_increase'] = (
            features['display_temp'].shift(-12) > features['display_temp']
        ).astype(int)
        print(f"    - temp_will_increase (binary classification)")

        self.features_df = features
        return features

    def clean_features(self, features):
        """清洗特征"""
        print(f"\n{'='*60}")
        print(f"Step 8: Cleaning Features")
        print(f"{'='*60}")

        initial_rows = len(features)

        # 1. 删除滞后和目标产生的NaN行
        print(f"  Removing NaN from lag/target features...")
        features = features.dropna().reset_index(drop=True)

        removed = initial_rows - len(features)
        print(f"    Removed {removed:,} rows with NaN")

        # 2. 过滤异常温度值
        print(f"  Filtering temperature outliers...")
        before = len(features)

        # 过滤不合理温度 (< -20°C 或 > 60°C 可能是传感器错误)
        if 'display_temp' in features.columns:
            features = features[
                (features['display_temp'] >= -20) &
                (features['display_temp'] <= 60)
            ].copy()

        if 'ambient_temp' in features.columns:
            features = features[
                (features['ambient_temp'] >= -20) &
                (features['ambient_temp'] <= 60)
            ].copy()

        removed = before - len(features)
        print(f"    Removed {removed:,} outlier rows")

        # 3. 删除常量特征
        print(f"  Removing constant features...")
        constant_cols = []
        for col in features.columns:
            if col not in ['timestamp'] and features[col].nunique() <= 1:
                constant_cols.append(col)

        if constant_cols:
            features = features.drop(columns=constant_cols)
            print(f"    Removed {len(constant_cols)} constant features")

        print(f"\n  Final shape: {features.shape}")

        return features

    def save_features(self, features):
        """保存特征"""
        print(f"\n{'='*60}")
        print(f"Step 9: Saving Features")
        print(f"{'='*60}")

        # 1. 完整特征集
        full_output = self.output_dir / 'features_full.csv'
        features.to_csv(full_output, index=False, encoding='utf-8-sig')
        print(f"  Full features: {full_output.name} ({len(features):,} rows, {len(features.columns)} cols)")

        # 2. 按任务分开保存
        print(f"\n  Saving task-specific datasets...")

        # 温度预测数据集
        temp_cols = [col for col in features.columns if 'target_temp' in col or 'temp' in col or col == 'timestamp']
        temp_cols += [c for c in ['hour', 'day_of_week', 'month', 'is_weekend', 'is_daytime',
                                   'hour_sin', 'hour_cos', 'day_sin', 'day_cos'] if c in features.columns]

        if len(temp_cols) > 1:
            temp_features = features[temp_cols].copy()
            temp_output = self.output_dir / 'features_temperature_prediction.csv'
            temp_features.to_csv(temp_output, index=False, encoding='utf-8-sig')
            print(f"    Temperature prediction: {temp_output.name} ({len(temp_features):,} rows, {len(temp_features.columns)} cols)")

        # 阀门控制数据集
        valve_cols = [col for col in features.columns if 'valve' in col or 'temp' in col or col == 'timestamp']
        if 'valve_position' in features.columns and len(valve_cols) > 1:
            valve_features = features[valve_cols].copy()
            valve_output = self.output_dir / 'features_valve_control.csv'
            valve_features.to_csv(valve_output, index=False, encoding='utf-8-sig')
            print(f"    Valve control: {valve_output.name} ({len(valve_features):,} rows, {len(valve_features.columns)} cols)")

        # 负荷估算数据集
        load_cols = [col for col in features.columns if
                     'water_temp' in col or 'valve' in col or
                     'compressor' in col or col == 'timestamp']
        if len(load_cols) > 1:
            load_features = features[load_cols].copy()
            load_output = self.output_dir / 'features_load_estimation.csv'
            load_features.to_csv(load_output, index=False, encoding='utf-8-sig')
            print(f"    Load estimation: {load_output.name} ({len(load_features):,} rows, {len(load_features.columns)} cols)")

        # 特征字典
        feature_dict = {
            'total_features': len(features.columns),
            'total_samples': len(features),
            'features': {
                col: {
                    'dtype': str(features[col].dtype),
                    'non_null': int(features[col].notna().sum()),
                    'mean': float(features[col].mean()) if features[col].dtype in ['float64', 'int64'] else None,
                    'std': float(features[col].std()) if features[col].dtype in ['float64', 'int64'] else None
                }
                for col in features.columns
            }
        }

        dict_output = self.output_dir / 'features_dictionary.json'
        import json
        with open(dict_output, 'w', encoding='utf-8') as f:
            json.dump(feature_dict, f, ensure_ascii=False, indent=2, default=str)
        print(f"    Feature dictionary: {dict_output.name}")

        print(f"\n  All features saved to: {self.output_dir.absolute()}")

    def generate_feature_report(self, features):
        """生成特征报告"""
        print(f"\n{'='*60}")
        print(f"Step 10: Generating Feature Report")
        print(f"{'='*60}")

        report_file = self.output_dir / 'feature_engineering_report.txt'

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("Feature Engineering Report\n")
            f.write("="*60 + "\n\n")

            f.write("1. Overview\n")
            f.write("-"*60 + "\n")
            f.write(f"  Total samples: {len(features):,}\n")
            f.write(f"  Total features: {len(features.columns)}\n")
            f.write(f"  Target variables: target_temp_1h, target_temp_3h, target_temp_6h\n\n")

            f.write("2. Feature Categories\n")
            f.write("-"*60 + "\n\n")

            # 分类统计
            categories = {
                'Temperature': [c for c in features.columns if 'temp' in c.lower()],
                'Valve/Control': [c for c in features.columns if 'valve' in c.lower() or 'compressor' in c.lower()],
                'Time': [c for c in features.columns if c in ['hour', 'day_of_week', 'month', 'is_weekend', 'is_daytime']],
                'Cyclic': [c for c in features.columns if '_sin' in c or '_cos' in c],
                'Rolling Stats': [c for c in features.columns if 'rolling' in c],
                'Lag Features': [c for c in features.columns if '_lag' in c],
                'Targets': [c for c in features.columns if 'target' in c]
            }

            for cat, cols in categories.items():
                if cols:
                    f.write(f"  {cat}: {len(cols)} features\n")
                    for col in cols[:5]:
                        f.write(f"    - {col}\n")
                    if len(cols) > 5:
                        f.write(f"    ... and {len(cols)-5} more\n")
                    f.write("\n")

            f.write("3. Feature Statistics\n")
            f.write("-"*60 + "\n\n")
            f.write(f"{'Feature':<40} {'Type':<12} {'Non-Null':>10} {'Mean':>12} {'Std':>12}\n")
            f.write("-"*60 + "\n")

            for col in features.columns[:50]:  # Show first 50
                dtype = str(features[col].dtype)
                non_null = features[col].notna().sum()
                mean = features[col].mean() if features[col].dtype in ['float64', 'int64'] else None
                std = features[col].std() if features[col].dtype in ['float64', 'int64'] else None

                mean_str = f"{mean:>12.3f}" if mean is not None else f"{'N/A':>12}"
                std_str = f"{std:>12.3f}" if std is not None else f"{'N/A':>12}"

                f.write(f"{col:<40} {dtype:<12} {non_null:>10,} {mean_str} {std_str}\n")

            f.write("\n4. Usage Examples\n")
            f.write("-"*60 + "\n\n")

            f.write("Temperature Prediction (1h ahead):\n")
            f.write("  Features: X = df[['display_temp', 'temp_error', 'water_temp_diff',\n")
            f.write("                  'hour_sin', 'hour_cos', 'day_sin', 'day_cos',\n")
            f.write("                  'display_temp_rolling_1h_mean', ...]]\n")
            f.write("  Target: y = df['target_temp_1h']\n")
            f.write("  Model: LightGBM Regressor\n\n")

            f.write("Valve Control Prediction:\n")
            f.write("  Features: X = df[['temp_error', 'display_temp', 'water_temp_diff',\n")
            f.write("                  'hour', 'is_working_hour', ...]]\n")
            f.write("  Target: y = df['target_valve_1h']\n")
            f.write("  Model: LightGBM Regressor\n\n")

            f.write("Load Estimation:\n")
            f.write("  Features: X = df[['valve_position', 'water_temp_diff',\n")
            f.write("                  'compressor_speed', ...]]\n")
            f.write("  Target: (relative load - need calibration)\n")
            f.write("  Model: Linear Regression or LightGBM\n")

        print(f"  Report saved: {report_file.name}")

    def run(self, sample_size=None):
        """
        运行完整特征工程流程
        :param sample_size: 采样大小(None=全部)
        """
        print("="*60)
        print("Feature Engineering Pipeline")
        print("="*60)

        try:
            # 1. 加载数据
            self.load_data(nrows=sample_size)

            # 2. 创建温度特征
            features = self.create_temperature_features()

            # 3. 创建时间特征
            features = self.create_time_features(features)

            # 保存中间结果
            self.features_df = features

            # 4. 创建控制特征
            features = self.create_control_features()

            # 5. 创建滑动窗口特征
            features = self.create_rolling_features(features, windows=[1, 3, 6])

            # 6. 创建滞后特征
            features = self.create_lag_features(features, lags=[1, 2, 3, 6, 12])

            # 7. 创建目标变量
            features = self.create_target_variables()

            # 8. 清洗特征
            features = self.clean_features(features)

            # 9. 保存特征
            self.save_features(features)

            # 10. 生成报告
            self.generate_feature_report(features)

            print(f"\n{'='*60}")
            print(f"Feature Engineering Complete!")
            print(f"{'='*60}")
            print(f"\nSummary:")
            print(f"  Total samples: {len(features):,}")
            print(f"  Total features: {len(features.columns)}")
            print(f"  Output directory: {self.output_dir.absolute()}")
            print(f"\nNext steps:")
            print(f"  1. Review features in {self.output_dir}")
            print(f"  2. Train baseline model with features_temperature_prediction.csv")
            print(f"  3. Validate using time-series cross-validation")

        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()


def main():
    """主函数"""
    print("\n" + "="*60)
    print("Feature Engineering for Building Load Prediction")
    print("="*60)

    # 配置
    DATA_FILE = "processed_data/data_merged.csv"
    OUTPUT_DIR = "./features"
    SAMPLE_SIZE = None  # None=全部数据,或设置如 100000

    # 创建特征工程器
    engineer = FeatureEngineer(DATA_FILE, OUTPUT_DIR)

    # 运行特征工程
    engineer.run(sample_size=SAMPLE_SIZE)


if __name__ == "__main__":
    main()
