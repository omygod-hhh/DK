"""
字段映射工具 - 将数字字段ID映射到参数名称
使用Excel协议文件作为参考
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime


class FieldMapper:
    """IoT数据字段映射器"""

    def __init__(self, protocol_file='NET-DK-M1101-B_DK.xls', test_data_file='processed_data/data_test_10000.csv'):
        """
        初始化字段映射器
        :param protocol_file: 协议定义文件
        :param test_data_file: 测试数据文件
        """
        self.protocol_file = Path(protocol_file)
        self.test_data_file = Path(test_data_file)
        self.protocol_df = None
        self.field_map = {}
        self.reverse_map = {}

        print("Field Mapper Initialized")
        print(f"Protocol file: {self.protocol_file}")
        print(f"Test data file: {self.test_data_file}")

    def load_protocol(self):
        """加载协议定义"""
        print(f"\n{'='*60}")
        print(f"Loading Protocol Definitions")
        print(f"{'='*60}")

        # 读取Excel协议文件
        xl = pd.ExcelFile(self.protocol_file)
        self.protocol_df = xl.parse('protocal')

        print(f"Loaded protocol: {len(self.protocol_df)} parameter definitions")

        # 创建字段映射 (点位 -> 参数名称)
        for idx, row in self.protocol_df.iterrows():
            point = str(row.get('点位', ''))
            param_name = str(row.get('参数名称', ''))
            param_en = str(row.get('参数英文名称', ''))
            desc = str(row.get('描述', ''))
            desc_en = str(row.get('英文描述', ''))
            unit = str(row.get('显示单位', ''))

            if point and point != 'nan':
                self.field_map[point] = {
                    'name_cn': param_name if param_name != 'nan' else '',
                    'name_en': param_en if param_en != 'nan' else '',
                    'desc_cn': desc if desc != 'nan' else '',
                    'desc_en': desc_en if desc_en != 'nan' else '',
                    'unit': unit if unit != 'nan' else ''
                }

        print(f"Created field map: {len(self.field_map)} entries")

    def load_test_data(self):
        """加载测试数据"""
        print(f"\n{'='*60}")
        print(f"Loading Test Data")
        print(f"{'='*60}")

        df = pd.read_csv(self.test_data_file, nrows=100)
        print(f"Loaded test data: {len(df)} rows")

        return df

    def analyze_fields(self, df):
        """分析字段并映射到协议"""
        print(f"\n{'='*60}")
        print(f"Analyzing Fields and Mapping to Protocol")
        print(f"{'='*60}")

        # 识别数据字段(排除元数据)
        meta_cols = {'log_id', 'device_code', 'timestamp', 'start_address'}
        data_fields = [col for col in df.columns if col not in meta_cols]

        print(f"\nTotal data fields: {len(data_fields)}")

        # 分析每个字段
        field_stats = []
        for field in data_fields:
            # 尝试映射到协议
            mapping = self.field_map.get(field, None)

            # 计算统计信息
            values = pd.to_numeric(df[field], errors='coerce')
            non_null = values.notna().sum()
            valid_pct = non_null / len(df) * 100

            field_info = {
                'field_id': field,
                'mapped': mapping is not None,
                'name_cn': mapping['name_cn'] if mapping else 'Unknown',
                'name_en': mapping['name_en'] if mapping else 'Unknown',
                'unit': mapping['unit'] if mapping else '',
                'non_null': non_null,
                'valid_pct': valid_pct,
                'min': values.min(),
                'max': values.max(),
                'mean': values.mean(),
                'std': values.std()
            }
            field_stats.append(field_info)

        # 创建统计DataFrame
        stats_df = pd.DataFrame(field_stats)
        stats_df = stats_df.sort_values('valid_pct', ascending=False)

        # 显示有映射的字段
        mapped = stats_df[stats_df['mapped'] == True]
        print(f"\nMapped fields: {len(mapped)}")

        print(f"\nTop 30 High-Quality Fields (by data availability):")
        print(f"{'Field':<10} {'Name CN':<20} {'Name EN':<25} {'Unit':<8} {'Valid%':>8}")
        print("-"*75)

        for idx, row in stats_df.head(30).iterrows():
            print(f"{row['field_id']:<10} {row['name_cn'][:19]:<20} {row['name_en'][:24]:<25} "
                  f"{row['unit'][:7]:<8} {row['valid_pct']:>7.1f}%")

        return stats_df

    def identify_key_fields(self, stats_df):
        """识别关键字段"""
        print(f"\n{'='*60}")
        print(f"Identifying Key Fields")
        print(f"{'='*60}")

        # 基于关键字识别关键字段
        keywords = {
            'temperature': ['temp', '温度', 'Temp'],
            'valve': ['valve', '阀门', 'Valve'],
            'load': ['load', '负载', 'Load'],
            'power': ['power', '功率', 'Power'],
            'pressure': ['pressure', '压力', 'Pressure'],
            'flow': ['flow', '流量', 'Flow'],
            'setpoint': ['set', '设定', 'Set'],
            'status': ['status', '状态', 'Status']
        }

        print(f"\nIdentifying fields by keyword:")

        key_fields = {}
        for category, kws in keywords.items():
            matched = []
            for idx, row in stats_df.iterrows():
                field_name = str(row['name_en']) + str(row['name_cn'])
                if any(kw in field_name for kw in kws):
                    matched.append({
                        'field_id': row['field_id'],
                        'name_cn': row['name_cn'],
                        'name_en': row['name_en'],
                        'valid_pct': row['valid_pct'],
                        'mean': row['mean']
                    })

            if matched:
                key_fields[category] = matched
                print(f"\n  {category.upper()}: {len(matched)} fields")
                for m in matched[:3]:  # Show top 3
                    print(f"    - {m['field_id']}: {m['name_cn']} / {m['name_en']} "
                          f"({m['valid_pct']:.1f}% valid)")

        return key_fields

    def generate_field_report(self, stats_df, output_file='field_mapping_report.txt'):
        """生成字段映射报告"""
        print(f"\n{'='*60}")
        print(f"Generating Field Mapping Report")
        print(f"{'='*60}")

        output_path = Path(output_file)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("Field Mapping Report\n")
            f.write("="*60 + "\n\n")

            # 统计概览
            f.write("1. Overview\n")
            f.write("-"*60 + "\n")
            f.write(f"Total fields: {len(stats_df)}\n")
            f.write(f"Mapped fields: {stats_df['mapped'].sum()}\n")
            f.write(f"Unmapped fields: {(~stats_df['mapped']).sum()}\n\n")

            # 字段详细列表
            f.write("2. Field Details (sorted by data availability)\n")
            f.write("-"*60 + "\n")
            f.write(f"{'Field ID':<10} {'Name CN':<20} {'Name EN':<25} {'Unit':<8} "
                    f"{'Valid%':>8} {'Min':>10} {'Max':>10} {'Mean':>10}\n")
            f.write("-"*60 + "\n")

            for idx, row in stats_df.iterrows():
                f.write(f"{row['field_id']:<10} {row['name_cn'][:19]:<20} "
                        f"{row['name_en'][:24]:<25} {row['unit'][:7]:<8} "
                        f"{row['valid_pct']:>7.1f}% {row['min']:>10.2f} "
                        f"{row['max']:>10.2f} {row['mean']:>10.2f}\n")

            # 关键字段总结
            f.write("\n\n3. Key Fields Summary\n")
            f.write("-"*60 + "\n")

            key_categories = {
                'Temperature': ['temp', '温度'],
                'Valve': ['valve', '阀门'],
                'Power/Load': ['power', 'load', '功率', '负载'],
                'Pressure': ['pressure', '压力'],
                'Flow': ['flow', '流量'],
                'Setpoint': ['set', '设定'],
                'Status': ['status', '状态']
            }

            for category, keywords in key_categories.items():
                f.write(f"\n{category} Fields:\n")
                matched = stats_df[
                    stats_df['name_en'].str.contains('|'.join(keywords), case=False, na=False) |
                    stats_df['name_cn'].str.contains('|'.join(keywords), na=False)
                ]

                if len(matched) > 0:
                    for idx, row in matched.head(10).iterrows():
                        f.write(f"  - {row['field_id']}: {row['name_cn']} / "
                                f"{row['name_en']} ({row['valid_pct']:.1f}% valid)\n")
                else:
                    f.write(f"  No fields found\n")

        print(f"Report saved to: {output_path}")

    def generate_field_dict(self, stats_df, output_file='field_dictionary.json'):
        """生成字段字典JSON"""
        print(f"\n{'='*60}")
        print(f"Generating Field Dictionary")
        print(f"{'='*60}")

        output_path = Path(output_file)

        field_dict = {}
        for idx, row in stats_df.iterrows():
            field_dict[row['field_id']] = {
                'name_cn': row['name_cn'],
                'name_en': row['name_en'],
                'unit': row['unit'],
                'valid_pct': float(row['valid_pct']),
                'min': float(row['min']) if not pd.isna(row['min']) else None,
                'max': float(row['max']) if not pd.isna(row['max']) else None,
                'mean': float(row['mean']) if not pd.isna(row['mean']) else None,
                'mapped': bool(row['mapped'])
            }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(field_dict, f, ensure_ascii=False, indent=2)

        print(f"Dictionary saved to: {output_path}")

    def run(self):
        """执行完整的字段映射流程"""
        print("="*60)
        print("Field Mapping Tool")
        print("="*60)

        try:
            # 1. 加载协议
            self.load_protocol()

            # 2. 加载测试数据
            df = self.load_test_data()

            # 3. 分析字段
            stats_df = self.analyze_fields(df)

            # 4. 识别关键字段
            key_fields = self.identify_key_fields(stats_df)

            # 5. 生成报告
            self.generate_field_report(stats_df)

            # 6. 生成字典
            self.generate_field_dict(stats_df)

            print(f"\n{'='*60}")
            print(f"Field Mapping Complete!")
            print(f"{'='*60}")
            print(f"\nOutput files:")
            print(f"  - field_mapping_report.txt (detailed report)")
            print(f"  - field_dictionary.json (machine-readable)")

        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    mapper = FieldMapper()
    mapper.run()
