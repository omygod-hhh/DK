"""
数据翻译脚本 - 将address_value中的数字字段ID翻译为中文参数名
保持原有JSON格式，仅替换键名
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class DataTranslator:
    """数据翻译器 - 将数字字段ID翻译为可读的参数名称"""

    def __init__(self, csv_file='ph_ds_iot_data_all_202603181049.csv',
                 protocol_file='NET-DK-M1101-B_DK.xls',
                 output_file='translated_data.csv',
                 chunk_size=10000):
        """
        初始化翻译器
        :param csv_file: 输入CSV文件
        :param protocol_file: 协议定义文件
        :param output_file: 输出CSV文件
        :param chunk_size: 分块大小
        """
        self.csv_file = Path(csv_file)
        self.protocol_file = Path(protocol_file)
        self.output_file = Path(output_file)
        self.chunk_size = chunk_size

        # 加载字段映射
        self.field_map = self._load_field_map()

        print("="*60)
        print("Data Translator")
        print("="*60)
        print(f"Input: {self.csv_file}")
        print(f"Protocol: {self.protocol_file}")
        print(f"Output: {self.output_file}")
        print(f"Chunk size: {self.chunk_size:,}")
        print(f"Loaded {len(self.field_map)} field mappings")

    def _load_field_map(self):
        """从Excel协议文件加载字段映射"""
        print(f"\nLoading protocol from {self.protocol_file}...")

        try:
            xl = pd.ExcelFile(self.protocol_file)
            df = xl.parse('protocal')

            field_map = {}
            for idx, row in df.iterrows():
                point = str(row.get('点位', ''))
                param_name = str(row.get('参数名称', ''))
                param_en = str(row.get('参数英文名称', ''))
                unit = str(row.get('显示单位', ''))

                if point and point != 'nan' and point not in field_map:
                    # 构建友好的字段名: 中文名(英文名) [单位]
                    name_parts = []
                    if param_name and param_name != 'nan':
                        name_parts.append(param_name)
                    if param_en and param_en != 'nan':
                        name_parts.append(f"({param_en})")

                    field_name = ' '.join(name_parts)
                    if unit and unit != 'nan':
                        field_name += f" [{unit}]"

                    field_map[point] = field_name

            print(f"Loaded {len(field_map)} field definitions")
            return field_map

        except Exception as e:
            print(f"Error loading protocol: {e}")
            return {}

    def _translate_json_keys(self, json_str):
        """
        翻译JSON中的数字键为中文参数名
        :param json_str: JSON字符串
        :return: 翻译后的JSON字符串
        """
        try:
            data_dict = json.loads(json_str)

            # 翻译键
            translated = {}
            for key, value in data_dict.items():
                key_str = str(key)
                if key_str in self.field_map:
                    new_key = self.field_map[key_str]
                else:
                    # 未映射的字段保持原样
                    new_key = key_str
                translated[new_key] = value

            # 转回JSON字符串
            return json.dumps(translated, ensure_ascii=False)

        except Exception as e:
            # 解析失败返回原字符串
            return json_str

    def translate_data(self, sample_only=False, max_rows=None):
        """
        翻译数据
        :param sample_only: 只翻译前10000行
        :param max_rows: 最多翻译行数
        """
        print(f"\n{'='*60}")
        print(f"Translating Data")
        print(f"{'='*60}")

        # 确定读取行数
        nrows = None
        if sample_only:
            nrows = 10000
        elif max_rows:
            nrows = max_rows

        # 读取数据
        print(f"\nReading data...")
        if nrows:
            df = pd.read_csv(self.csv_file, nrows=nrows)
            print(f"Loaded {len(df):,} rows (sample)")
        else:
            # 分块读取
            chunks = []
            total_rows = 0
            for chunk in pd.read_csv(self.csv_file, chunksize=self.chunk_size):
                chunks.append(chunk)
                total_rows += len(chunk)
                if total_rows % (self.chunk_size * 10) == 0:
                    print(f"  Read {total_rows:,} rows...")
            df = pd.concat(chunks, ignore_index=True)
            print(f"Loaded {len(df):,} rows")

        print(f"Columns: {list(df.columns)}")

        # 翻译address_value列
        print(f"\nTranslating 'address_value' column...")
        if 'address_value' not in df.columns:
            print("ERROR: 'address_value' column not found!")
            return None

        translated_count = 0
        unmapped_keys = set()

        for idx, row in df.iterrows():
            original = row['address_value']

            try:
                # 解析原始JSON
                orig_dict = json.loads(original)

                # 翻译键
                translated_dict = {}
                for key, value in orig_dict.items():
                    key_str = str(key)
                    if key_str in self.field_map:
                        new_key = self.field_map[key_str]
                    else:
                        # 记录未映射的键
                        unmapped_keys.add(key_str)
                        new_key = key_str
                    translated_dict[new_key] = value

                # 转回JSON字符串
                translated = json.dumps(translated_dict, ensure_ascii=False)

            except Exception as e:
                # JSON解析失败，保持原样
                translated = original
                if idx < 5:
                    print(f"  Warning: JSON parse error at row {idx}: {str(e)[:100]}")

            df.at[idx, 'address_value'] = translated
            translated_count += 1

            if (idx + 1) % 1000 == 0:
                print(f"  Translated {idx + 1:,}/{len(df):,} rows... (unmapped: {len(unmapped_keys)})")

        print(f"\nTranslation Complete:")
        print(f"  Total rows: {len(df):,}")
        print(f"  Unmapped field IDs: {len(unmapped_keys)}")
        if unmapped_keys:
            print(f"  Sample unmapped IDs: {list(unmapped_keys)[:10]}")

        # 保存结果
        print(f"\nSaving to {self.output_file}...")
        df.to_csv(self.output_file, index=False, encoding='utf-8-sig')
        print(f"Saved! File size: {self.output_file.stat().st_size / 1024**2:.2f} MB")

        # 生成翻译报告
        self._generate_report(df, unmapped_keys)

        return df

    def _generate_report(self, df, unmapped_keys):
        """生成翻译报告"""
        report_file = self.output_file.parent / 'translation_report.txt'

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("Data Translation Report\n")
            f.write("="*60 + "\n\n")

            f.write(f"Input file: {self.csv_file}\n")
            f.write(f"Output file: {self.output_file}\n")
            f.write(f"Protocol file: {self.protocol_file}\n")
            f.write(f"Translation time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write(f"Statistics:\n")
            f.write(f"  Total rows translated: {len(df):,}\n")
            f.write(f"  Field mappings loaded: {len(self.field_map)}\n")
            f.write(f"  Unmapped field IDs: {len(unmapped_keys)}\n\n")

            if unmapped_keys:
                f.write(f"Unmapped Field IDs (first 50):\n")
                for key in sorted(unmapped_keys)[:50]:
                    f.write(f"  - {key}\n")
                if len(unmapped_keys) > 50:
                    f.write(f"  ... and {len(unmapped_keys) - 50} more\n")

        print(f"Report saved to: {report_file}")

    def run(self, sample_only=False, max_rows=None):
        """执行翻译流程"""
        print("="*60)
        print("Data Translation Started")
        print("="*60)

        try:
            self.translate_data(sample_only=sample_only, max_rows=max_rows)

            print(f"\n{'='*60}")
            print(f"Translation Complete!")
            print(f"{'='*60}")
            print(f"\nOutput files:")
            print(f"  - {self.output_file} (translated data)")
            print(f"  - translation_report.txt (translation report)")

        except Exception as e:
            print(f"\nError during translation: {str(e)}")
            import traceback
            traceback.print_exc()


def main():
    """主函数"""
    print("="*60)
    print("IoT Data Translator")
    print("Translate field IDs to human-readable names")
    print("="*60)

    # 配置
    CSV_FILE = "ph_ds_iot_data_all_202603181049.csv"
    PROTOCOL_FILE = "NET-DK-M1101-B_DK.xls"
    OUTPUT_FILE = "translated_data.csv"

    # 创建翻译器
    translator = DataTranslator(CSV_FILE, PROTOCOL_FILE, OUTPUT_FILE)

    # 执行翻译
    # 选项1: 翻译前10000行 (快速测试)
    # translator.run(sample_only=True)

    # 选项2: 翻译指定行数
    # translator.run(max_rows=50000)

    # 选项3: 翻译全部数据 (可能需要较长时间)
    translator.run()


if __name__ == "__main__":
    main()

