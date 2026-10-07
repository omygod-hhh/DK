# IoT建筑冷热负荷预测 - 数据解析模块

## 📋 项目概述

本项目用于解析和处理IoT建筑传感器数据,为后续的冷热负荷和室内温度预测做准备。

## 📁 文件说明

### 核心脚本

1. **data_parser.py** - 主数据解析脚本
   - 读取2.12GB的CSV数据
   - 解析JSON字段并展开为结构化表格
   - 数据清洗和质量检查
   - 分块保存处理结果

2. **field_mapping.py** - 字段映射工具
   - 读取协议定义文件
   - 将数字字段ID映射到参数名称
   - 生成字段字典和报告

3. **data_explorer.py** - 数据探索可视化
   - 时间序列分析
   - 分布统计
   - 相关性矩阵
   - 生成可视化图表

4. **quick_parse_test.py** - 快速测试脚本
   - 处理前10000行数据
   - 用于快速验证和开发测试

### 数据文件

- `ph_ds_iot_data_all_202603181049.csv` - 原始IoT数据 (2.12GB)
- `NET-DK-M1101-B_DK.xls` - 设备通信协议定义

### 输出目录

```
processed_data/          - 解析后的结构化数据
reports/figures/         - 可视化图表
field_mapping_report.txt - 字段映射报告
field_dictionary.json    - 字段字典(JSON格式)
```

## 🚀 快速开始

### 前置要求

```bash
pip install pandas numpy matplotlib seaborn openpyxl xlrd
```

### 步骤1: 快速测试 (推荐)

先使用测试数据熟悉流程:

```bash
# 解析前10000行数据
python quick_parse_test.py

# 字段映射
python field_mapping.py

# 数据探索
python data_explorer.py
```

### 步骤2: 完整数据解析

```bash
# 解析完整的2.12GB数据(需要30-60分钟)
python data_parser.py
```

**注意**:
- 需要至少8GB可用内存
- 建议在后台运行
- 输出文件将保存在 `processed_data/` 目录

### 步骤3: 查看结果

```bash
# 查看数据统计报告
cat processed_data/data_statistics.txt

# 查看字段映射
cat field_mapping_report.txt

# 查看可视化图表
# 图表在 reports/figures/ 目录
```

## 📊 关键字段说明

### 温度相关字段

| 字段ID | 名称 | 说明 | 单位 |
|--------|------|------|------|
| 2028 | Ambient Temp | 环境温度 | °C |
| 2029 | Target Temp | 目标温度 | °C |
| 2030 | Display Temp | 显示/实际温度 | °C |
| 2054 | Total In Water Temp | 总进水温度 | °C |
| 2055 | Total Out Water Temp | 总出水温度 | °C |

### 控制相关字段

| 字段ID | 名称 | 说明 | 单位 |
|--------|------|------|------|
| 2032 | Compressor Speed | 压缩机转速 | RPM |
| 2033 | Valve 1 Position | 阀门1开度 | N |
| 2034 | Valve 2 Position | 阀门2开度 | N |
| 2031 | Bus Voltage | 母线电压 | V |

## 📈 数据探索结果

### 数据概况

- **时间跨度**: 2025-10-15 至 2025-12-19 (65天)
- **设备数量**: 28台
- **记录总数**: 186,003条 (合并后)
- **字段数量**: 1,053个数据字段

### 数据分布

- data_merged.csv: 186,003行 (完整合并数据)
- data_address_2001.csv: 469,958行 (主设备数据)
- data_address_2101.csv: 467,240行 (辅助设备数据)
- data_address_101.csv: 20,622行 (系统参数)
- 其他地址: 4,183行

### 数据质量

- JSON解析成功率: 100%
- 关键字段数据可用性: 100%
- 缺失值处理: 前向填充+后向填充
- 数据完整性: 优秀

## 🔧 常见问题

### Q1: 内存不足怎么办?

A: 使用分块处理模式:
```python
# 修改 data_parser.py,使用更小的chunk_size
parser = IoTDataParser(csv_path, chunk_size=5000)
```

### Q2: 中文显示乱码?

A: 这是Windows终端编码问题,不影响数据本身。
建议查看 `field_dictionary.json` 或生成的报告文件。

### Q3: 如何只处理特定地址类型的数据?

A: 修改解析脚本,过滤start_address:
```python
df_filtered = df[df['start_address'] == 2001]
```

### Q4: 处理速度太慢怎么办?

A: 方案:
1. 使用更快的SSD存储
2. 增加chunk_size
3. 只处理需要的字段(不展开所有字段)

## 📝 下一步工作

当前已完成数据解析和探索,下一步可以:

1. **特征工程** - 构建预测特征
2. **负荷估算** - 基于物理模型估算当前负荷
3. **温度预测** - 使用ML模型预测未来温度
4. **负荷预测** - 预测未来冷热需求
5. **可视化看板** - 实时监控和展示

详见 `DATA_PARSING_SUMMARY.md`

## 📞 技术支持

如有问题,请检查:
1. 数据文件路径是否正确
2. Python版本 >= 3.8
3. 依赖包已正确安装
4. 磁盘空间充足(至少10GB)

---

**最后更新**: 2026-10-06
**版本**: 1.0
