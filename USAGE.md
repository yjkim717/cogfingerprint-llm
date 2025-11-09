# News Generation 使用指南

## 快速开始

### 基本用法

```bash
# 生成 Level 1 的新闻文件（推荐先测试）
python generate_news.py --levels 1
```

## 参数说明

### `--levels` (必需)

指定要处理的级别（1, 2, 或 3）

**示例:**

```bash
# 处理 Level 1
python generate_news.py --levels 1

# 处理多个级别
python generate_news.py --levels 1 2 3

# 只处理 Level 2
python generate_news.py --levels 2
```

### `--workers` (可选)

并发处理的线程数，默认: 5

**建议值:**

- **保守**: 3-5 (避免 API 速率限制)
- **中等**: 8-10 (平衡速度和稳定性)
- **激进**: 15-20 (可能触发速率限制)

**示例:**

```bash
# 使用10个并发worker
python generate_news.py --levels 1 --workers 10

# 使用3个worker（更安全）
python generate_news.py --levels 1 --workers 3
```

### `--verbose` (可选)

显示详细的处理日志

**示例:**

```bash
# 显示每个文件的处理详情
python generate_news.py --levels 1 --verbose
```

## 使用场景

### 1. 首次测试

```bash
# 先用默认设置测试
python generate_news.py --levels 1 --workers 5
```

### 2. 快速处理

```bash
# 使用更多worker加速处理
python generate_news.py --levels 1 --workers 10
```

### 3. 批量处理所有级别

```bash
# 依次处理（避免API压力）
python generate_news.py --levels 1 --workers 10
python generate_news.py --levels 2 --workers 10
python generate_news.py --levels 3 --workers 10
```

### 4. 调试模式

```bash
# 显示详细日志，便于排查问题
python generate_news.py --levels 1 --workers 5 --verbose
```

## 输出说明

### 进度信息

```
Progress: [100/3685] (2.7%) | Rate: 0.85 files/s | ETA: 70.2 min
```

- `[100/3685]`: 已完成/总数
- `Rate`: 处理速度（文件/秒）
- `ETA`: 预计剩余时间（分钟）

### 统计信息

```
📊 Level 1 Summary:
   ✅ Processed: 3500
   ⏭️  Skipped: 185
   ❌ Failed: 0
   📏 Length Accuracy:
      Within ±5%: 3200/3500 (91.43%)
      Within ±10%: 3400/3500 (97.14%)
```

## 重要提示

1. **API 速率限制**: 如果遇到 429 错误，减少`--workers`数量
2. **自动跳过**: 已存在的文件会自动跳过，不会重复处理
3. **中断恢复**: 可以随时中断（Ctrl+C），已处理的文件会保留
4. **输出目录**: 默认输出到 `dataset/llm/news`

## 性能优化

### 根据 API 提供商调整 workers

- **DeepSeek**: 可尝试 10-15 workers
- **OpenRouter (Gemma/Llama)**: 建议 5-8 workers

### 预期处理时间

- **串行处理**: ~7 小时（3685 个文件）
- **5 workers**: ~1-2 小时
- **10 workers**: ~30-60 分钟（取决于 API 限制）

## 故障排除

### 问题: 遇到 429 速率限制错误

**解决**: 减少`--workers`数量

```bash
python generate_news.py --levels 1 --workers 3
```

### 问题: 处理速度太慢

**解决**: 增加`--workers`数量（注意 API 限制）

```bash
python generate_news.py --levels 1 --workers 10
```

### 问题: 想知道具体错误信息

**解决**: 使用`--verbose`选项

```bash
python generate_news.py --levels 1 --verbose
```

## 查看帮助

```bash
python generate_news.py --help
```
