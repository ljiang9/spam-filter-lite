# spam-filter-lite

轻量**垃圾短信/邮件贝叶斯过滤器**：在朴素贝叶斯思路上做「垃圾 vs 正常」二分类，
内置少量正常/垃圾样本，开箱即可演示，输出垃圾概率。零第三方依赖。

## 功能简介

- 二分类：垃圾（spam）/ 正常（ham）；
- 内置 7 条正常 + 7 条垃圾中文样本，开箱即用；
- `spam_proba(text)` 输出 0~1 的垃圾概率（logistic 归一化）；
- `classify(text, threshold)` 给出判定与概率；
- 可用 `fit(ham=[...], spam=[...])` 追加自有语料再训练。

## 快速开始

环境：Python 3.10+，零依赖。

```bash
python3 spam_filter.py --check "恭喜您中奖了，点击链接立即领奖"
python3 spam_filter.py --check "明天下午三点开会，请准备汇报材料"
```

作为库：

```python
from spam_filter import SpamFilter
sf = SpamFilter()
print(sf.classify("免费领取 100 元红包"))
```

## 使用示例（真实命令）

```bash
$ python3 spam_filter.py --check "恭喜中奖，点击链接领奖"
判定：垃圾（垃圾概率 99.xx%）

$ python3 spam_filter.py --check "记得晚上买牛奶和鸡蛋"
判定：正常（垃圾概率 x.xx%）
```

## 无 API key 如何运行

纯本地统计模型，**不需要任何 API key**，不联网。直接运行即可。

## 目录结构

```
spam-filter-lite/
├── spam_filter.py            # 过滤器 + 内置样本 + CLI
├── tests/
│   └── test_spam_filter.py   # unittest
├── README.md
├── LICENSE
└── .gitignore
```

## 运行测试

```bash
python3 -m unittest discover -s tests -v
```

## 许可证

[MIT](./LICENSE) © 2026 ljiang9
