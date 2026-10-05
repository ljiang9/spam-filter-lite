#!/usr/bin/env python3
"""spam_filter —— 轻量垃圾短信/邮件贝叶斯过滤器（二分类）。

在朴素贝叶斯思路上做「垃圾 vs 正常」二分类：
  - 内置少量正常 / 垃圾样本，开箱即可演示；
  - predict_spam_proba(text) 返回垃圾概率（0~1）；
  - 可继续用 fit(corpus) 追加自有语料再训练；
  - 零第三方依赖。
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from collections import defaultdict

_WORD = re.compile(r"[a-z0-9]+")
_CJK = re.compile(r"[\u4e00-\u9fff]")

HAM_SAMPLES = [
    "明天下午三点开会，请准备好季度汇报材料",
    "记得晚上去超市买牛奶和鸡蛋",
    "项目周报已经发到邮箱，请查收附件",
    "周末一起去公园跑步吗？天气不错",
    "你的快递已到达小区驿站，请凭取件码领取",
    "孩子学校周五开家长会，记得参加",
    "这篇文章写得真好，推荐你读一读",
]
SPAM_SAMPLES = [
    "恭喜您中奖了！点击链接立即领取奖金",
    "免费领取 100 元红包，仅限今日",
    "你的账户异常，请立即点击验证否则冻结",
    "低价代购名牌手表，走私免税，联系微信",
    "重金招聘 在家兼职 日赚千元",
    "VIP 会员免费试用，马上扣费退订请回 T",
    "您好，您有一笔退税待领取，请填银行卡信息",
]


def _tokens(text: str) -> list[str]:
    toks = _WORD.findall(text.lower())
    buf: list[str] = []

    def flush():
        for i in range(len(buf) - 1):
            toks.append(buf[i] + buf[i + 1])
        buf.clear()

    for ch in text:
        if _CJK.match(ch):
            buf.append(ch)
        else:
            flush()
    flush()
    return toks


class SpamFilter:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.vocab: set[str] = set()
        self.ham_words: dict[str, int] = defaultdict(int)
        self.spam_words: dict[str, int] = defaultdict(int)
        self.ham_total = 0
        self.spam_total = 0
        self.ham_docs = 0
        self.spam_docs = 0
        self._train_builtin()

    def _add(self, text: str, is_spam: bool) -> None:
        for t in _tokens(text):
            self.vocab.add(t)
            if is_spam:
                self.spam_words[t] += 1
                self.spam_total += 1
            else:
                self.ham_words[t] += 1
                self.ham_total += 1
        if is_spam:
            self.spam_docs += 1
        else:
            self.ham_docs += 1

    def _train_builtin(self) -> None:
        for h in HAM_SAMPLES:
            self._add(h, is_spam=False)
        for s in SPAM_SAMPLES:
            self._add(s, is_spam=True)

    def fit(self, ham: list[str], spam: list[str]) -> None:
        for h in ham:
            self._add(h, is_spam=False)
        for s in spam:
            self._add(s, is_spam=True)

    def _log_odds(self, text: str) -> float:
        score = 0.0
        n = len(self.vocab)
        for t in _tokens(text):
            sw = (self.spam_words.get(t, 0) + self.alpha) / (self.spam_total + self.alpha * n)
            hw = (self.ham_words.get(t, 0) + self.alpha) / (self.ham_total + self.alpha * n)
            score += math.log(sw / hw)
        score += math.log((self.spam_docs + 1) / (self.ham_docs + 1))
        return score

    def spam_proba(self, text: str) -> float:
        logit = self._log_odds(text)
        return 1.0 / (1.0 + math.exp(-logit))

    def classify(self, text: str, threshold: float = 0.5) -> dict:
        p = self.spam_proba(text)
        return {
            "text": text,
            "spam_probability": round(p, 4),
            "is_spam": p >= threshold,
        }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="轻量垃圾短信/邮件贝叶斯过滤器")
    p.add_argument("--check", help="检查这段文本是否为垃圾")
    p.add_argument("--threshold", type=float, default=0.5)
    args = p.parse_args(argv)
    sf = SpamFilter()
    if args.check:
        r = sf.classify(args.check, threshold=args.threshold)
        tag = "垃圾" if r["is_spam"] else "正常"
        print(f"判定：{tag}（垃圾概率 {r['spam_probability']:.2%}）")
    else:
        print("用法：python3 spam_filter.py --check '要检查的短信内容'")
        demo = sf.classify("恭喜中奖，点击链接领奖")
        print(f"演示：垃圾概率 = {demo['spam_probability']:.2%} -> {'垃圾' if demo['is_spam'] else '正常'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
