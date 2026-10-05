import unittest

from spam_filter import SpamFilter, HAM_SAMPLES, SPAM_SAMPLES, _tokens


class TestTokens(unittest.TestCase):
    def test_tokens(self):
        self.assertIn("免费", _tokens("免费领取红包"))
        self.assertIn("win", _tokens("WIN money"))


class TestSpamFilter(unittest.TestCase):
    def setUp(self):
        self.sf = SpamFilter()

    def test_spam_detected(self):
        r = self.sf.classify("恭喜您中奖了，点击链接免费领奖")
        self.assertTrue(r["is_spam"])
        self.assertGreater(r["spam_probability"], 0.5)

    def test_ham_detected(self):
        r = self.sf.classify("明天下午开会，记得带汇报材料")
        self.assertFalse(r["is_spam"])
        self.assertLess(r["spam_probability"], 0.5)

    def test_proba_in_range(self):
        for s in ["随便一句", "退款 验证 账户", "牛奶 鸡蛋 超市"]:
            p = self.sf.spam_proba(s)
            self.assertGreaterEqual(p, 0.0)
            self.assertLessEqual(p, 1.0)

    def test_fit_improves(self):
        self.sf.fit(ham=["内部通知 会议纪要"], spam=["澳门 赌博 博彩 网址"])
        p = self.sf.spam_proba("澳门赌博网址")
        self.assertGreater(p, 0.5)

    def test_threshold(self):
        r = self.sf.classify("中奖 领奖", threshold=0.99)
        self.assertIn("is_spam", r)


if __name__ == "__main__":
    unittest.main()
