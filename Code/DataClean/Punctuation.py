import re

punctuation = set("!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~.,;《》？！“”‘’@#￥%…&×（）——+【】{};；●，。&～、|\s:：\n")
en_punctuation = ",().!;:"
zh_punctuation = "，（）。！；："

def cleanPunctuation(sentence: str) -> str:
    sentence = re.sub(' |　', '，', sentence) 

    ans = ''
    n = len(sentence)
    p = 0
    while p < n:
        ans += sentence[p]

        while p + 1 < n and sentence[p] in punctuation and sentence[p + 1] in punctuation:
            p += 1
        p += 1

    return ans