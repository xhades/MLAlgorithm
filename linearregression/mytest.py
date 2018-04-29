# !/usr/bin/env python
# -*-coding:utf-8-*-

"""
@author: xhades
@Date: 2018/3/26

"""


class MyTest(object):
    def __init__(self, name):
        self.name = name

    def test(self):
        return self


if __name__ == '__main__':
    t = MyTest("aa")
    a = t.test()
    print(a)