# -*- coding: utf-8 -*-
# @Time    : 2018/8/20 13:52
# @Author  : JWDUAN
# @Email   : 494056012@qq.com
# @File    : FilepathFilenameFileext.py
# @Software: PyCharm
import os


def filepath_filename_fileext(filename):
    (filepath,tempfilename) = os.path.split(filename)
    (shotname,extension) = os.path.splitext(tempfilename)
    return filepath,shotname,extension