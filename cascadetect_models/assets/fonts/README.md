# 中文字体文件目录

本目录用于存放用于显示中文文本的字体文件。

## 如何添加字体

1. 请将以下任一中文字体文件复制到本目录：
   - `simhei.ttf` - 黑体
   - `msyh.ttc` - 微软雅黑
   - `simsun.ttc` - 宋体

## 获取字体的方法

### Windows系统
可以从 `C:\Windows\Fonts` 目录复制字体文件到此目录：
```
copy C:\Windows\Fonts\simhei.ttf D:\MyCode\Yolo和CNN多模型级联检测\assets\fonts\
```

### 下载字体
也可以从以下链接下载免费可商用的中文字体：
1. 文泉驿微米黑：http://wenq.org/wqy2/index.cgi?MicroHei
2. 思源黑体：https://github.com/adobe-fonts/source-han-sans/tree/release

## 如果不添加字体会怎样？

如果没有添加中文字体，系统会尝试使用系统字体目录中的字体。如果找不到合适的字体，将使用默认字体，中文文本可能无法正常显示（显示为方块或问号）。 