---
name: Question
about: Ask a YOLOv6 question
title: ''
labels: question
assignees: ''

---

name: ❓ Question
description: Ask a YOLOv6 question
# title: " "
labels: [question]
body:
  - type: markdown
    attributes:
      value: |
        Thanks for your attention. We will try our best to solve your problem, but more concrete information is necessary to reproduce your problem.
  - type: checkboxes
    attributes:
      label: Before Asking
      description: >
        Please check and try following methods to solve it by yourself
      options:
        - label: >
           I have read the [README](https://github.com/meituan/YOLOv6/blob/main/README.md) carefully and installed YOLOv6 correctly;
           I have read the [tutorials for training your custom data](https://github.com/meituan/YOLOv6/blob/main/docs/Train_custom_data.md) carefully and organize my dataset correctly;
           I have pulled the latest code to run again and the problem still existed.
          required: true
 
  - type: checkboxes
    attributes:
      label: Search before asking
      description: >
        Please search the [issues](https://github.com/meituan/YOLOv6/issues) to see if a similar question already exists.
      options:
        - label: >
            I have searched the YOLOv6 [issues](https://github.com/meituan/YOLOv6/issues) and found no similar questions.
          required: true

  - type: textarea
    attributes:
      label: Question
      description: What is your question?
      placeholder: |
        💡 ProTip! Include as much information as possible (screenshots, logs, tracebacks etc.) to receive the most helpful response. (请仔细阅读上面的信息先进行问题排查，如果仍不能解决您的问题，请将问题尽可能地描述详细，以及提供相关命令、超参配置、报错日志等信息。)
    validations:
      required: true

  - type: textarea
    attributes:
      label: Additional
      description: Anything else you would like to share?
