---
license: mit
image:
  image-to-text:
    size_scale:
      - 100-10k
tags:
  - OCR
  - KIE
  - Doc Parsing
  - Multilingual
  - 通用识别
  - 多语言
  - 信息抽取
  - 文档解析
  - 公式识别

configs:
  - config_name: multi_scene_ocr
    data_files:
    - split: test
      path: "multi_scene_ocr/*/*.tsv"
  - config_name: multi_lan_ocr
    data_files:
    - split: test
      path: "multi_lan_ocr/*/*.tsv"
  - config_name: kie
    data_files:
    - split: test
      path: "kie/*/*.tsv"
  - config_name: doc_parsing
    data_files:
    - split: test
      path: "doc_parsing/*/*.tsv"
---



# CC-OCR

This is the Repository for CC-OCR Benchmark.

Dataset and evaluation code for the Paper "CC-OCR: A Comprehensive and Challenging OCR Benchmark for Evaluating Large Multimodal Models in Literacy".

<p align="center">
🚀 <a href="https://github.com/AlibabaResearch/AdvancedLiterateMachinery/tree/main/Benchmarks/CC-OCR">GitHub</a>&nbsp&nbsp | &nbsp&nbsp🤗 <a href="https://huggingface.co/datasets/wulipc/CC-OCR">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://www.modelscope.cn/datasets/Qwen/CC-OCR">ModelScope</a>&nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://arxiv.org/abs/2412.02210">Paper</a> &nbsp&nbsp | &nbsp&nbsp📗 <a href="https://zhibogogo.github.io/ccocr.github.io">Blog</a>

</p>

> Here is hosting the `tsv` version of CC-OCR data, which is used for evaluation in [VLMEvalKit](https://github.com/open-compass/VLMEvalKit). Please refer to our GitHub for more information.


## Benchmark Leaderboard
![](assets/images/cc_ocr_overall_performance.jpg)

| Model            | Multi-Scene Text Reading | Multilingual Text Reading | Document Parsing | Visual Information Extraction   | Total |
|------------------| --------------- | ------------- | ----------- | ----- |-------|
| Gemini-1.5-pro   | 83.25           | 78.97         | 62.37       | 67.28 | 72.97 |
| Qwen-VL-72B      | 77.95           | 71.14         | 53.78       | 71.76 | 68.66 |
| GPT-4o           | 76.40           | 73.44         | 53.30       | 63.45 | 66.65 |
| Claude3.5-sonnet | 72.87           | 65.68         | 47.79       | 64.58 | 62.73 |
| InternVL2-76B    | 76.92           | 46.57         | 35.33       | 61.60 | 55.11 |
| GOT              | 61.00           | 24.95         | 39.18       | 0.00  | 31.28 |
| Florence         | 49.24           | 49.70         | 0.00        | 0.00  | 24.74 |
| KOSMOS2.5        | 47.55           | 36.23         | 0.00        | 0.00  | 20.95 |
| TextMonkey       | 56.88           | 0.00          | 0.00        | 0.00  | 14.22 |

* The versions of APIs are GPT-4o-2024-08-06, Gemini-1.5-Pro-002, Claude-3.5-Sonnet-20241022, and Qwen-VL-Max-2024-08-09; 
* We conducted the all test around November 20th, 2024, please refer to our paper for more information.

## Benchmark Introduction
![](assets/images/cc_ocr_cover.jpg)

The CC-OCR benchmark is specifically designed for evaluating the OCR-centric capabilities of Large Multimodal Models. CC-OCR possesses a diverse range of scenarios, tasks, and challenges. CC-OCR comprises four OCR-centric tracks: multi-scene text reading, multilingual text reading, document parsing, and key information extraction. It includes 39 subsets with 7,058 full annotated images, of which 41% are sourced from real applications, being released for the first time.


The main features of our CC-OCR include:
* We focus on four OCR-centric tasks, namely `Multi-Scene Text Reading`, `Multilingual Text Reading`, `Document Parsing`, `Visual Information Extraction`;
* The CC-OCR covers fine-grained visual challenges (i.e., orientation-sensitivity, natural noise, and artistic text), decoding of various expressions, and structured inputs and outputs;


## Citation
If you find our work helpful, feel free to give us a cite.

```
@misc{yang2024ccocr,
      title={CC-OCR: A Comprehensive and Challenging OCR Benchmark for Evaluating Large Multimodal Models in Literacy}, 
      author={Zhibo Yang and Jun Tang and Zhaohai Li and Pengfei Wang and Jianqiang Wan and Humen Zhong and Xuejing Liu and Mingkun Yang and Peng Wang and Shuai Bai and LianWen Jin and Junyang Lin},
      year={2024},
      eprint={2412.02210},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.02210}, 
}
```

## License Agreement

The source code is licensed under the [MIT License](./LICENSE) that can be found at the root directory.

## Contact Us

If you have any questions, feel free to send an email to: wpf272043@alibaba-inc.com or xixing.tj@alibaba-inc.com.
