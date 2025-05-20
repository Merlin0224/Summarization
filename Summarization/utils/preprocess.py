"""
BBC 新闻摘要数据集预处理管道

功能：
1. 将原始文本数据转换为结构化CSV
2. 构建文本和摘要的词汇表
3. 创建文本预处理转换管道

原始数据预期结构：
/bbc
    /business
        001.txt
        002.txt
        ...
    /politics
    ...
"""

import os
import csv
import logging
from pathlib import Path
from glob import glob
from typing import Tuple, Iterable, Callable
import torchtext
torchtext.disable_torchtext_deprecation_warning()
from torch import nn
import torch
from torch import Tensor
from torchtext.data import get_tokenizer
from torchtext.vocab import Vocab, build_vocab_from_iterator
from torchtext.transforms import Sequential, AddToken, VocabTransform, Truncate, ToTensor
from typing import Tuple, Iterable, List  # 添加List导入


class TextNormalization(nn.Module):
    """自定义文本规范化模块"""

    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    def forward(self, text: str) -> List[str]:
        return [token.lower() for token in self.tokenizer(text)]

# 配置参数
DATA_DIR = Path("./data")
RAW_DATA_DIR = Path("./data/bbc")  # 假设原始数据在项目根目录的bbc文件夹中
TRAIN_CSV = DATA_DIR / "bbc_news_summary.csv"
MAX_TEXT_SEQ_LEN = 256  # 文章最大长度
MAX_SUMMARY_SEQ_LEN = 64  # 摘要最大长度
SEPARATOR = "\n\n"  # 文章与摘要分隔符

# 初始化日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_to_csv(input_dir: Path, output_csv: Path, separator: str = SEPARATOR) -> None:
    """将原始文本数据转换为结构化CSV文件

    Args:
        input_dir: 包含分类子目录的根目录（每个子目录包含*.txt文件）
        output_csv: 输出CSV路径
        separator: 文章与摘要分隔符

    Raises:
        FileNotFoundError: 当输入目录不存在时
    """
    if not input_dir.exists():
        raise FileNotFoundError(f"原始数据目录不存在: {input_dir}")

    # 递归获取所有文本文件
    file_paths = glob(str(input_dir / "**/*[0-9].txt"), recursive=True)

    # 自然排序（考虑多位数排序）
    file_paths.sort(key=lambda x: int(Path(x).stem))

    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["category", "file_path", "article", "summary"])
        writer.writeheader()

        for file_path in file_paths:
            try:
                file_path = Path(file_path)
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()

                # 分割文章和摘要
                article, summary = content.split(separator, 1)

                # 获取相对路径和分类信息
                rel_path = file_path.relative_to(input_dir)
                category = rel_path.parts[0]

                writer.writerow({
                    "category": category,
                    "file_path": str(rel_path),
                    "article": article.strip(),
                    "summary": summary.strip()
                })

            except ValueError as ve:
                logger.error(f"分割失败 {file_path}: 缺少分隔符 - {str(ve)}")
            except Exception as e:
                logger.error(f"处理 {file_path} 时出错: {str(e)}")
                continue


def build_vocabularies(csv_path: Path) -> Tuple[Vocab, Vocab]:
    # 确保使用相同的tokenizer
    tokenizer = get_tokenizer("basic_english")

    def yield_article_tokens():
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                yield [token.lower() for token in tokenizer(row["article"])
                       if not token.isdigit()]

    def yield_summary_tokens():
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                yield [token.lower() for token in tokenizer(row["summary"])
                       if not token.isdigit()]

    # 文章词汇表
    article_vocab = build_vocab_from_iterator(
        yield_article_tokens(),
        min_freq=2,
        specials=["<pad>", "<unk>", "<sos>", "<eos>"]
    )
    article_vocab.set_default_index(article_vocab["<unk>"])

    # 摘要词汇表（独立构建）
    summary_vocab = build_vocab_from_iterator(
        yield_summary_tokens(),
        min_freq=1,  # 摘要词汇出现频率较低
        specials=["<pad>", "<unk>", "<sos>", "<eos>"]
    )
    summary_vocab.set_default_index(summary_vocab["<unk>"])

    return article_vocab, summary_vocab


def build_transform_pipeline(vocab: Vocab, max_seq_len: int) -> Callable[[str], Tensor]:
    """构建文本预处理转换管道

    Args:
        vocab: 对应的词汇表
        max_seq_len: 序列最大长度

    Returns:
        预处理函数：str -> Tensor
    """
    tokenizer = get_tokenizer("basic_english")

    return Sequential(
        TextNormalization(tokenizer),  # 替换lambda为自定义模块
        AddToken(token="<sos>", begin=True),
        AddToken(token="<eos>", begin=False),
        Truncate(max_seq_len=max_seq_len),
        VocabTransform(vocab),
        ToTensor(dtype=torch.long)
    )

def main():
    # 准备数据目录
    DATA_DIR.mkdir(exist_ok=True)

    # 转换原始数据为CSV
    if not TRAIN_CSV.exists():
        logger.info("正在转换原始数据为CSV...")
        try:
            convert_to_csv(RAW_DATA_DIR, TRAIN_CSV)
            logger.info(f"数据已保存至 {TRAIN_CSV}")
        except Exception as e:
            logger.error(f"CSV转换失败: {str(e)}")
            return

    # 构建词汇表
    try:
        article_vocab, summary_vocab = build_vocabularies(TRAIN_CSV)
        logger.info("\n词汇表统计:")
        logger.info(f"文章词汇表大小: {len(article_vocab)}")
        logger.info(f"摘要词汇表大小: {len(summary_vocab)}")
        logger.info(f"文章前5词: {article_vocab.get_itos()[:5]}")
        logger.info(f"摘要前5词: {summary_vocab.get_itos()[:5]}")
    except Exception as e:
        logger.error(f"词汇表构建失败: {str(e)}")
        return

    # 示例转换流程
    try:
        article_transform = build_transform_pipeline(article_vocab, MAX_TEXT_SEQ_LEN)
        summary_transform = build_transform_pipeline(summary_vocab, MAX_SUMMARY_SEQ_LEN)

        sample_article = "New AI breakthrough in protein folding accelerates drug discovery."
        sample_summary = "Researchers develop novel AI model for predicting protein structures."

        print("\n示例预处理:")
        print(f"原始文章: {sample_article}")
        print(f"处理后: {article_transform(sample_article)}")
        print(f"原始摘要: {sample_summary}")
        print(f"处理后: {summary_transform(sample_summary)}")
    except Exception as e:
        logger.error(f"预处理示例失败: {str(e)}")


if __name__ == "__main__":
    main()