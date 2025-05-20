"""
train.py - 重构后的训练与推理程序
"""
import argparse
from model import Seq2SeqSummarizer
from utils.preprocess import build_transform_pipeline, MAX_TEXT_SEQ_LEN, MAX_SUMMARY_SEQ_LEN
import torch
from torch.utils.data import Dataset, DataLoader
import csv
from pathlib import Path
from utils.preprocess import build_vocabularies, build_transform_pipeline, MAX_TEXT_SEQ_LEN, MAX_SUMMARY_SEQ_LEN

class NewsSummaryDataset(Dataset):
    def __init__(self, csv_path, article_pipeline, summary_pipeline):
        self.data = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.data.append((
                    row['article'],
                    row['summary']
                ))
        self.article_pipeline = article_pipeline
        self.summary_pipeline = summary_pipeline

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        article, summary = self.data[idx]
        return (
            self.article_pipeline(article),
            self.summary_pipeline(summary)
        )


def train_model(data_dir, checkpoint_path, epochs, device):
    """ 完整的训练流程 """
    # 加载预处理组件

    train_csv = Path(data_dir) /'bbc_news_summary.csv'

    # 构建或加载词汇表
    article_vocab, summary_vocab = build_vocabularies(train_csv)

    # 构建预处理管道
    article_pipeline = build_transform_pipeline(article_vocab, MAX_TEXT_SEQ_LEN)
    summary_pipeline = build_transform_pipeline(summary_vocab, MAX_SUMMARY_SEQ_LEN)

    # 创建数据集和数据加载器
    dataset = NewsSummaryDataset(train_csv, article_pipeline, summary_pipeline)

    def collate_fn(batch):
        src_batch = [item[0].squeeze(0) for item in batch]
        tgt_batch = [item[1].squeeze(0) for item in batch]

        src_pad = torch.nn.utils.rnn.pad_sequence(
            src_batch, padding_value=article_vocab['<pad>']
        )
        tgt_pad = torch.nn.utils.rnn.pad_sequence(
            tgt_batch, padding_value=summary_vocab['<pad>']
        )
        return src_pad, tgt_pad

    dataloader = DataLoader(
        dataset, batch_size=32, shuffle=True, collate_fn=collate_fn
    )

    # 初始化模型
    model = Seq2SeqSummarizer(
        article_vocab_size=len(article_vocab),  # 新增文章词汇表参数
        summary_vocab_size=len(summary_vocab),  # 新增摘要词汇表参数
        embed_dim=256,
        hidden_dim=512
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=summary_vocab['<pad>'])

    # 训练循环
    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for src, tgt in dataloader:
            src = src.to(device)
            tgt = tgt.to(device)


            optimizer.zero_grad()

            # 准备解码器输入/输出
            decoder_input = tgt[:-1, :]  # 移除最后一个token
            decoder_output = tgt[1:, :]  # 移除第一个token

            # 前向传播
            outputs = model(src, decoder_input)

            # 计算损失
            loss = criterion(
                outputs.view(-1, outputs.size(-1)),
                decoder_output.reshape(-1)
            )

            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1} | Loss: {avg_loss:.4f}")

        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'article_vocab': article_vocab,
                'summary_vocab': summary_vocab
            }, checkpoint_path)


def run_inference(checkpoint_path, text, device):
    """ 完整的推理流程 """
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 初始化模型
    model = checkpoint['model_state_dict']['model_class'](
        article_vocab_size=len(checkpoint['article_vocab']),
        summary_vocab_size=len(checkpoint['summary_vocab']),
        embed_dim=256,
        hidden_dim=512
    ).to(device)
    model.eval()

    # 构建预处理管道
    article_pipeline = build_transform_pipeline(
        checkpoint['article_vocab'], MAX_TEXT_SEQ_LEN
    )

    # 预处理输入文本
    src = article_pipeline(text).unsqueeze(1).to(device)  # 添加batch维度

    # 生成摘要
    with torch.no_grad():
        # 编码器前向传播
        src_emb = model.encoder_emb(src)
        enc_output, (h, c) = model.encoder_lstm(src_emb)

        # 准备解码器初始输入
        tgt = torch.LongTensor([[checkpoint['summary_vocab']['<sos>']]]).to(device)
        summary_ids = []

        for _ in range(MAX_SUMMARY_SEQ_LEN):
            # 注意力计算
            energy = torch.tanh(model.attn(
                torch.cat([enc_output, h.repeat(enc_output.size(0), 1, 1)], dim=2)
            ))
            attn_weights = torch.softmax(energy, dim=0)
            context = torch.sum(enc_output * attn_weights, dim=0)

            # 解码步骤
            tgt_emb = model.decoder_emb(tgt)
            dec_input = torch.cat([tgt_emb, context.unsqueeze(0)], dim=2)
            dec_output, (h, c) = model.decoder_lstm(dec_input, (h, c))

            # 生成token
            output = model.fc_out(dec_output.squeeze(0))
            pred_token = output.argmax(1)

            # 遇到eos则停止
            if pred_token.item() == checkpoint['summary_vocab']['<eos>']:
                break

            summary_ids.append(pred_token.item())
            tgt = pred_token.unsqueeze(0)

    # 转换token到文本
    summary = ' '.join([
        checkpoint['summary_vocab'].get_itos()[token]
        for token in summary_ids
        if token not in {'<pad>', '<unk>'}
    ])

    print("\n生成摘要：")
    print(summary)


def main():
    parser = argparse.ArgumentParser(description="文本摘要训练程序")
    parser.add_argument('--mode', type=str,
                       choices=['train', 'predict'], default='train', help="运行模式：train/predict")
    parser.add_argument('--data_dir', type=str, default='./utils/data',
                       help="数据集目录路径")
    parser.add_argument('--checkpoint', type=str, default='checkpoint.pt',
                       help="模型检查点路径")
    parser.add_argument('--text', type=str,
                       help="预测模式下的输入文本")
    parser.add_argument('--epochs', type=int, default=100,help="训练轮数")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备：{device}")

    if args.mode == 'train':
        train_model(args.data_dir, args.checkpoint, args.epochs, device)
    elif args.mode == 'predict':
        if not args.text:
            raise ValueError("预测模式需要提供--text参数")
        run_inference(args.checkpoint, args.text, device)

if __name__ == "__main__":
    main()