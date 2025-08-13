import json
import pickle
from pathlib import Path
from collections import Counter, OrderedDict
from typing import Dict, Iterable, List, Optional, Tuple, Union
from typing_extensions import Self
import random
import numpy as np
import pandas as pd
import torch
import torchtext.vocab as torch_vocab
from torchtext.vocab import Vocab
from tqdm import tqdm
# select module
import logging

logger = logging.getLogger(__name__)


class GeneVocab(Vocab):
    """
    基因名称的词汇表。
    """

    def __init__(
        self,
        gene_list_or_vocab: Union[List[str], Vocab],
        specials: Optional[List[str]] = None,
        special_first: bool = True,
        default_token: Optional[str] = "<pad>",
    ) -> None:
        """
        初始化词汇表。
        注意：仅当从基因列表初始化时才添加特殊符号。

        Args:
            gene_list_or_vocab (List[str] or Vocab): 基因名列表或Vocab对象。
            specials (List[str]): 特殊符号列表。
            special_first (bool): 是否将特殊符号添加到词汇表的开始。
            default_token (str): 默认符号，如果"<pad>"在词汇表中，则默认设置为"<pad>"。
        """
        if isinstance(gene_list_or_vocab, Vocab):
            _vocab = gene_list_or_vocab  # 如果是Vocab对象，直接赋值
            if specials is not None:
                raise ValueError(
                    "receive non-empty specials when init from a Vocab object."
                )  # 如果是Vocab对象但提供了specials，则抛出异常
        elif isinstance(gene_list_or_vocab, list):
            _vocab = self._build_vocab_from_iterator(
                gene_list_or_vocab,
                specials=specials,
                special_first=special_first,
            )  # 如果是列表，构建词汇表
        else:
            raise ValueError(
                "gene_list_or_vocab must be a list of gene names or a Vocab object."
            )  # 如果不是Vocab对象也不是列表，则抛出异常
        super().__init__(_vocab.vocab)  # 调用超类的构造函数
        if default_token is not None and default_token in self:
            self.set_default_token(default_token)  # 设置默认符号

    @classmethod
    def from_file(cls, file_path: Union[Path, str]) -> Self:
        """
        从文件加载词汇表。文件应为pickle或json格式的标记到索引的映射。
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)  # 如果文件路径为字符串，则转换为Path对象
        if file_path.suffix == ".pkl":
            with file_path.open("rb") as f:
                vocab = pickle.load(f)  # 读取pickle文件
                return cls(vocab)  # 返回类的实例
        elif file_path.suffix == ".json":
            with file_path.open("r") as f:
                token2idx = json.load(f)  # 读取json文件
                return cls.from_dict(token2idx)  # 从字典创建类的实例
        else:
            raise ValueError(
                f"{file_path} is not a valid file type. "
                "Only .pkl and .json are supported."
            )  # 如果文件类型不是pkl或json，则抛出异常

    @classmethod
    def from_dict(
        cls,
        token2idx: Dict[str, int],  # 标记到索引的字典
        default_token: Optional[str] = "<pad>",  # 默认符号
    ) -> Self:
        """
        从字典加载词汇表。

        Args:
            token2idx (Dict[str, int]): 标记到索引的映射字典。
        """
        _vocab = cls([])  # 首先初始化一个空的词汇表

        # 将标记添加到词汇表中，GeneVocab要求连续的索引
        for t, i in sorted(token2idx.items(), key=lambda x: x[1]):
            _vocab.insert_token(t, i)  # 按索引顺序插入标记

        if default_token is not None and default_token in _vocab:
            _vocab.set_default_token(default_token)  # 设置默认符号

        return _vocab  # 返回构建好的词汇表实例

    def _build_vocab_from_iterator(
        self,
        iterator: Iterable,
        min_freq: int = 1,
        specials: Optional[List[str]] = None,
        special_first: bool = True,
    ) -> Vocab:
        """
        从迭代器构建词汇表。此函数是从torchtext.vocab.build_vocab_from_iterator修改的。
        原始函数总是将标记拆分为字符，这不是我们想要的。

        Args:
            iterator (Iterable): 用于构建词汇表的迭代器。必须产生标记的列表或迭代器。
            min_freq (int): 包含标记在词汇表中需要的最小频率。
            specials (List[str]): 要添加的特殊符号。提供的标记将保持其顺序。
            special_first (bool): 是否将特殊标记添加到开头。

        Returns:
            torchtext.vocab.Vocab: 一个`Vocab`对象
        """

        counter = Counter()
        counter.update(iterator)

        if specials is not None:
            for tok in specials:
                del counter[tok]

        sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[0])
        sorted_by_freq_tuples.sort(key=lambda x: x[1], reverse=True)
        ordered_dict = OrderedDict(sorted_by_freq_tuples)

        if specials is not None:
            if special_first:
                specials = specials[::-1]
            for symbol in specials:
                ordered_dict.update({symbol: min_freq})
                ordered_dict.move_to_end(symbol, last=not special_first)

        word_vocab = torch_vocab.vocab(ordered_dict, min_freq=min_freq)
        return word_vocab

    @property
    def pad_token(self) -> Optional[str]:
        """
        获取填充标记。
        """
        if getattr(self, "_pad_token", None) is None:
            self._pad_token = None
        return self._pad_token

    @pad_token.setter
    def pad_token(self, pad_token: str) -> None:
        """
        设置填充标记。不会将填充标记添加到词汇表中。

        Args:
            pad_token (str): 填充标记，应该在词汇表中。
        """
        if pad_token not in self:
            raise ValueError(f"{pad_token} is not in the vocabulary.")
        self._pad_token = pad_token

    def save_json(self, file_path: Union[Path, str]) -> None:
        """
        将词汇表保存为json文件。
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)
        with file_path.open("w") as f:
            json.dump(self.get_stoi(), f, indent=2)

    def set_default_token(self, default_token: str) -> None:
        """
        设置默认标记。

        Args:
            default_token (str): 默认标记。
        """
        if default_token not in self:
            raise ValueError(
                f"{default_token} is not in the vocabulary."
            )  # 如果默认标记不在词汇表中，抛出异常
        self.set_default_index(self[default_token])  # 设置默认标记的索引


def get_default_gene_vocab() -> GeneVocab:
    """
    获取默认的基因词汇表，包含基因符号和ID。
    """
    vocab_file = Path(__file__).parent / "default_gene_vocab.json"
    if not vocab_file.exists():
        logger.info(
            f"No existing default vocab, will build one and save to {vocab_file}"
        )
        # 如果没有现成的默认词汇表，就创建一个并保存
        return _build_default_gene_vocab(save_vocab_to=vocab_file)
    logger.info(f"Loading gene vocabulary from {vocab_file}")
    # 从文件加载基因词汇表
    return GeneVocab.from_file(vocab_file)


def _build_default_gene_vocab(
    download_source_to: str = "/tmp",
    save_vocab_to: Union[Path, str, None] = None,
) -> GeneVocab:
    """
    从HGNC基因符号构建默认的基因词汇表。

    Args:
        download_source_to (str): 下载源数据的目录。
        save_vocab_to (Path or str): 保存词汇表的路径。如果为None，
            则不保存词汇表。默认为None。
    """
    gene_collection_file = (
        Path(download_source_to) / "human.gene_name_symbol.from_genenames.org.tsv"
    )
    if not gene_collection_file.exists():
        # 如果文件不存在，则从url下载并保存文件
        url = (
            "https://www.genenames.org/cgi-bin/download/custom?col=gd_app_sym&"
            "col=md_ensembl_id&status=Approved&status=Entry%20Withdrawn&hgnc_dbtag"
            "=on&order_by=gd_app_sym_sort&format=text&submit=submit"
        )
        import requests

        r = requests.get(url)
        gene_collection_file.write_text(r.text)

    logger.info(f"Building gene vocabulary from {gene_collection_file}")
    # 从文件构建基因词汇表
    df = pd.read_csv(gene_collection_file, sep="\t")
    gene_list = df["Approved symbol"].dropna().unique().tolist()
    # 创建基因列表
    gene_vocab = GeneVocab(gene_list)  # 默认词汇表中没有特殊标记
    if save_vocab_to is not None:
        gene_vocab.save_json(Path(save_vocab_to))
        # 如果提供了保存路径，保存词汇表为json文件
    return gene_vocab
    # 返回构建的基因词汇表


def tokenize_batch_edit(
    data,  # 输入数据，形状为 (batch_size, n_features)，即每个样本有n_features个特征
    gene_ids,  # 基因ID数组，形状为 (n_features,)
    pad_token_id,
    max_len,
    target_length,
    pad_value: int = -2,
    append_cls: bool = False,  # 是否在每个样本前添加一个类别标识符（<cls>）
    all_nonzero_value_set_1: bool = True,
    cls_id: int = "<cls>",  # 类别标识符，默认为"<cls>"
) -> List[Tuple[Union[torch.Tensor, np.ndarray]]]:
    """
    对一批数据进行标记化处理，返回一个包含(基因ID, 表达量)元组的列表。

    参数:
        data (array-like): 输入数据，每行代表一个样本，每列代表一个基因的特征。
        gene_ids (array-like): 基因ID数组。
        return_pt (bool): 是否返回PyTorch的张量，默认为True。
        append_cls (bool): 是否在每个样本的基因ID和表达量数组前加上一个<cls>标识符，默认为True。
        include_zero_gene (bool): 是否包括表达量为0的基因，默认为False。
        cls_id (int): <cls>标识符的ID，默认为"<cls>"。
        mod_type (np.ndarray): 可选的，模态类型数组，用于标记每个基因的类型。
        cls_id_mod_type (int): <cls>标识符的模态类型。

    返回:
        list: 包含(基因ID, 表达量)元组的列表，对于每个非零基因表达量的样本。
    """

    for i in tqdm(range(len(data))): 


        values = np.array(data[i],dtype=np.int8)
        if all_nonzero_value_set_1:
            values[values > 0] = 1
        genes = np.array(gene_ids)
        
        num_tokens = len(genes)  # 更新num_tokens到当前基因数量
        

        
        # 初始化列表来存储每个patch的masked_values、mask_pos、和原始values
        patched_values = values
    
    
        # 对每个区间进行操作
        

        if append_cls:
            genes = np.insert(genes, 0, cls_id)  # 在基因ID数组前插入<cls>
            cls_value = np.zeros(target_length)    
            patched_values =np.vstack((cls_value[np.newaxis, :], patched_values))
            num_tokens += 1  # 更新num_tokens以包括CLS标识符


        if num_tokens < max_len:
            pad_length = max_len - num_tokens
            # 创建一个包含pad_token_id的数组
            padding = np.full((pad_length,), pad_token_id, dtype=genes.dtype)
            # padding位置为1
            # token + padding token
            genes = np.concatenate((genes, padding))
            patched_values_padding = np.full(
                    (pad_length, target_length), -2, dtype=patched_values.dtype
                )
            patched_values = np.concatenate((patched_values, patched_values_padding))
        
        data[i] = patched_values
 

    return np.array(data,dtype=np.int8),genes



