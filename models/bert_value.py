import math
import torch
import numpy as np
from torch import Tensor, nn
import torch.nn.functional as F
# from flash_attn.ops.fused_dense import FusedDense
from transformers.optimization import get_cosine_schedule_with_warmup
import lightning as pl
from lightning.pytorch.utilities import grad_norm
from models.bert import BERT
from torch.nn import LayerNorm,CrossEntropyLoss
from .grad_reverse import grad_reverse
def get_optimizers_for_lightning(
    params,
    learning_rate: float,
    adam_weight_decay: float,
    warmup_steps: int,
    max_steps: int,
):
    # 使用AdamW优化器，包含权重衰减
    optimizer = torch.optim.AdamW(
        params,
        lr=learning_rate,
        weight_decay=adam_weight_decay,
    )
    # 创建余弦退火调度器，包含预热步骤
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_steps,
    )
    # 返回优化器和调度器配置
    return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance by focusing on hard-to-classify examples.

    Args:
        alpha (float): Weighting factor for the focal loss.
        gamma (float): Focusing parameter to reduce the impact of easy examples.
    """
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        """
        Forward pass for focal loss computation.

        Args:
            inputs (tensor): Predicted logits.
            targets (tensor): Ground truth labels.

        Returns:
            tensor: Computed focal loss.
        """
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # Prevents nans when probability is 0
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()
class ExprHead(nn.Module):
    def __init__(self, d_model: int,peak_length: int):
        super().__init__()
        self.expr_fc1 = nn.Linear(d_model, d_model)
        self.expr_fc2 = nn.Linear(d_model, d_model)
        self.expr_fc3 = nn.Linear(d_model, peak_length)
        self.act = nn.LeakyReLU()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.expr_fc1.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.expr_fc2.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.expr_fc3.weight, gain=1 / math.sqrt(2))

    def forward(self, x: Tensor):
        """x is the output of the transformer, (batch, seq_len, d_model)"""
        x = self.act(self.expr_fc1(x))
        x = self.act(self.expr_fc2(x))
        pred_value = self.expr_fc3(x) # (batch, seq_len)
        return pred_value
class ExprDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        explicit_zero_prob: bool = False,
        use_batch_labels: bool = False,
        peak_length = 1,
    ):
        super().__init__()
        d_in = d_model * 2 if use_batch_labels else d_model
        self.fc = nn.Sequential(
            nn.Linear(d_in, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, peak_length),
        )
        self.explicit_zero_prob = explicit_zero_prob
        if explicit_zero_prob:
            self.zero_logit = nn.Sequential(
                nn.Linear(d_in, d_model),
                nn.LeakyReLU(),
                nn.Linear(d_model, d_model),
                nn.LeakyReLU(),
                nn.Linear(d_model, peak_length),
            )
    def forward(self, x: Tensor):
        """x is the output of the transformer, (batch, seq_len, d_model)"""
        pred_value = self.fc(x)  # (batch, seq_len)

        if not self.explicit_zero_prob:
            return dict(pred=pred_value)
        zero_logits = self.zero_logit(x) # (batch, seq_len)
        zero_probs = torch.sigmoid(zero_logits)
        return dict(pred=pred_value, zero_probs=zero_probs)

class BatchLabelEncoder(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx=None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)  # (batch, embsize)
        x = self.enc_norm(x)
        return x
class MLM_Decoder(nn.Module):
    def __init__(self, d_model: int,vocab_size: int):
        super().__init__()
        self.mlmDecoder = nn.Linear(d_model, vocab_size, bias=False) 
        self.mlmBias = nn.Parameter(torch.zeros(vocab_size))
        self.mlmDecoder.bias = self.mlmBias
        self.mlmDense = nn.Linear(d_model, d_model)
        self.transform_act_fn = nn.LeakyReLU()
        self.mlmLayerNorm = LayerNorm(d_model, eps=1e-12)
        self.act = nn.LeakyReLU()
    def forward(self, x: Tensor):
        """x is the output of the transformer, (batch, seq_len, d_model)"""
        mlm_hidden_state = self.mlmDense(x)
        mlm_hidden_state = self.transform_act_fn(mlm_hidden_state)
        mlm_hidden_state = self.mlmLayerNorm(mlm_hidden_state)
        mlm_scores = self.mlmDecoder(mlm_hidden_state)
        return mlm_scores
class AdversarialDiscriminator(nn.Module):
    """
    Discriminator for the adversarial training for batch correction.
    """

    def __init__(
        self,
        d_model: int,
        n_cls: int,
        nlayers: int = 3,
        activation: callable = nn.LeakyReLU,
        reverse_grad: bool = False,
    ):
        super().__init__()
        # module list
        self._decoder = nn.ModuleList()
        for i in range(nlayers - 1):
            self._decoder.append(nn.Linear(d_model, d_model))
            self._decoder.append(activation())
            self._decoder.append(nn.LayerNorm(d_model))
        self.out_layer = nn.Linear(d_model, n_cls)
        self.reverse_grad = reverse_grad

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, embsize]
        """
        if self.reverse_grad:
            x = grad_reverse(x, lambd=1.0)
        for layer in self._decoder:
            x = layer(x)
        return self.out_layer(x)


class ClsDecoder(nn.Module):
    """
    A neural network classifier with multiple layers for cell type classification.

    Args:
        embedding_dim (int): Dimension of the input embeddings.
        num_classes (int): Number of cell type classes to predict.
    """
    def __init__(self, embedding_dim, num_classes):
        super(ClsDecoder, self).__init__()
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(embedding_dim, 256)
        self.layer_norm1 = nn.LayerNorm(256)
        self.fc2 = nn.Linear(256, 128)
        self.layer_norm2 = nn.LayerNorm(128)
        self.fc3 = nn.Linear(128, num_classes)


    def forward(self, X):
        """
        Forward pass for the classifier.

        Args:
            X (tensor): Input features.
            calculate_loss (bool): Whether to calculate loss (default: False).
            target (tensor): Ground truth labels (required if calculate_loss is True).

        Returns:
            tensor: Predicted labels.
            tensor (optional): Computed loss if calculate_loss is True.
        """
        predicted_label = self.fc3(
            self.layer_norm2(self.gelu(self.fc2(self.dropout(self.layer_norm1(self.fc1(X))))))
        )
        
        return predicted_label
class RNA_Decoder(nn.Module):
    """
    A neural network classifier with multiple layers for cell type classification.

    Args:
        embedding_dim (int): Dimension of the input embeddings.
        num_classes (int): Number of cell type classes to predict.
    """
    def __init__(self, embedding_dim, num_classes):
        super(RNA_Decoder, self).__init__()
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(embedding_dim, 512)
        self.layer_norm1 = nn.LayerNorm(512)
        self.fc2 = nn.Linear(512, 1024)
        self.layer_norm2 = nn.LayerNorm(1024)
        self.fc3 = nn.Linear(1024, num_classes)


    def forward(self, X):
        """
        Forward pass for the classifier.

        Args:
            X (tensor): Input features.
            calculate_loss (bool): Whether to calculate loss (default: False).
            target (tensor): Ground truth labels (required if calculate_loss is True).

        Returns:
            tensor: Predicted labels.
            tensor (optional): Computed loss if calculate_loss is True.
        """
        predicted_label = self.fc3(
            self.layer_norm2(self.gelu(self.fc2(self.dropout(self.layer_norm1(self.fc1(X))))))
        )
        
        return predicted_label
class BertForPretrain(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        # 自动保存所有传递到 init 的超参数
        self.save_hyperparameters()
        # 配置
        self.config = config
        # 任务类型
        self.task = config.task

        # 主编码模型
        self.bert = BERT(config)
   
        if config.cell_type_annotation:
            self.cell_type_annotation = ClsDecoder(config.encoder_embed_dim,config.cell_type_number)
        if config.RNA_prediction:
            self.RNA_prediction = RNA_Decoder(config.encoder_embed_dim,config.hvg)
        if config.batch_correction:
            self.batch_encoder = BatchLabelEncoder(config.batch_number, config.encoder_embed_dim)
            self.decoder = ExprDecoder(d_model= config.encoder_embed_dim,explicit_zero_prob= True,use_batch_labels= config.batch_correction,peak_length=config.peak_length)
            self.grad_reverse_discriminator = AdversarialDiscriminator(
                config.encoder_embed_dim,
                n_cls=config.batch_number,
                reverse_grad=True,
            )
        self.mlm_scorer = ExprHead(config.encoder_embed_dim,config.peak_length)

        if config.mask_token:
            self.atac_mlm_decoder = MLM_Decoder(config.encoder_embed_dim,config.atac_vocab_size)
   

    def infer_atac_mlm(self,batch):
        """
        vision masking pretrain
        just return the masked position vision token predicates
        """
        """
        text_padding_position: just text padding postion, 1 means the padding postion, avoid performing attention on padding token indices; as attention_mask in bert
        attn_mask: attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8); attn_weights += attn_mask
            This argument indicates to the model which tokens should be attended to, and which should not.
        vision_masked_position: mask vision input embedding(VisionEmbedding), value 1 will mask the related vision position
        positions: position ids to get position embedding, if none, it will do automatically
        """
        # as the no, the length is fixed, no vision padding token
  
        gene_tokens = torch.from_numpy(self.config.gene_tokens).repeat(len(batch), 1,1).squeeze().to(batch.device) 
        
        padding_mask = gene_tokens.eq(self.config.pad_id)
        mlm_token_label = gene_tokens.clone() 
        original_tokenized_batch = batch.clone() 
     

    
        n_mask = int(self.config.context_length * 0.15)

        random_integers = torch.randperm(self.config.context_length)[:n_mask] 
        batch[:,random_integers] = -1  

        if self.config.mask_token:
            gene_tokens[:,random_integers] = self.config.mask_id

    
        outputs = self.bert(
            atac_tokens= gene_tokens,
            values_atac=batch.float() ,
            atac_padding_position=padding_mask,
            attn_mask=None,  # 注意力掩码
        )
        atac_feats = outputs["encoder_out"]
       
        # 如果存在语言掩码位置，则只获取这些位置的特征
        
        
        
        atac_feats_mlm = self.mlm_scorer(atac_feats)

        if self.config.mask_token:
            mlm_token_logits = self.atac_mlm_decoder(atac_feats)
         
            atac_feats = atac_feats_mlm[:,random_integers]

            return atac_feats, original_tokenized_batch.float()[:,random_integers],mlm_token_logits[:,random_integers],mlm_token_label[:,random_integers]
        
        
        atac_feats =atac_feats_mlm[:,random_integers]
        return atac_feats, original_tokenized_batch.float()[:,random_integers]
    

    def infer_atac_cell_type_annotation(self,batch_value,batch_label,embedding_type):
        """
        vision masking pretrain
        just return the masked position vision token predicates
        """
        """
        text_padding_position: just text padding postion, 1 means the padding postion, avoid performing attention on padding token indices; as attention_mask in bert
        attn_mask: attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8); attn_weights += attn_mask
            This argument indicates to the model which tokens should be attended to, and which should not.
        vision_masked_position: mask vision input embedding(VisionEmbedding), value 1 will mask the related vision position
        positions: position ids to get position embedding, if none, it will do automatically
        """
        # as the no, the length is fixed, no vision padding token
       
      
        gene_tokens = torch.from_numpy(self.config.gene_tokens).repeat(len(batch_value), 1,1).squeeze().to(self.device) 
        padding_mask = gene_tokens.eq(self.config.pad_id)
      
        outputs = self.bert(
            atac_tokens= gene_tokens.long(),
            values_atac=batch_value.float().to(self.device) ,
            atac_padding_position=padding_mask,
            attn_mask=None,  # 注意力掩码
        )
        if embedding_type == "cls":
            atac_feats = outputs["encoder_out"][:, 0, :]
        elif embedding_type == "avg_pool":
            atac_feats_without_cls = outputs["encoder_out"][
                    :, 1:, :
                ]  # 去除每个序列的第一个元素
                # 如果padding_mask也需要相应调整
            padding_mask_without_cls = padding_mask[
                :, 1:
            ]  # 同样去除每个序列的第一个元素
            padding_mask_without_cls_ = ~padding_mask_without_cls
            padding_mask_without_cls_ = torch.unsqueeze(
                padding_mask_without_cls_, dim=2
            )
            repr_wopadding = atac_feats_without_cls * padding_mask_without_cls_
            atac_feats = torch.sum(repr_wopadding, dim=1) / torch.unsqueeze(
                torch.sum(~padding_mask_without_cls, dim=1), dim=1
            )

        
     
        

      
        atac_feats = self.cell_type_annotation(atac_feats)
  
        
        return atac_feats,batch_label.to(self.device)  
    
    def infer_RNA_predict(self, batch_value, embedding_type):
        # 添加类型转换确保输入为 float32
        batch_value = batch_value.float() 
        """
        vision masking pretrain
        just return the masked position vision token predicates
        """
        """
        text_padding_position: just text padding postion, 1 means the padding postion, avoid performing attention on padding token indices; as attention_mask in bert
        attn_mask: attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8); attn_weights += attn_mask
            This argument indicates to the model which tokens should be attended to, and which should not.
        vision_masked_position: mask vision input embedding(VisionEmbedding), value 1 will mask the related vision position
        positions: position ids to get position embedding, if none, it will do automatically
        """
        # as the no, the length is fixed, no vision padding token
       
      
        gene_tokens = torch.from_numpy(self.config.gene_tokens).repeat(len(batch_value), 1,1).squeeze().to(self.device) 
        padding_mask = gene_tokens.eq(self.config.pad_id)
      
        outputs = self.bert(
            atac_tokens= gene_tokens.long(),
            rna_tokens=None,  # 文本token输入
            values_atac=batch_value.float().to(self.device) ,
            values_rna=None,
            atac_padding_position=padding_mask,
            rna_padding_position=None,  # 文本padding位置
            attn_mask=None,  # 注意力掩码
        )
        if embedding_type == "cls":
            atac_feats = outputs["encoder_out"][:, 0, :]
        elif embedding_type == "avg_pool":
            atac_feats_without_cls = outputs["encoder_out"][
                    :, 1:, :
                ]  # 去除每个序列的第一个元素
                # 如果padding_mask也需要相应调整
            padding_mask_without_cls = padding_mask[
                :, 1:
            ]  # 同样去除每个序列的第一个元素
            padding_mask_without_cls_ = ~padding_mask_without_cls
            padding_mask_without_cls_ = torch.unsqueeze(
                padding_mask_without_cls_, dim=2
            )
            repr_wopadding = atac_feats_without_cls * padding_mask_without_cls_
            atac_feats = torch.sum(repr_wopadding, dim=1) / torch.unsqueeze(
                torch.sum(~padding_mask_without_cls, dim=1), dim=1
            )

        
        # 如果存在语言掩码位置，则只获取这些位置的特征
        

      
       
        # atac_feats = self.mlm_scorer(atac_feats).reshape(len(atac_feats),self.config.context_length*256)
        RNA_feats = self.RNA_prediction(atac_feats)
  
        
        return RNA_feats  
    def infer_atac_batch_correction(self,batch_value,batch_label,embedding_type):
        """
        vision masking pretrain
        just return the masked position vision token predicates
        """
        """
        text_padding_position: just text padding postion, 1 means the padding postion, avoid performing attention on padding token indices; as attention_mask in bert
        attn_mask: attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8); attn_weights += attn_mask
            This argument indicates to the model which tokens should be attended to, and which should not.
        vision_masked_position: mask vision input embedding(VisionEmbedding), value 1 will mask the related vision position
        positions: position ids to get position embedding, if none, it will do automatically
        """
        # as the no, the length is fixed, no vision padding token
       
      
        gene_tokens = torch.from_numpy(self.config.gene_tokens).repeat(len(batch_value), 1,1).squeeze().to(self.device) 
        padding_mask = gene_tokens.eq(self.config.pad_id)
        original_tokenized_batch = batch_value.clone() 

        n_mask = int(self.config.context_length * 0.15)
     
        random_integers = torch.randperm(self.config.context_length)[:n_mask] 
        batch_value[:,random_integers] = -1  
        output = {}

        outputs = self.bert(
            atac_tokens= gene_tokens.long(),
            values_atac=batch_value.float().to(self.device) ,
            atac_padding_position=padding_mask,
            attn_mask=None,  # 注意力掩码
        )
        
        batch_emb = self.batch_encoder(batch_label.long())  # (batch, embsize)
        mlm_output = self.decoder(
            torch.cat(
                [
                    outputs["encoder_out"],
                    batch_emb.repeat(1, outputs["encoder_out"].shape[1], 1),
                ],
                dim=2,
            ),
      
        )
        output["mlm_output"] = mlm_output["pred"][:,random_integers]  # (batch, seq_len)
      
        output["mlm_zero_probs"] = mlm_output["zero_probs"][:,random_integers]

        if embedding_type == "cls":
            cell_emb = outputs["encoder_out"][:, 0, :]
        elif embedding_type == "avg_pool":
            atac_feats_without_cls = outputs["encoder_out"][
                    :, 1:, :
                ]  # 去除每个序列的第一个元素
                # 如果padding_mask也需要相应调整
            padding_mask_without_cls = padding_mask[
                :, 1:
            ]  # 同样去除每个序列的第一个元素
            padding_mask_without_cls_ = ~padding_mask_without_cls
            padding_mask_without_cls_ = torch.unsqueeze(
                padding_mask_without_cls_, dim=2
            )
            repr_wopadding = atac_feats_without_cls * padding_mask_without_cls_
            cell_emb = torch.sum(repr_wopadding, dim=1) / torch.unsqueeze(
                torch.sum(~padding_mask_without_cls, dim=1), dim=1
            )
        output["cell_emb"] = cell_emb

        cell_emb_normed = F.normalize(cell_emb, p=2, dim=1)
        cos_sim = torch.mm(cell_emb_normed, cell_emb_normed.t())  # (batch, batch)

        # mask out diagnal elements
        mask = torch.eye(cos_sim.size(0)).bool().to(cos_sim.device)
        cos_sim = cos_sim.masked_fill(mask, 0.0)
        # only optimize positive similarities
        cos_sim = F.relu(cos_sim)

        output["loss_ecs"] = torch.mean(1 - (cos_sim - 0.8) ** 2)

    
        output["dab_output"] = self.grad_reverse_discriminator(cell_emb)
        output["value"]=original_tokenized_batch[:,random_integers]

        
        return output
    def training_step(self, batch, batch_idx):
        """
        定义训练步骤，包括前向计算和损失计算
        """
        loss = None
        if "rnamlm" == self.task:
        # 视觉-文本掩码语言模型预训练
            mlm_logits, mlm_labels = self.infer_rna_mlm(**batch)
            loss = F.mse_loss(mlm_logits, mlm_labels, reduction="mean")
        if "atacmlm" == self.task:
        # 视觉-文本掩码语言模型预训练
            
            if self.config.mask_token:
                mlm_logits, mlm_labels,mlm_token_logits,mlm_token_label = self.infer_atac_mlm(batch)

                BCEWithLogitsLoss = nn.BCEWithLogitsLoss(reduction = 'mean')
                mlm_labels = mlm_labels.view(-1)
                non_negative_two_indices = torch.where(mlm_labels != -2)[0]
                loss_1 = BCEWithLogitsLoss(mlm_logits.view(-1)[non_negative_two_indices], mlm_labels[non_negative_two_indices])


                CE_loss =CrossEntropyLoss()
                loss_2 = CE_loss(mlm_token_logits.view(-1, self.config.atac_vocab_size).float(), mlm_token_label.view(-1).long())    
                  
                loss = loss_1 + loss_2
            elif self.config.cell_type_annotation:

                batch_value = batch[:, :-1,:].clone()
                batch_label = batch[:, -1, 0].reshape(-1, 1).clone()
                atac_logits,atac_labels  = self.infer_atac_cell_type_annotation(batch_value,batch_label,self.config.embedding_type)
     
                FocalCELoss = FocalLoss()
                loss =FocalCELoss(atac_logits, atac_labels.long().view(-1))

            elif self.config.RNA_prediction:
                # 添加显式类型转换
                ATAC_value = torch.stack(batch[0], dim=1).float() if isinstance(batch[0], list) else batch[0].float()
                RNA_logits = self.infer_RNA_predict(ATAC_value.to(self.device), self.config.embedding_type)
                
                RNA_value = torch.stack(batch[1]["value"], dim=1).float() if isinstance(batch[1]["value"], list) else batch[1]["value"].float()
                RNA_value = RNA_value.to(self.device)
                
             
                
                loss = F.mse_loss(RNA_logits, RNA_value, reduction="mean")
            
            elif self.config.batch_correction:

                batch_value = batch[:, :-1,:].clone()
                batch_label = batch[:, -1, 0].reshape(-1, 1).clone()
                output  = self.infer_atac_batch_correction(batch_value,batch_label,self.config.embedding_type)
                BCEWithLogitsLoss = nn.BCEWithLogitsLoss(reduction = 'mean')
                mlm_labels = output["value"]
                non_negative_two_indices = torch.where(mlm_labels.view(-1) != -2)[0]
                mlm_logits = output["mlm_output"]
                loss = BCEWithLogitsLoss(mlm_logits.float().view(-1)[non_negative_two_indices], mlm_labels.float().view(-1)[non_negative_two_indices])

              

            else:
                mlm_logits, mlm_labels= self.infer_atac_mlm(batch)
                # loss = F.mse_loss(mlm_logits, mlm_labels, reduction="mean")
                BCEWithLogitsLoss = nn.BCEWithLogitsLoss(reduction = 'mean')
                mlm_labels = mlm_labels.view(-1)
                non_negative_two_indices = torch.where(mlm_labels != -2)[0]
                loss = BCEWithLogitsLoss(mlm_logits.view(-1)[non_negative_two_indices], mlm_labels[non_negative_two_indices])
            

        # 记录训练损失
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
    def validation_step(self, batch, batch_idx):
        """
        定义训练步骤，包括前向计算和损失计算
        """
        loss = None
        if "rnamlm" == self.task:
        # 视觉-文本掩码语言模型预训练
            mlm_logits, mlm_labels = self.infer_rna_mlm(**batch)
            loss = F.mse_loss(mlm_logits, mlm_labels, reduction="mean")
        if "atacmlm" == self.task:
        # 视觉-文本掩码语言模型预训练
            
            if self.config.mask_token:
                mlm_logits, mlm_labels,mlm_token_logits,mlm_token_label = self.infer_atac_mlm(batch)

                BCEWithLogitsLoss = nn.BCEWithLogitsLoss(reduction = 'mean')
                mlm_labels = mlm_labels.view(-1)
                non_negative_two_indices = torch.where(mlm_labels != -2)[0]
                loss_1 = BCEWithLogitsLoss(mlm_logits.view(-1)[non_negative_two_indices], mlm_labels[non_negative_two_indices])


                CE_loss =CrossEntropyLoss()
                loss_2 = CE_loss(mlm_token_logits.view(-1, self.config.atac_vocab_size).float(), mlm_token_label.view(-1).long())    
                  
                loss = loss_1 + loss_2
            elif self.config.cell_type_annotation:
                batch_value = batch[:, :-1,:].clone()
                batch_label = batch[:, -1, 0].reshape(-1, 1).clone()
                atac_logits,atac_labels  = self.infer_atac_cell_type_annotation(batch_value,batch_label,self.config.embedding_type)
           
                CELoss = FocalLoss()
                loss = CELoss(atac_logits, atac_labels.long().view(-1))

            


            
            elif self.config.batch_correction:

                batch_value = batch[:, :-1,:].clone()
                batch_label = batch[:, -1, 0].reshape(-1, 1).clone()
                output  = self.infer_atac_batch_correction(batch_value,batch_label,self.config.embedding_type)
                BCEWithLogitsLoss = nn.BCEWithLogitsLoss(reduction = 'mean')
                mlm_labels = output["value"]
                non_negative_two_indices = torch.where(mlm_labels.view(-1) != -2)[0]
                mlm_logits = output["mlm_output"]
           
                loss = BCEWithLogitsLoss(mlm_logits.float().view(-1)[non_negative_two_indices], mlm_labels.float().view(-1)[non_negative_two_indices])
 
            else:
           
                mlm_logits, mlm_labels= self.infer_atac_mlm(batch)
           
   
                BCEWithLogitsLoss = nn.BCEWithLogitsLoss(reduction = 'mean')
                mlm_labels = mlm_labels.view(-1)
                non_negative_two_indices = torch.where(mlm_labels != -2)[0]
                loss = BCEWithLogitsLoss(mlm_logits.view(-1)[non_negative_two_indices], mlm_labels[non_negative_two_indices])
                
          

        # 记录训练损失
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss

    def configure_optimizers(self):
        """
        configure optimizers
        """
        return get_optimizers_for_lightning(
                self.parameters(),
                self.config.learning_rate,
                self.config.adam_weight_decay,
                self.config.num_warmup_steps,
                self.config.max_steps,
            )
    
    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        bert_norms = grad_norm(self.bert, norm_type=2)
        self.log_dict(bert_norms)
        
        mlm_norms = grad_norm(self.mlm_scorer, norm_type=2)
        self.log_dict(mlm_norms)
