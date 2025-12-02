import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

class RelPositionalEncoding(nn.Module):
    """상대적 위치 인코딩"""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        
        # Sinusoidal positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        # seq_len에 맞는 길이의 인코딩을 잘라서 반환
        return self.dropout(self.pe[:seq_len, :].to(device))

class RelMultiHeadAttn(nn.Module):
    """상대적 위치 인코딩을 사용하는 멀티헤드 어텐션"""
    def __init__(self, d_model: int, n_head: int, dropout: float):
        super().__init__()
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.scale = self.d_head ** -0.5

        self.qkv_net = nn.Linear(d_model, 3 * d_model)
        self.o_net = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Relative positional encoding을 위한 파라미터
        self.r_net = nn.Linear(d_model, d_model)
        self.r_w_bias = nn.Parameter(torch.randn(self.n_head, self.d_head))
        self.r_r_bias = nn.Parameter(torch.randn(self.n_head, self.d_head))
    
    def _rel_shift(self, x: torch.Tensor) -> torch.Tensor:
        # 효율적인 상대적 어텐션 계산을 위한 행렬 이동 트릭
        zero_pad = torch.zeros((x.size(0), x.size(1), x.size(2), 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)
        x_padded = x_padded.view(x.size(0), x.size(1), x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:, :].view_as(x)
        return x

    def forward(self, x: torch.Tensor, mem: Optional[torch.Tensor], pos_emb: torch.Tensor) -> torch.Tensor:
        # x: (batch, q_len, d_model) - 현재 청크
        # mem: (batch, m_len, d_model) - 이전 청크의 메모리
        # pos_emb: (k_len, d_model) - 위치 인코딩
        
        q_len, bsz = x.size(1), x.size(0)
        
        # 1. Q, K, V 계산
        cat = torch.cat([mem, x], dim=1) if mem is not None else x
        k_len = cat.size(1)
        
        qkv = self.qkv_net(cat)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        
        # 현재 청크의 Q만 사용
        q = q[:, -q_len:, :]
        
        # (batch, len, n_head, d_head) 형태로 변환
        q = q.view(bsz, q_len, self.n_head, self.d_head)
        k = k.view(bsz, k_len, self.n_head, self.d_head)
        v = v.view(bsz, k_len, self.n_head, self.d_head)
        
        # 2. 상대적 위치 임베딩 R 계산
        r = self.r_net(pos_emb) # (k_len, d_model)
        r = r.view(k_len, self.n_head, self.d_head) # (k_len, n_head, d_head)
        
        # 3. 어텐션 스코어 계산 (네 가지 텀)
        # 텀 1: content-content
        ac = torch.einsum('bqhd,bkhd->bhqk', q + self.r_w_bias, k)
        # 텀 2: content-position
        bd = torch.einsum('bqhd,khd->bhqk', q + self.r_r_bias, r)
        bd = self._rel_shift(bd)
        
        attn_score = (ac + bd) * self.scale
        
        # 4. 어텐션 가중치 및 최종 출력 계산
        attn_prob = F.softmax(attn_score, dim=-1)
        attn_prob = self.dropout(attn_prob)
        
        attn_vec = torch.einsum('bhqk,bkhd->bqhd', attn_prob, v)
        attn_vec = attn_vec.contiguous().view(bsz, q_len, self.n_head * self.d_head)
        
        output = self.o_net(attn_vec)
        return output

class TransformerXLLayer(nn.Module):
    """Transformer-XL의 단일 레이어"""
    def __init__(self, d_model: int, n_head: int, d_ff: int, dropout: float):
        super().__init__()
        self.attn = RelMultiHeadAttn(d_model, n_head, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mem: Optional[torch.Tensor], pos_emb: torch.Tensor) -> torch.Tensor:
        # Attention
        x_norm = self.norm1(x)
        attn_out = self.attn(x_norm, mem, pos_emb)
        x = x + self.dropout(attn_out)

        # Feed-forward
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = x + self.dropout(ffn_out)
        
        return x

class SignLanguageTransformerXL(nn.Module):
    """전체 Transformer-XL 모델 (메모리 관리 포함)"""
    def __init__(self, num_classes, input_features=274, model_dim=512, n_head=8, n_layers=6, d_ff=2048, dropout=0.1, mem_len=30):
        super().__init__()
        self.mem_len = mem_len
        self.n_layers = n_layers
        
        self.embedding = nn.Linear(input_features, model_dim)
        self.pos_emb = RelPositionalEncoding(model_dim, dropout)
        self.layers = nn.ModuleList([
            TransformerXLLayer(model_dim, n_head, d_ff, dropout) for _ in range(n_layers)
        ])
        self.classifier = nn.Linear(model_dim, num_classes)

    def _init_mems(self, bsz: int, device: torch.device) -> List[torch.Tensor]:
        # 비디오 시퀀스 시작 시 메모리를 초기화
        if self.mem_len > 0:
            return [torch.empty(bsz, 0, self.embedding.out_features, dtype=torch.float, device=device) for _ in range(self.n_layers)]
        else:
            return []

    def _update_mems(self, mems: List[torch.Tensor], hiddens: List[torch.Tensor]) -> List[torch.Tensor]:
        # 이전 메모리와 현재 출력을 합쳐 새로운 메모리 생성
        if not mems: return []
        
        new_mems = []
        for mem, hidden in zip(mems, hiddens):
            # hidden은 detach하여 그래디언트 전파를 막음
            cat = torch.cat([mem, hidden.detach()], dim=1)
            # 정해진 mem_len 만큼만 최신 정보를 유지
            new_mems.append(cat[:, -self.mem_len:, :])
        return new_mems

    def forward(self, x: torch.Tensor, mems: Optional[List[torch.Tensor]] = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        # x: (batch, chunk_size, 274)
        # mems: 각 레이어의 이전 메모리 리스트
        
        bsz, q_len = x.size(0), x.size(1)
        device = x.device

        if mems is None:
            mems = self._init_mems(bsz, device)
        
        # 1. 입력 임베딩
        x_emb = self.embedding(x)
        
        # 2. 위치 인코딩 생성 (메모리 길이 + 현재 청크 길이)
        m_len = mems[0].size(1) if mems else 0
        k_len = m_len + q_len
        pos_emb = self.pos_emb(k_len, device)
        pos_emb = pos_emb[-k_len:] # 필요한 길이만큼만 사용
        
        # 3. Transformer-XL 레이어 통과
        hiddens = [x_emb]
        for i, layer in enumerate(self.layers):
            mem = mems[i] if mems else None
            hiddens.append(layer(hiddens[-1], mem, pos_emb))
        
        # 4. 메모리 업데이트
        new_mems = self._update_mems(mems, hiddens[1:])
        
        # 5. 최종 출력 및 분류
        output = hiddens[-1]
        
        # # 현재 청크의 마지막 프레임의 출력만 사용하여 분류
        # # (또는 전체 프레임의 평균을 사용할 수도 있음)
        # last_frame_output = output[:, -1, :] 
        # logits = self.classifier(last_frame_output)

        bsz, q_len, d_model = output.shape
        output_flat = output.view(bsz * q_len, d_model)
        logits_flat = self.classifier(output_flat) # (batch * chunk_size, num_classes)
        logits = logits_flat.view(bsz, q_len, self.classifier.out_features)
        
        return logits, new_mems
