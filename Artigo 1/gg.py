"""
Código adaptado de: https://github.com/cljosegfer/gg-neurips24/tree/main

Grafo de Gabriel (GG) com PyTorch, com foco em legibilidade e armazenamento de rótulos.

Principais recursos:
- build(X, y): calcula e armazena a adjacência do Grafo de Gabriel e, opcionalmente, os rótulos.
- bootstrap(X, y, batch_size, epochs): interseção de grafos em minibatches para estabilizar arestas.
- edges(): itera sobre as arestas (i, j) com i < j.
 - support_edges(): arestas de suporte (conectam vértices de classes opostas), requer y.
 - support_edges(): arestas de suporte (conectam vértices de classes opostas), requer y.

Compatibilidade: mantém uma classe "GG" com os métodos .torch(X) e .bootstrap(X, btsz, epochs)
para evitar quebra de código existente. Prefira usar a classe GabrielGraph diretamente.
"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple, Union

import numpy as np
import torch
import warnings

# Alias para evitar conflito com o método GG.torch nas decorações dentro da classe
_TORCH = torch


TensorLike = Union[np.ndarray, torch.Tensor]


class GabrielGraph:
    """Construtor do Grafo de Gabriel com PyTorch.

    Observação sobre a condição de Gabriel:
    Existe uma aresta entre i e j se, para todo k != i, j:
        d(i,k)^2 + d(j,k)^2 > d(i,j)^2
    Implementação vetorizada por linha, usando o mínimo sobre k.
    """

    def __init__(self, device: str = "cpu") -> None:
        """Parâmetros:
        - device: "cpu" ou "cuda" (se disponível)
        """
        if device == "cuda" and not torch.cuda.is_available():
            warnings.warn("CUDA não disponível; usando CPU.")
            device = "cpu"
        self.device = torch.device(device)

        self.X: Optional[torch.Tensor] = None  # (N, D) float32
        self.y: Optional[torch.Tensor] = None  # (N,) labels
        self.adj: Optional[torch.Tensor] = None  # (N, N) bool

    # ---------------------------
    # Gestão de devices (CPU/GPU)
    # ---------------------------
    def to_device(self, device: str) -> "GabrielGraph":
        """Move tensores armazenados (X, y, adj) para o device especificado."""
        dev = torch.device(device)
        if self.X is not None:
            self.X = self.X.to(dev)
        if self.y is not None:
            self.y = self.y.to(dev)
        if self.adj is not None:
            self.adj = self.adj.to(dev)
        self.device = dev
        return self

    def to_cpu(self, empty_cache: bool = True) -> "GabrielGraph":
        """Move dados armazenados para CPU e (opcionalmente) libera cache da GPU."""
        if self.X is not None:
            self.X = self.X.detach().cpu()
        if self.y is not None:
            self.y = self.y.detach().cpu()
        if self.adj is not None:
            self.adj = self.adj.detach().cpu()
        self.device = torch.device("cpu")
        if empty_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()
        return self

    @classmethod
    @torch.no_grad()
    def build_on_gpu_then_cpu(
        cls,
        X: TensorLike,
        y: Optional[TensorLike] = None,
    ) -> "GabrielGraph":
        """
        Constrói o grafo em CUDA (se disponível) e move tudo para CPU ao final,
        liberando o cache da GPU. Retorna a instância com tensores na CPU.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        g = cls(device=device)
        g.build(X, y)
        if device == "cuda":
            g.to_cpu(empty_cache=True)
        return g

    @classmethod
    @torch.no_grad()
    def bootstrap_on_gpu_then_cpu(
        cls,
        X: TensorLike,
        y: Optional[TensorLike] = None,
        batch_size: int = 256,
        epochs: int = 1,
        generator: Optional[np.random.Generator] = None,
    ) -> "GabrielGraph":
        """
        Executa bootstrap em CUDA (se disponível) e move tudo para CPU ao final,
        liberando o cache da GPU. Retorna a instância com tensores na CPU.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        g = cls(device=device)
        g.bootstrap(X=X, y=y, batch_size=batch_size, epochs=epochs, generator=generator)
        if device == "cuda":
            g.to_cpu(empty_cache=True)
        return g

    # ---------------------------
    # API Pública
    # ---------------------------
    @torch.no_grad()
    def build(self, X: TensorLike, y: Optional[TensorLike] = None) -> torch.Tensor:
        """Constrói a matriz de adjacência do Grafo de Gabriel.

        Parâmetros:
        - X: (N, D) np.ndarray ou torch.Tensor
        - y: (N,) ou (N,1), opcional

        Retorna:
        - adj: (N, N) bool, adj[i, j] == True indica aresta entre i e j
        """
        X_t = self._to_float_tensor_2d(X)
        adj = self._gabriel_adjacency(X_t)

        self.X = X_t
        self.adj = adj
        if y is not None:
            y_t = self._to_label_tensor_1d(y)
            if y_t.shape[0] != X_t.shape[0]:
                raise ValueError(
                    f"Dimensão inconsistente: X tem N={X_t.shape[0]}, y tem N={y_t.shape[0]}"
                )
            self.y = y_t

        return adj

    @torch.no_grad()
    def bootstrap(
        self,
        X: TensorLike,
        y: Optional[TensorLike] = None,
        batch_size: int = 256,
        epochs: int = 1,
        generator: Optional[np.random.Generator] = None,
    ) -> torch.Tensor:
        """Interseção de adjacências em minibatches ao longo de várias épocas.

        Parâmetros:
        - X: (N, D)
        - y: (N,), opcional; será armazenado se fornecido
        - batch_size: tamanho do lote (>0)
        - epochs: número de épocas
        - generator: RNG NumPy opcional para reprodutibilidade

        Retorna:
        - adjb: (N, N) bool
        """
        if batch_size <= 0:
            raise ValueError("batch_size deve ser > 0")

        X_np = self._ensure_numpy_2d(X)
        N = X_np.shape[0]
        idx = np.arange(N)
        rng = generator if generator is not None else np.random.default_rng()

        # Começa com tudo True e intersecta (AND) com subgrafos por lote
        adjb = torch.ones((N, N), dtype=torch.bool, device=self.device)

        for _ in range(epochs):
            rng.shuffle(idx)
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                idx_batch = idx[start:end]
                X_batch = X_np[idx_batch, :]

                adj_batch = self._gabriel_adjacency(self._to_float_tensor_2d(X_batch))
                adjb[np.ix_(idx_batch, idx_batch)] &= adj_batch

        # Garante simetria e diagonal falsa
        adjb = torch.logical_or(adjb, adjb.T)
        adjb.fill_diagonal_(False)

        # Armazena
        self.X = torch.as_tensor(X_np, dtype=torch.float32, device=self.device)
        self.adj = adjb
        if y is not None:
            self.y = self._to_label_tensor_1d(y)

        return adjb

    def edges(self) -> Iterable[Tuple[int, int]]:
        """Itera pelas arestas (i, j) com i < j. Requer que self.adj exista."""
        if self.adj is None:
            raise RuntimeError("Grafo ainda não foi construído. Use build(...) ou bootstrap(...).")
        adj_cpu = self.adj.detach().cpu()
        i_idx, j_idx = torch.triu(adj_cpu, diagonal=1).nonzero(as_tuple=True)
        for i, j in zip(i_idx.tolist(), j_idx.tolist()):
            yield (i, j)

    def support_adjacency(self) -> torch.Tensor:
        """
        Retorna uma adjacência booleana contendo apenas as arestas de suporte,
        ou seja, arestas entre vértices de classes diferentes (y[i] != y[j]).

        Requer que self.adj e self.y estejam definidos.
        """
        if self.adj is None:
            raise RuntimeError("Grafo ainda não foi construído. Use build(...) ou bootstrap(...).")
        if self.y is None:
            raise RuntimeError("Rótulos 'y' não estão disponíveis. Passe y em build(...) ou bootstrap(...).")
        y = self.y.view(-1)
        opp = y.unsqueeze(1) != y.unsqueeze(0)  # (N, N) bool
        supp = torch.logical_and(self.adj, opp)
        supp = torch.logical_or(supp, supp.T)
        supp = supp.clone()
        supp.fill_diagonal_(False)
        return supp

    def support_edges(self) -> Iterable[Tuple[int, int]]:
        """Itera pelas arestas de suporte (i, j) com i < j. Requer que y esteja armazenado."""
        supp = self.support_adjacency()
        i_idx, j_idx = torch.triu(supp.detach().cpu(), diagonal=1).nonzero(as_tuple=True)
        for i, j in zip(i_idx.tolist(), j_idx.tolist()):
            yield (i, j)

    def support_adjacency(self) -> torch.Tensor:
        """
        Retorna uma matriz de adjacência booleana contendo apenas as arestas de suporte,
        isto é, arestas entre vértices de classes diferentes (y[i] != y[j]).

        Requer que self.adj e self.y estejam definidos.
        """
        if self.adj is None:
            raise RuntimeError("Grafo ainda não foi construído. Use build(...) ou bootstrap(...).")
        if self.y is None:
            raise RuntimeError("Rótulos 'y' não estão disponíveis. Passe y em build(...) ou bootstrap(...).")
        y = self.y.view(-1)
        # Máscara de classes opostas
        opp = y.unsqueeze(1) != y.unsqueeze(0)  # (N, N) bool
        supp = torch.logical_and(self.adj, opp)
        # Garante simetria e diagonal falsa (por robustez)
        supp = torch.logical_or(supp, supp.T)
        supp = supp.clone()
        supp.fill_diagonal_(False)
        return supp

    def support_edges(self) -> Iterable[Tuple[int, int]]:
        """Itera pelas arestas de suporte (i, j) com i < j. Requer y armazenado."""
        supp = self.support_adjacency()
        i_idx, j_idx = torch.triu(supp.detach().cpu(), diagonal=1).nonzero(as_tuple=True)
        for i, j in zip(i_idx.tolist(), j_idx.tolist()):
            yield (i, j)

    # ---------------------------
    # Helpers internos
    # ---------------------------
    def _to_float_tensor_2d(self, X: TensorLike) -> torch.Tensor:
        X_t = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        if X_t.dim() != 2:
            raise ValueError(f"Esperado X com 2 dimensões (N, D), obtido shape={tuple(X_t.shape)}")
        return X_t

    def _ensure_numpy_2d(self, X: TensorLike) -> np.ndarray:
        if isinstance(X, torch.Tensor):
            X_np = X.detach().cpu().numpy()
        else:
            X_np = np.asarray(X)
        if X_np.ndim != 2:
            raise ValueError(f"Esperado X com 2 dimensões (N, D), obtido shape={X_np.shape}")
        return X_np

    def _to_label_tensor_1d(self, y: TensorLike) -> torch.Tensor:
        """Converte rótulos para tensor 1D no device (squeeze (N,1)->(N,))."""
        y_t = torch.as_tensor(y, device=self.device)
        if y_t.dim() == 2 and y_t.shape[1] == 1:
            y_t = y_t.squeeze(1)
        if y_t.dim() != 1:
            raise ValueError(f"Esperado y com 1 dimensão (N,), obtido shape={tuple(y_t.shape)}")
        return y_t

    @torch.no_grad()
    def _gabriel_adjacency(self, X: torch.Tensor) -> torch.Tensor:
        """Calcula a adjacência do Grafo de Gabriel para pontos X (N, D)."""
        N = X.shape[0]
        if N == 0:
            return torch.zeros((0, 0), dtype=torch.bool, device=self.device)

        # Distâncias ao quadrado
        F = torch.cdist(X, X, p=2) ** 2  # (N, N)
        F.fill_diagonal_(float("inf"))

        adj = torch.zeros((N, N), dtype=torch.bool, device=self.device)

        for i in range(N - 1):
            # A[j, k] = F[i, k] + F[j, k] para j > i
            A = F[i] + F[i + 1 :]  # (N-i-1, N)

            # min_k (F[i,k] + F[j,k])
            idx_min = torch.argmin(A, dim=1)  # (N-i-1,)
            min_sum = A[torch.arange(A.shape[0], device=self.device), idx_min]

            # Condição: min_k(...) - F[i,j] > 0
            fij = F[i, i + 1 :]
            cond = (min_sum - fij) > 0

            adj[i, i + 1 :] = cond

        # Simetria e zera diagonal
        adj = torch.logical_or(adj, adj.T)
        adj.fill_diagonal_(False)
        return adj


