import torch
import numpy as np
from safetensors import safe_open
from safetensors.torch import save_file
import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re
from tqdm import tqdm
import matplotlib.pyplot as plt

class SwiGLUPruner:
    def __init__(self, input_dir: str, output_dir: str, pruning_ratio: float = 0.3, 
                 gate_threshold: float = 0.01, analyze_only: bool = False):
        """
        SwiGLU MLPの次元プルーニングを行うクラス
        
        Args:
            input_dir: 入力SafeTensorフォルダのパス
            output_dir: 出力フォルダのパス
            pruning_ratio: プルーニング比率 (0.0 ~ 1.0)
            gate_threshold: ゲートが閉じていると判断する閾値
            analyze_only: 分析のみ実行（プルーニングしない）
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.pruning_ratio = pruning_ratio
        self.gate_threshold = gate_threshold
        self.analyze_only = analyze_only
        if not analyze_only:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_safetensors(self) -> Dict[str, torch.Tensor]:
        """SafeTensorファイルを読み込む"""
        tensors = {}
        
        # すべての.safetensorsファイルを読み込む
        for file_path in self.input_dir.glob("*.safetensors"):
            with safe_open(file_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    tensors[key] = f.get_tensor(key)
        
        return tensors
    
    def identify_mlp_layers(self, tensors: Dict[str, torch.Tensor]) -> Dict[int, Dict[str, str]]:
        """MLP層を特定し、層番号ごとにグループ化"""
        mlp_layers = {}
        pattern = r'model\.layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)\.weight'
        
        for key in tensors.keys():
            match = re.match(pattern, key)
            if match:
                layer_idx = int(match.group(1))
                proj_type = match.group(2)
                
                if layer_idx not in mlp_layers:
                    mlp_layers[layer_idx] = {}
                
                mlp_layers[layer_idx][proj_type] = key
        
        return mlp_layers
    
    def analyze_gate_closure(self, gate_weight: torch.Tensor, threshold: float = None) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Gateの閉じ具合を分析
        
        Args:
            gate_weight: [hidden_dim, input_dim]
            threshold: ゲートが閉じていると判断する閾値
        
        Returns:
            gate_scores: 各次元のゲートスコア（低いほど閉じている）
            closed_mask: 閉じていると判断された次元のマスク
            analysis: 分析結果の辞書
        """
        if threshold is None:
            threshold = self.gate_threshold
            
        # Gate weightのL2ノルムを計算（各hidden次元について）
        gate_norms = torch.norm(gate_weight, p=2, dim=1)
        
        # Gate weightの絶対値の平均も計算
        gate_abs_mean = torch.abs(gate_weight).mean(dim=1)
        
        # 最大値の絶対値も確認（一つでも大きな値があれば開いている可能性）
        gate_max_abs = torch.abs(gate_weight).max(dim=1)[0]
        
        # 統合スコア：ノルムと平均絶対値と最大値を考慮
        # 全てが小さい場合のみ「閉じている」と判断
        gate_scores = gate_norms * gate_abs_mean * gate_max_abs
        
        # 閾値以下の次元を「閉じている」と判断
        # スコアを正規化して判断
        normalized_scores = gate_scores / gate_scores.max() if gate_scores.max() > 0 else gate_scores
        closed_mask = normalized_scores < threshold
        
        # 分析結果
        analysis = {
            'total_dims': len(gate_scores),
            'closed_dims': closed_mask.sum().item(),
            'closed_ratio': closed_mask.sum().item() / len(gate_scores),
            'score_mean': gate_scores.mean().item(),
            'score_std': gate_scores.std().item(),
            'score_min': gate_scores.min().item(),
            'score_max': gate_scores.max().item(),
            'score_percentiles': {
                '1%': torch.quantile(gate_scores, 0.01).item(),
                '5%': torch.quantile(gate_scores, 0.05).item(),
                '10%': torch.quantile(gate_scores, 0.10).item(),
                '25%': torch.quantile(gate_scores, 0.25).item(),
                '50%': torch.quantile(gate_scores, 0.50).item(),
                '75%': torch.quantile(gate_scores, 0.75).item(),
                '90%': torch.quantile(gate_scores, 0.90).item(),
                '95%': torch.quantile(gate_scores, 0.95).item(),
                '99%': torch.quantile(gate_scores, 0.99).item(),
            }
        }
        
        return gate_scores, closed_mask, analysis
    
    def find_optimal_threshold(self, gate_weight: torch.Tensor, target_pruning_ratio: float) -> float:
        """
        目標プルーニング率を達成するための最適な閾値を見つける
        
        Args:
            gate_weight: Gate weight tensor
            target_pruning_ratio: 目標プルーニング率
        
        Returns:
            optimal_threshold: 最適な閾値
        """
        gate_scores, _, _ = self.analyze_gate_closure(gate_weight, threshold=1.0)
        
        # スコアを正規化
        normalized_scores = gate_scores / gate_scores.max() if gate_scores.max() > 0 else gate_scores
        
        # 目標プルーニング率に対応するパーセンタイルを計算
        percentile = target_pruning_ratio * 100
        threshold = torch.quantile(normalized_scores, target_pruning_ratio).item()
        
        return threshold
    
    def prune_mlp_layer_by_gate(self, 
                                gate_weight: torch.Tensor,
                                up_weight: torch.Tensor, 
                                down_weight: torch.Tensor,
                                max_pruning_ratio: float = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Gate閉塞に基づいてMLP層をプルーニング
        
        Returns:
            pruned weights, mask, and analysis results
        """
        if max_pruning_ratio is None:
            max_pruning_ratio = self.pruning_ratio
            
        # まず現在の閾値で分析
        gate_scores, closed_mask, analysis = self.analyze_gate_closure(gate_weight)
        
        # 実際にプルーニングする割合を決定
        actual_pruning_ratio = min(analysis['closed_ratio'], max_pruning_ratio)
        
        # 実際のプルーニング率に基づいて閾値を調整
        if actual_pruning_ratio > 0:
            optimal_threshold = self.find_optimal_threshold(gate_weight, actual_pruning_ratio)
            gate_scores, closed_mask, analysis = self.analyze_gate_closure(gate_weight, optimal_threshold)
        
        # プルーニングする次元を決定（スコアが低い順）
        hidden_dim = gate_weight.shape[0]
        num_prune = int(hidden_dim * actual_pruning_ratio)
        num_keep = hidden_dim - num_prune
        
        # スコアでソートして、最も閉じている次元を特定
        _, indices = torch.sort(gate_scores, descending=False)  # 昇順（小さい値が先）
        prune_indices = indices[:num_prune]
        keep_indices = indices[num_prune:].sort()[0]  # 残す次元をソート
        
        # マスクを作成
        mask = torch.zeros(hidden_dim, dtype=torch.bool)
        mask[keep_indices] = True
        
        # プルーニング実行
        if num_prune > 0:
            pruned_gate = gate_weight[keep_indices, :]
            pruned_up = up_weight[keep_indices, :]
            pruned_down = down_weight[:, keep_indices]
        else:
            pruned_gate = gate_weight
            pruned_up = up_weight
            pruned_down = down_weight
        
        # 分析結果を更新
        analysis['actual_pruning_ratio'] = actual_pruning_ratio
        analysis['num_pruned'] = num_prune
        analysis['num_kept'] = num_keep
        
        print(f"  Gate closure analysis:")
        print(f"    Closed dims (threshold={self.gate_threshold:.4f}): {analysis['closed_dims']}/{hidden_dim} ({analysis['closed_ratio']*100:.1f}%)")
        print(f"    Actual pruning: {num_prune}/{hidden_dim} ({actual_pruning_ratio*100:.1f}%)")
        print(f"    Gate score range: [{gate_scores.min():.6f}, {gate_scores.max():.6f}]")
        
        return pruned_gate, pruned_up, pruned_down, mask, analysis
    
    def visualize_gate_distribution(self, all_analyses: Dict[int, Dict], output_dir: Path = None):
        """Gate閉塞の分布を可視化"""
        if output_dir is None:
            output_dir = self.output_dir if not self.analyze_only else Path(".")
        
        layers = sorted(all_analyses.keys())
        closed_ratios = [all_analyses[l]['closed_ratio'] for l in layers]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. 層ごとの閉塞率
        axes[0, 0].bar(layers, closed_ratios)
        axes[0, 0].set_xlabel('Layer Index')
        axes[0, 0].set_ylabel('Closed Gate Ratio')
        axes[0, 0].set_title('Gate Closure Ratio by Layer')
        axes[0, 0].axhline(y=self.pruning_ratio, color='r', linestyle='--', label=f'Max pruning ratio: {self.pruning_ratio}')
        axes[0, 0].legend()
        
        # 2. 閉塞率のヒストグラム
        axes[0, 1].hist(closed_ratios, bins=20, edgecolor='black')
        axes[0, 1].set_xlabel('Closed Gate Ratio')
        axes[0, 1].set_ylabel('Number of Layers')
        axes[0, 1].set_title('Distribution of Gate Closure Ratios')
        
        # 3. パーセンタイル分布（全層の平均）
        percentile_keys = ['1%', '5%', '10%', '25%', '50%', '75%', '90%', '95%', '99%']
        avg_percentiles = {}
        for key in percentile_keys:
            values = [all_analyses[l]['score_percentiles'][key] for l in layers]
            avg_percentiles[key] = np.mean(values)
        
        axes[1, 0].plot(percentile_keys, list(avg_percentiles.values()), 'o-')
        axes[1, 0].set_xlabel('Percentile')
        axes[1, 0].set_ylabel('Average Gate Score')
        axes[1, 0].set_title('Average Gate Score Percentiles Across Layers')
        axes[1, 0].set_yscale('log')
        
        # 4. 統計サマリー
        total_closed = sum([all_analyses[l]['closed_dims'] for l in layers])
        total_dims = sum([all_analyses[l]['total_dims'] for l in layers])
        avg_closed_ratio = total_closed / total_dims
        
        summary_text = f"""Global Statistics:
Total dimensions: {total_dims:,}
Total closed dimensions: {total_closed:,}
Average closure ratio: {avg_closed_ratio:.2%}
Min closure ratio: {min(closed_ratios):.2%}
Max closure ratio: {max(closed_ratios):.2%}

Recommended pruning ratio: {min(avg_closed_ratio, self.pruning_ratio):.2%}
(Based on gate closure analysis)"""
        
        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
                       fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightgray'))
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        # 保存
        output_path = output_dir / 'gate_analysis.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to {output_path}")
        plt.close()
        
        return avg_closed_ratio
    
    def process(self):
        """メイン処理"""
        print(f"Loading safetensors from {self.input_dir}")
        tensors = self.load_safetensors()
        
        print(f"Found {len(tensors)} tensors")
        
        # MLP層を特定
        mlp_layers = self.identify_mlp_layers(tensors)
        print(f"Found {len(mlp_layers)} MLP layers")
        
        # 分析結果を格納
        all_analyses = {}
        
        if self.analyze_only:
            print("\n=== ANALYSIS MODE - No pruning will be performed ===")
        
        # 新しいテンソル辞書
        new_tensors = {}
        
        # MLP以外のテンソルをコピー（分析モードではスキップ）
        if not self.analyze_only:
            mlp_keys = set()
            for layer_dict in mlp_layers.values():
                mlp_keys.update(layer_dict.values())
            
            for key, tensor in tensors.items():
                if key not in mlp_keys:
                    new_tensors[key] = tensor
        
        # 各MLP層を処理
        for layer_idx in sorted(mlp_layers.keys()):
            print(f"\nProcessing layer {layer_idx}")
            layer_dict = mlp_layers[layer_idx]
            
            # 必要な3つの投影が存在することを確認
            if not all(k in layer_dict for k in ['gate_proj', 'up_proj', 'down_proj']):
                print(f"  Skipping layer {layer_idx}: missing projections")
                continue
            
            gate_weight = tensors[layer_dict['gate_proj']]
            up_weight = tensors[layer_dict['up_proj']]
            down_weight = tensors[layer_dict['down_proj']]
            
            # データ型を保存
            dtype = gate_weight.dtype
            
            # Float32に変換して処理
            gate_weight = gate_weight.float()
            up_weight = up_weight.float()
            down_weight = down_weight.float()
            
            if self.analyze_only:
                # 分析のみ
                _, _, analysis = self.analyze_gate_closure(gate_weight)
                all_analyses[layer_idx] = analysis
                print(f"  Analysis complete: {analysis['closed_ratio']*100:.1f}% gates closed")
            else:
                # プルーニング実行
                pruned_gate, pruned_up, pruned_down, mask, analysis = self.prune_mlp_layer_by_gate(
                    gate_weight, up_weight, down_weight, self.pruning_ratio
                )
                all_analyses[layer_idx] = analysis
                
                # 元のデータ型に戻す
                new_tensors[layer_dict['gate_proj']] = pruned_gate.to(dtype)
                new_tensors[layer_dict['up_proj']] = pruned_up.to(dtype)
                new_tensors[layer_dict['down_proj']] = pruned_down.to(dtype)
        
        # 分析結果を可視化
        avg_closed_ratio = self.visualize_gate_distribution(all_analyses)
        
        print(f"\n=== SUMMARY ===")
        print(f"Average gate closure ratio across all layers: {avg_closed_ratio:.2%}")
        print(f"Requested max pruning ratio: {self.pruning_ratio:.2%}")
        
        if self.analyze_only:
            print(f"\nRecommended pruning ratio: {min(avg_closed_ratio, self.pruning_ratio):.2%}")
            print("To apply pruning, run without --analyze-only flag")
        else:
            # 保存
            print(f"\nSaving pruned model to {self.output_dir}")
            self.save_safetensors(new_tensors)
            print("Pruning completed successfully!")
    
    def save_safetensors(self, tensors: Dict[str, torch.Tensor], max_size_gb: float = 5.0):
        """SafeTensor形式で保存（大きい場合は分割）"""
        max_size_bytes = max_size_gb * 1024 * 1024 * 1024
        
        current_size = 0
        current_tensors = {}
        file_index = 0
        total_files = 1  # 予測されるファイル数
        
        # ファイル数を事前に計算
        total_size = sum(tensor.element_size() * tensor.numel() for tensor in tensors.values())
        total_files = max(1, int(np.ceil(total_size / max_size_bytes)))
        
        for key, tensor in tensors.items():
            tensor_size = tensor.element_size() * tensor.numel()
            
            if current_size + tensor_size > max_size_bytes and current_tensors:
                # 現在のバッチを保存
                output_path = self.output_dir / f"model-{file_index+1:05d}-of-{total_files:05d}.safetensors"
                save_file(current_tensors, output_path)
                print(f"  Saved {output_path.name} ({len(current_tensors)} tensors)")
                
                current_tensors = {}
                current_size = 0
                file_index += 1
            
            current_tensors[key] = tensor
            current_size += tensor_size
        
        # 残りを保存
        if current_tensors:
            if file_index == 0 and total_files == 1:
                output_path = self.output_dir / "model.safetensors"
            else:
                output_path = self.output_dir / f"model-{file_index+1:05d}-of-{total_files:05d}.safetensors"
            save_file(current_tensors, output_path)
            print(f"  Saved {output_path.name} ({len(current_tensors)} tensors)")


def main():
    parser = argparse.ArgumentParser(description="SwiGLU MLP Gate-based Dimension Pruning for SafeTensors")
    parser.add_argument("--input_dir", default="/home/client/Projects/llm/RWKV-Reka-Flash-Gen2", 
                       type=str, help="Input directory containing safetensor files")
    parser.add_argument("--output_dir", default="/home/client/Projects/llm/pruned", 
                       type=str, help="Output directory for pruned model")
    parser.add_argument("--pruning-ratio", type=float, default=0.15,
                       help="Maximum pruning ratio (0.0-1.0, default: 0.3 = 30%)")
    parser.add_argument("--gate-threshold", type=float, default=0.03,
                       help="Threshold for determining if a gate is closed (default: 0.01)")
    parser.add_argument("--analyze-only", action="store_true",
                       help="Only analyze gate closure without pruning")
    
    args = parser.parse_args()
    
    if not 0 < args.pruning_ratio < 1:
        raise ValueError("Pruning ratio must be between 0 and 1")
    
    pruner = SwiGLUPruner(
        args.input_dir, 
        args.output_dir, 
        args.pruning_ratio,
        args.gate_threshold,
        args.analyze_only
    )
    pruner.process()


if __name__ == "__main__":
    main()