import argparse
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os

def bootstrap_ci(diff, n_boot=10000, ci=95):
    """
    Calcula a média e o intervalo de confiança usando Bootstrap.
    """
    np.random.seed(42) # Para reprodutibilidade na dissertação
    n = len(diff)
    
    # Gera n_boot amostras com reposição e calcula a média de cada uma
    # Otimizado usando numpy vectorization para ficar super rápido
    indices = np.random.randint(0, n, size=(n_boot, n))
    boot_means = np.mean(diff[indices], axis=1)
    
    alpha = (100 - ci) / 2
    lower = np.percentile(boot_means, alpha)
    upper = np.percentile(boot_means, 100 - alpha)
    
    return np.mean(diff), lower, upper

def main():
    parser = argparse.ArgumentParser(description="Teste Estatístico para Métricas de Áudio (Poli-USP)")
    parser.add_argument("csv_base", help="Caminho para o CSV do modelo Base (ex: sem augmentation)")
    parser.add_argument("csv_novo", help="Caminho para o CSV do modelo Novo (ex: com augmentation)")
    parser.add_argument("--out_dir", default=".", help="Diretório para salvar os gráficos de boxplot")
    args = parser.parse_args()

    # 1. Carrega os dados
    print(f"Carregando Base: {args.csv_base}")
    print(f"Carregando Novo: {args.csv_novo}\n")
    df_base = pd.read_csv(args.csv_base)
    df_novo = pd.read_csv(args.csv_novo)

    if len(df_base) != len(df_novo):
        raise ValueError("Os CSVs têm tamanhos diferentes! Eles devem ter as mesmas amostras na mesma ordem.")

    # 2. Métricas de interesse (Apenas as do áudio gerado)
    metricas = [
        'sample/PESQ', 'sample/STOI', 'sample/SI_SDR', 'sample/MSE_AUDIO',
        'DNSMOS/MOS', 'DNSMOS/SIG', 'DNSMOS/BAK', 'DNSMOS/OVL'
    ]

    print("="*65)
    print(f"{'MÉTRICA':<18} | {'p-VALOR':<10} | {'MELHORA MÉDIA (IC 95%)':<25}")
    print("="*65)

    resultados_grafico = {}

    # 3. Executa os testes para cada métrica
    for metrica in metricas:
        if metrica not in df_base.columns or metrica not in df_novo.columns:
            continue
            
        m_base = df_base[metrica].values
        m_novo = df_novo[metrica].values
        
        # A diferença: se for positiva, o modelo novo é maior.
        # Exceção: MSE_AUDIO. Para MSE, menor é melhor.
        if metrica == 'sample/MSE_AUDIO':
            diff = m_base - m_novo  # Inverte a diferença: ganho é quando o erro diminui
        else:
            diff = m_novo - m_base

        # Trata casos em que as duas colunas são exatamente iguais
        if np.all(diff == 0):
            print(f"{metrica:<18} | {'IDÊNTICOS':<10} | 0.000 (0.000 a 0.000)")
            continue

        # Teste de Wilcoxon
        try:
            stat, p_value = stats.wilcoxon(m_base, m_novo)
            p_str = f"{p_value:.4e}" if p_value < 0.001 else f"{p_value:.4f}"
            if p_value < 0.05:
                p_str += " *" # Coloca um asterisco para chamar atenção
        except ValueError:
             p_str = "Erro stat"

        # Bootstrap
        mean_diff, ci_low, ci_high = bootstrap_ci(diff)
        
        # Log no console
        ic_str = f"{mean_diff:+.3f} ({ci_low:+.3f} a {ci_high:+.3f})"
        print(f"{metrica:<18} | {p_str:<10} | {ic_str}")

        resultados_grafico[metrica] = (m_base, m_novo, p_value)

    print("="*65)
    print("* p < 0.05 (Diferença Estatisticamente Significante)\n")

    # 4. Gera um Boxplot Lindo para a Dissertação
    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    fig.suptitle('Comparação de Distribuições: Modelo Base vs Modelo com Augmentation', fontsize=16)
    
    for ax, (metrica, (base, novo, p_val)) in zip(axes.flatten(), resultados_grafico.items()):
        # Remove outliers extremos apenas no gráfico para melhor visualização (opcional)
        ax.boxplot([base, novo], tick_labels=['Base', 'Augmentation'], showfliers=False,
                   patch_artist=True, boxprops=dict(facecolor="lightblue"))
        
        titulo = metrica.split('/')[-1]
        ax.set_title(titulo, fontweight='bold')
        
        # Adiciona o p-valor no gráfico
        if p_val < 0.05:
            ax.text(0.5, 0.95, f'p < 0.05*', horizontalalignment='center', 
                    verticalalignment='center', transform=ax.transAxes, color='green', fontweight='bold')
        else:
            ax.text(0.5, 0.95, f'p = {p_val:.3f}', horizontalalignment='center', 
                    verticalalignment='center', transform=ax.transAxes, color='red')
            
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    os.makedirs(args.out_dir, exist_ok=True)
    plot_path = os.path.join(args.out_dir, 'boxplot_estatistico.png')
    plt.savefig(plot_path, dpi=300)
    print(f"Gráfico Boxplot salvo com sucesso em: {plot_path}")

if __name__ == "__main__":
    main()