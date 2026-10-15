import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# Configurações Estéticas para a Dissertação
# ==========================================
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams.update({
    "font.family": "serif",
    "figure.autolayout": True,
    "axes.titlesize": 14,
    "axes.labelsize": 12
})

COLOR_NOISY = "#E63946"     # Vermelho (Entrada)
COLOR_ENHANCED = "#1D3557"  # Azul escuro (Modelo CFM)
COLOR_CLEAN = "#7F8C8D"     # Cinza (Gabarito/Teto)

def load_and_prep_data(csv_path="test_metrics.csv"):
    """Carrega os dados e remove amostras sem SNR identificado."""
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["meta/SNR_DB"])
    
    # Arredondar o SNR para agrupar corretamente nos gráficos de linha
    df["meta/SNR_DB"] = df["meta/SNR_DB"].round(0)
    return df

def plot_global_improvement(df, output_dir="."):
    """Gera Violinplots emparelhados com a linha de teto (Baseline Limpo) em formato 2x2."""
    
    # Mapeamento: "Nome de Exibição": ("Coluna Ruidosa", "Coluna CFM", "Coluna Limpa", [valor_minimo, valor_maximo])
    metrics_map = {
        "PESQ (WB)": ("slow/PESQ_WB_NOISY", "slow/PESQ_WB", "slow/PESQ_WB_CLEAN", [0, 5]),
        "STOI": ("slow/STOI_NOISY", "slow/STOI", "slow/STOI_CLEAN", [0, 1]),
        "UTMOS": ("slow/UTMOS_NOISY", "slow/UTMOS", "slow/UTMOS_CLEAN", [1, 5]),
        "SPK-SIM": ("slow/SPK_SIM_NOISY", "slow/SPK_SIM", "slow/SPK_SIM_CLEAN", [0, 1])
    }
    
    plot_data = []
    
    for display_name, (col_noisy, col_enh, col_clean, _) in metrics_map.items():
        # Dados Ruidosos
        temp_noisy = df[[col_noisy]].copy()
        temp_noisy.columns = ["Valor"]
        temp_noisy["Métrica"] = display_name
        temp_noisy["Condição"] = "Ruidoso"
        
        # Dados Aprimorados
        temp_enh = df[[col_enh]].copy()
        temp_enh.columns = ["Valor"]
        temp_enh["Métrica"] = display_name
        temp_enh["Condição"] = "CFM (Aprimorado)"
        
        plot_data.extend([temp_noisy, temp_enh])
        
    df_melted = pd.concat(plot_data, ignore_index=True)
    
    # Alterado para grid 2x2 e proporção de tela adequada
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten() # Achata a matriz 2x2 para o loop iterar de 0 a 3
    
    for ax, (metric, values) in zip(axes, metrics_map.items()):
        # Plota os Violinplots (Ruidoso vs Aprimorado)
        sns.violinplot(
            data=df_melted[df_melted["Métrica"] == metric],
            x="Condição", y="Valor", hue="Condição",
            palette=[COLOR_NOISY, COLOR_ENHANCED],
            ax=ax, width=0.5, inner="box"
        )
        ax.set_ylim(values[3][0], values[3][1])
        
        # Calcula e plota a linha tracejada do Áudio Limpo
        col_clean = values[2]
        if col_clean in df.columns:
            clean_mean = df[col_clean].mean()
            ax.axhline(
                y=clean_mean, color=COLOR_CLEAN, linestyle='--', 
                linewidth=2, label='Gabarito Limpo'
            )
            ax.legend(loc='lower right', fontsize=10)

        ax.set_title(f"Distribuição Global - {metric}")
        ax.set_xlabel("")
        ax.set_ylabel("Pontuação")
        
    # Garante espaçamento adequado entre os 4 painéis
    plt.tight_layout()
    # Atualizado o nome do arquivo para refletir o tipo de gráfico
    plt.savefig(f"{output_dir}/fig_violinplot_global.png", dpi=300)
    plt.close()

def plot_performance_by_snr(df, output_dir="."):
    """Gera Gráficos de Linha por SNR com o teto de qualidade (Baseline Limpo) e limites de eixo fixos."""
    
    # Mapeamento idêntico ao do Violinplot
    metrics_map = {
        "PESQ (WB)": ("slow/PESQ_WB_NOISY", "slow/PESQ_WB", "slow/PESQ_WB_CLEAN", [0, 5]),
        "STOI": ("slow/STOI_NOISY", "slow/STOI", "slow/STOI_CLEAN", [0, 1]),
        "UTMOS": ("slow/UTMOS_NOISY", "slow/UTMOS", "slow/UTMOS_CLEAN", [1, 5]),
        "SPK-SIM": ("slow/SPK_SIM_NOISY", "slow/SPK_SIM", "slow/SPK_SIM_CLEAN", [0, 1])
    }
    
    # Criando o grid 2x2
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for ax, (display_name, (col_noisy, col_enh, col_clean, ylims)) in zip(axes, metrics_map.items()):
        # Plot Ruidoso
        sns.lineplot(
            data=df, x="meta/SNR_DB", y=col_noisy, 
            ax=ax, marker='o', linestyle='--', color=COLOR_NOISY, label="Ruidoso", errorbar=None
        )
        
        # Plot CFM (Aprimorado)
        sns.lineplot(
            data=df, x="meta/SNR_DB", y=col_enh, 
            ax=ax, marker='s', linestyle='-', color=COLOR_ENHANCED, label="CFM (Aprimorado)", errorbar=None
        )
        
        # Linha do Áudio Limpo
        if col_clean in df.columns:
            clean_mean = df[col_clean].mean()
            ax.axhline(
                y=clean_mean, color=COLOR_CLEAN, linestyle='--', 
                linewidth=2, label="Gabarito Limpo"
            )
            
        # Aplicando os limites do eixo Y extraídos do dicionário
        ax.set_ylim(ylims[0], ylims[1])
        
        # Formatando textos do eixo
        ax.set_title(f"Evolução de {display_name} por Relação Sinal-Ruído")
        ax.set_xlabel("Relação Sinal-Ruído (dB)")
        ax.set_ylabel(f"Pontuação {display_name}")
        ax.legend(loc='lower right')
        
    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig_lineplot_snr.png", dpi=300)
    plt.close()

def plot_compression_vs_intelligibility(df, output_dir="."):
    """Gera Scatter Plot correlacionando reconstrução geométrica com perceptual."""
    plt.figure(figsize=(8, 6))
    
    sns.scatterplot(
        data=df, 
        x="sample/SI_SDR", 
        y="slow/PESQ_WB", 
        hue="meta/SNR_DB", 
        palette="viridis",
        alpha=0.7,
        edgecolor=None
    )
    
    plt.title("Relação Geométrica (SI-SDR) vs Percepção (PESQ)")
    plt.xlabel("SI-SDR de Saída (dB)")
    plt.ylabel("PESQ (WB) de Saída")
    
    plt.legend(title="SNR Original (dB)", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(f"{output_dir}/fig_scatter_sdr_pesq.png", dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    csv_file = "tb_logs/cfm-unet/version_0/test_results.csv"
    
    try:
        dados = load_and_prep_data(csv_file)
        
        print("Gerando Boxplots Globais com Limite Limpo...")
        plot_global_improvement(dados)
        
        print("Gerando Curvas de SNR Corrigidas com Limite Limpo...")
        plot_performance_by_snr(dados)
        
        print("Gerando Correlação SDR x PESQ...")
        plot_compression_vs_intelligibility(dados)
        
        print("Gráficos exportados com sucesso!")
        
    except FileNotFoundError:
        print(f"Erro: O arquivo '{csv_file}' não foi encontrado.")
