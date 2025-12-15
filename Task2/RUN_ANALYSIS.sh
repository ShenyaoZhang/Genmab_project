#!/bin/bash
# 快速运行分析脚本
# 使用方法: bash RUN_ANALYSIS.sh

echo "================================================================================"
echo "Epcoritamab CRS 分析 - 快速启动脚本"
echo "================================================================================"
echo ""

# 确保在正确的目录
cd "$(dirname "$0")"
echo "当前目录: $(pwd)"
echo ""

# 方法1：运行简化版本（推荐）
echo "方法 1: 运行简化版分析脚本（推荐）"
echo "命令: python3 run_epcoritamab_analysis_simple.py"
echo ""
read -p "按Enter继续，或输入 's' 跳过方法1: " choice

if [ "$choice" != "s" ]; then
    echo ""
    echo "开始运行..."
    echo "================================================================================"
    python3 run_epcoritamab_analysis_simple.py
    echo ""
    echo "方法1完成！"
    echo "================================================================================"
else
    echo "跳过方法1"
fi

echo ""
echo "================================================================================"
echo "方法 2: 运行可扩展管道"
echo "命令: python3 run_survival_analysis.py --drug epcoritamab --adverse_event 'cytokine release syndrome'"
echo ""
read -p "是否运行方法2? (y/N): " choice2

if [ "$choice2" = "y" ] || [ "$choice2" = "Y" ]; then
    echo ""
    echo "开始运行可扩展管道..."
    echo "================================================================================"
    python3 run_survival_analysis.py \
        --drug epcoritamab \
        --adverse_event "cytokine release syndrome" \
        --output_dir output/epcoritamab_crs \
        --limit 1000
    echo ""
    echo "方法2完成！"
    echo "================================================================================"
else
    echo "跳过方法2"
fi

echo ""
echo "================================================================================"
echo "分析完成！"
echo ""
echo "生成的文件："
echo "  - requirement2_epcoritamab_crs_analysis_data.csv"
echo "  - requirement2_epcoritamab_crs_km_curve.png"
echo "  - requirement2_epcoritamab_crs_risk_stratification.png"
echo "  - requirement2_epcoritamab_crs_clinical_report.txt"
echo "================================================================================"

