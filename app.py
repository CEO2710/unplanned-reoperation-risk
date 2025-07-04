import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split
from io import BytesIO

# 设置中文字体（确保SHAP图中的中文正常显示）
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 页面配置
st.set_page_config(
    page_title="Unplanned Reoperation Risk Prediction",
    page_icon="⚕️",
    layout="wide"
)

# 变量定义映射（英文）
VAR_DEFINITIONS = {
    "SEX": {0: "Female", 1: "Male"},
    "ASA scores": {0: "ASA < 3", 1: "ASA ≥ 3"},
    "tumor location": {
        1: "Off-axis AND Supracerebellar",
        2: "Intra-axis AND Supracerebellar",
        3: "Off-axis AND Subcerebellar",
        4: "Intra-axis AND Subcerebellar"
    },
    "Benign or malignant": {0: "Benign", 1: "Malignant"},
    "Admitted to NICU": {0: "No NICU", 1: "Admitted to NICU"},
    "Duration of surgery": {0: "≤4 hours", 1: ">4 hours"},
    "diabetes": {0: "No diabetes", 1: "Diabetes"},
    "CHF": {0: "No CHF", 1: "CHF"},
    "Functional dependencies": {0: "No", 1: "Yes"},
    "mFI-5": {
        0: "Robust (mFI = 0)",
        1: "Pre-frail (mFI = 1)",
        2: "Frail (mFI = 2)",
        3: "Severely frail (mFI ≥ 3)",
        4: "Severely frail (mFI ≥ 3)"
    },
    "Type of tumor": {
        1: "Meningiomas",
        2: "Primary malignant brain tumors",
        3: "Metastatic brain tumor",
        4: "Acoustic neuroma",
        5: "Other"
    }
}

# 加载数据
@st.cache_data
def load_data():
    try:
        df = pd.read_excel("data/2222222.xlsx")
        return df
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()

# 训练模型
@st.cache_data
def train_model(df):
    X = df.drop("Unplanned reoperation", axis=1)
    y = df["Unplanned reoperation"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model, X.columns

# 主应用
def main():
    st.title("⚕️ Unplanned Reoperation Risk Prediction System")
    st.markdown("This application uses machine learning to predict the risk of unplanned reoperation based on patient characteristics.")
    
    df = load_data()
    model, feature_names = train_model(df)
    
    # 侧边栏 - 变量详细定义
    with st.sidebar.expander("📚 Variable Definitions", expanded=False):
        st.markdown("### Categorical Variable Glossary")
        for feature, values in VAR_DEFINITIONS.items():
            st.markdown(f"**{feature}**")
            for code, desc in values.items():
                st.markdown(f"- `{code}` = {desc}")
            st.markdown("")
    
    # 患者信息表单
    st.subheader("🔍 Patient Risk Assessment")
    with st.form("prediction_form"):
        # 使用三列布局显示输入字段
        cols = st.columns(3)
        input_data = {}
        
        for i, feature in enumerate(feature_names):
            col = cols[i % 3]  # 循环使用三列
            
            # 获取变量定义和默认值
            feature_def = VAR_DEFINITIONS.get(feature, {})
            default_val = int(df[feature].mean())
            
            with col:
                # 分类变量使用下拉选择框
                if feature in VAR_DEFINITIONS:
                    options = list(feature_def.keys())
                    option_labels = [f"{code} - {feature_def[code]}" for code in options]
                    
                    # 查找默认值在选项中的索引
                    default_idx = options.index(default_val) if default_val in options else 0
                    
                    selected_code = st.selectbox(
                        label=feature,
                        options=options,
                        index=default_idx,
                        format_func=lambda x: feature_def.get(x, str(x)),
                        help=f"Options: {', '.join([f'{k} ({v})' for k, v in feature_def.items()])}"
                    )
                    input_data[feature] = selected_code
                else:
                    # 连续变量使用数字输入
                    min_val = int(df[feature].min())
                    max_val = int(df[feature].max())
                    
                    input_data[feature] = st.number_input(
                        label=feature,
                        min_value=min_val,
                        max_value=max_val,
                        value=default_val,
                        step=1
                    )
        
        # 提交按钮（居中显示）
        with st.columns(3)[1]:  # 中间列
            submitted = st.form_submit_button(
                "▶️ Predict Risk",
                use_container_width=True,
                type="primary"
            )
        
        if submitted:
            # 进行预测
            input_df = pd.DataFrame([input_data])
            prediction = model.predict(input_df)[0]
            proba = model.predict_proba(input_df)[0][1]
            
            # 显示预测结果
            st.subheader("📊 Prediction Results")
            if prediction == 1:
                st.error(f"⚠️ **Prediction: High Risk**")
                st.warning(f"Probability of unplanned reoperation: {proba:.2%}")
            else:
                st.success(f"✅ **Prediction: Low Risk**")
                st.info(f"Probability of unplanned reoperation: {proba:.2%}")
            
            # 生成SHAP力场图（不显示，直接提供下载）
            try:
                st.subheader("🔽 Download SHAP Explanation Plot")
                
                # 创建SHAP解释器
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(input_df)
                
                # 处理二分类SHAP值
                if isinstance(shap_values, list) and len(shap_values) == 2:
                    class_idx = 1  # 高风险类别
                    shap_value = shap_values[class_idx]
                    expected_value = explainer.expected_value[class_idx]
                else:
                    class_idx = 0
                    shap_value = shap_values
                    expected_value = explainer.expected_value
                
                # 生成SHAP力场图
                fig, ax = plt.subplots(figsize=(12, 6))
                shap.force_plot(
                    expected_value,
                    shap_value[0],
                    input_df.iloc[0],
                    feature_names=feature_names,
                    matplotlib=True,
                    show=False
                )
                plt.tight_layout()
                
                # 保存图像用于下载
                buf = BytesIO()
                plt.savefig(buf, format="png", dpi=300)  # 高DPI确保清晰度
                buf.seek(0)
                
                # 下载按钮
                st.download_button(
                    label="📥 Download SHAP Force Plot",
                    data=buf,
                    file_name="unplanned_reoperation_shap_plot.png",
                    mime="image/png"
                )
                
                plt.close(fig)  # 关闭图形以释放内存
                
                st.success("SHAP plot generated successfully! Click the button above to download.")
                
            except Exception as e:
                st.error(f"Failed to generate SHAP plot: {e}")
                st.markdown("Please try updating the SHAP library: `pip install shap --upgrade`")

if __name__ == "__main__":
    main()
